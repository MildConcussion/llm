import os
import torch
import torch.nn.functional as F

from .custom_phi2 import FullFusedChunk, Phi2FeatureMap


def mps_power_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    chunk: int | None = None,
    decay: float = 1.0,
    d_tile: int = 32,
    normalized: bool = True,
):
    """Drop-in replacement for F.scaled_dot_product_attention using Power Attention p=2 on MPS.

    Constraints for fused path:
      - q,k,v: [B, H, T, D], float32, device='mps'
      - is_causal=True
      - attn_mask is None
      - dropout_p == 0
      - scale is None or 1.0 (ignored)

    Otherwise, falls back to torch.nn.functional.scaled_dot_product_attention.
    """

    # Fallback conditions
    unsupported = (
        (attn_mask is not None)
        or (dropout_p not in (0, 0.0))
        or (not is_causal)
        or (scale is not None and scale != 1.0)
    )

    if unsupported or (q.dtype != torch.float32) or (k.dtype != torch.float32) or (v.dtype != torch.float32):
        print(f"[mps_power_attention][warn] unsupported conditions: {unsupported}")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    if not (q.is_mps and k.is_mps and v.is_mps):
        print(f"[mps_power_attention][warn] non-MPS tensors: {q.device}, {k.device}, {v.device}")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected q,k,v with shape [B,H,T,D]"
    b, h, t, d = q.shape
    assert k.shape == (b, h, t, d) and v.shape == (b, h, t, d), "q,k,v must have matching [B,H,T,D]"

    # Chunk default with heuristic + env override
    if chunk is None:
        env_chunk = os.environ.get('MPS_CHUNK')
        if env_chunk:
            try:
                chunk = int(env_chunk)
            except Exception:
                chunk = None
        if chunk is None:
            if d <= 64:
                chunk = min(512, t)
            elif d <= 128:
                chunk = min(768, t)
            else:
                chunk = min(256, t)

            # Heuristic refinement by head count: if many heads, reduce per-dispatch working set
            # Example: for D<=128 default 768 -> 512 when H>=16
            if h >= 16:
                old_chunk = chunk
                if chunk >= 768:
                    chunk = min(512, t)
                elif chunk >= 512:
                    # take a conservative step down for small-D path
                    chunk = min(384, t)
                if chunk != old_chunk:
                    print(f"[mps_power_attention][auto] chunk adjusted for H>=16: {old_chunk} -> {chunk}")

    # Allocate state and output tensors
    c = int(Phi2FeatureMap(d, normalized=normalized).out_dim)
    out = torch.empty_like(q)
    state = torch.zeros(b, h, c, d, device=q.device, dtype=q.dtype)
    norm_state = torch.zeros(b, h, c, device=q.device, dtype=q.dtype)
    full = FullFusedChunk(d, d_tile=d_tile, normalized=normalized)

    BH = b * h
    # Proactive stability guard: auto-lower chunk on very large shapes unless disabled
    auto_verbose = os.environ.get('MPS_AUTO_VERBOSE', '1') == '1'
    auto_chunk_guard = os.environ.get('MPS_AUTO_CHUNK_GUARD', '1') == '1'
    if auto_chunk_guard:
        # Estimate working set sizes and reduce chunk if needed
        # Rule of thumb: keep BH*C*D under ~120M elements and L under 2048 per dispatch
        est_c = c
        bh = b * h
        max_elems = 120_000_000
        max_L = 2048
        if bh * est_c * d > max_elems or chunk > max_L:
            old_chunk = chunk
            # Reduce stepwise
            if chunk > 1024:
                chunk = 1024
            if bh * est_c * d > max_elems and chunk > 768:
                chunk = 768
            if bh * est_c * d > max_elems * 2 and chunk > 512:
                chunk = 512
            if auto_verbose and chunk != old_chunk:
                print(f"[mps_power_attention][auto] chunk lowered for stability: {old_chunk} -> {chunk} (BH={bh} C={est_c} D={d})")

    n_chunks = (t + chunk - 1) // chunk

    # Optional debug: report potential copies if env enabled
    debug_copies = os.environ.get('MPS_DEBUG_COPIES', '0') == '1'

    prev_auto_bh = None
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk
        end = min(start + chunk, t)
        L = end - start

        q_ch = q[:, :, start:end]
        k_ch = k[:, :, start:end]
        v_ch = v[:, :, start:end]

        # Ensure BH views are contiguous once per chunk to avoid repeated internal copies in kernels
        q_bh = q_ch.reshape(BH, L, d).contiguous()
        k_bh = k_ch.reshape(BH, L, d).contiguous()
        v_bh = v_ch.reshape(BH, L, d).contiguous()

        state_bh = state.reshape(BH, c, d)
        norm_bh = norm_state.reshape(BH, c)
        out_bh = out[:, :, start:end].reshape(BH, L, d)
        decay_bh = torch.full((BH,), float(decay), device=q.device, dtype=q.dtype)

        # Adaptive BH tiling if user hasn't pinned MPS_BH_BLOCK
        # If env is set to a nonzero value, we respect it fully
        env_bh = os.environ.get('MPS_BH_BLOCK')
        if not env_bh or env_bh == '0':
            auto_bh = 8 if L < 256 else (12 if L < 1024 else 16)
            # Only set when changed to reduce env churn
            if prev_auto_bh != auto_bh:
                os.environ['MPS_BH_BLOCK'] = str(auto_bh)
                prev_auto_bh = auto_bh
                print(f"[mps_power_attention][auto] BH tiling set: L={L} -> BH_BLOCK={auto_bh}")

        if debug_copies and (not (q_bh.is_contiguous() and k_bh.is_contiguous() and v_bh.is_contiguous())):
            print(f"[mps_power_attention][debug] non-contiguous BH views at chunk {chunk_idx}")

        # Optional kernel signpost profiling per-dispatch
        use_signpost = bool(int(os.environ.get('MPS_PROF', '0')))
        if use_signpost:
            with torch.mps.profiler.profile():
                full.run(q_bh, k_bh, v_bh, state_bh, norm_bh, decay_bh, out_bh)
        else:
            full.run(q_bh, k_bh, v_bh, state_bh, norm_bh, decay_bh, out_bh)

    return out


