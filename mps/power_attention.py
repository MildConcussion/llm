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
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    if not (q.is_mps and k.is_mps and v.is_mps):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected q,k,v with shape [B,H,T,D]"
    b, h, t, d = q.shape
    assert k.shape == (b, h, t, d) and v.shape == (b, h, t, d), "q,k,v must have matching [B,H,T,D]"

    # Chunk default
    if chunk is None:
        # Reasonable default chunk; small enough to keep register/memory pressure low on MPS
        chunk = 128

    # Allocate state and output tensors
    c = int(Phi2FeatureMap(d, normalized=normalized).out_dim)
    out = torch.empty_like(q)
    state = torch.zeros(b, h, c, d, device=q.device, dtype=q.dtype)
    norm_state = torch.zeros(b, h, c, device=q.device, dtype=q.dtype)
    full = FullFusedChunk(d, d_tile=d_tile, normalized=normalized)

    BH = b * h
    n_chunks = (t + chunk - 1) // chunk

    # Optional debug: report potential copies if env enabled
    debug_copies = os.environ.get('MPS_DEBUG_COPIES', '0') == '1'

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk
        end = min(start + chunk, t)
        L = end - start

        q_ch = q[:, :, start:end]
        k_ch = k[:, :, start:end]
        v_ch = v[:, :, start:end]

        q_bh = q_ch.reshape(BH, L, d)
        k_bh = k_ch.reshape(BH, L, d)
        v_bh = v_ch.reshape(BH, L, d)

        state_bh = state.reshape(BH, c, d)
        norm_bh = norm_state.reshape(BH, c)
        out_bh = out[:, :, start:end].reshape(BH, L, d)
        decay_bh = torch.full((BH,), float(decay), device=q.device, dtype=q.dtype)

        if debug_copies:
            # Print minimal diagnostics about contiguity
            if not (q_bh.is_contiguous() and k_bh.is_contiguous() and v_bh.is_contiguous()):
                print(f"[mps_power_attention][debug] non-contiguous BH views at chunk {chunk_idx}")

        full.run(q_bh, k_bh, v_bh, state_bh, norm_bh, decay_bh, out_bh)

    return out


