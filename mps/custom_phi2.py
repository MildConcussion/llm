import os
import torch
import math
from .compiler import (
    compile_phi2_kernel,
    compile_phi2_kernel_normalized,
    compile_update_state_phi2,
    compile_update_state_phi2_bh,
    compile_ts_pow_intra,
    compile_query_state_phi2,
    compile_discumsum_state,
    compile_qk_pow2_intra_fused,
    compile_fused_update_state_phi2_bh,
    compile_fused_query_state_phi2_bh,
    compile_intra_inter_fused_chunk_bh,
    phi2_out_dim,
)


class Phi2FeatureMap:
    """Metal-backed symmetric power feature map (p=2) for attention.

    Expands last dimension (head_dim) to out_dim = head_dim * (head_dim + 1) // 2.
    Shapes:
      - Input:  [B, H, T, D]
      - Output: [B, H, T, C] where C = comb(D+1, 2)
    """

    def __init__(self, head_dim: int, normalized: bool = True):
        self.head_dim = int(head_dim)
        self.out_dim = int(phi2_out_dim(self.head_dim))
        self.lib = (
            compile_phi2_kernel_normalized(self.head_dim)
            if normalized
            else compile_phi2_kernel(self.head_dim)
        )
        self.normalized = normalized
        if self.normalized:
            # Precompute normalization factors for p=2: diag=1, off-diag=sqrt(2)
            factors = torch.ones(self.out_dim, dtype=torch.float32)
            off = 0
            for i in range(self.head_dim):
                off += 1  # skip diagonal (already 1)
                for j in range(i + 1, self.head_dim):
                    factors[off] = math.sqrt(2.0)
                    off += 1
            self.registered_factors = factors.to('mps')

    def get_pairs_and_factors(self):
        pairs = []
        factors = []
        off = 0
        for i in range(self.head_dim):
            pairs.append((i, i))
            factors.append(1.0)
            off += 1
            for j in range(i + 1, self.head_dim):
                pairs.append((i, j))
                factors.append(math.sqrt(2.0) if self.normalized else 1.0)
                off += 1
        pairs_tensor = torch.tensor(pairs, dtype=torch.int32, device='mps')
        factors_tensor = torch.tensor(factors, dtype=torch.float32, device='mps')
        return pairs_tensor, factors_tensor

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_mps, "Phi2FeatureMap requires MPS tensors"
        assert x.dtype == torch.float32, "Expect float32 for stable benchmarks"
        assert x.shape[-1] == self.head_dim
        assert x.is_contiguous() or x.stride(-1) == 1, "Input tensor should be contiguous or have last dim contiguous"

        b, h, t, d = x.shape
        num_tokens = b * h * t
        x_flat = x.contiguous().view(num_tokens, d)
        y_flat = torch.empty((num_tokens, self.out_dim), device=x.device, dtype=x.dtype)

        # Launch: one thread per token row
        if self.normalized:
            self.lib.phi2_expand_norm(
                x_flat,
                y_flat,
                self.registered_factors,
                num_tokens,
                self.head_dim,
                self.out_dim,
            )
        else:
            self.lib.phi2_expand(
                x_flat,
                y_flat,
                num_tokens,
                self.head_dim,
                self.out_dim,
            )

        return y_flat.view(b, h, t, self.out_dim)


class Phi2StateUpdater:
    """Metal-backed state update: state <- decay*state + sum_t phi_k[t]^T @ v[t]
    Also updates norm_state <- decay*norm_state + sum_t phi_k[t]
    """

    def __init__(self):
        self.lib = compile_update_state_phi2()
        self.lib_bh = compile_update_state_phi2_bh()

    def update(self, phi_k: torch.Tensor, v: torch.Tensor, state: torch.Tensor, norm_state: torch.Tensor, decay: float) -> None:
        # Shapes: phi_k [T, C], v [T, D], state [C, D], norm_state [C]
        assert phi_k.is_mps and v.is_mps and state.is_mps and norm_state.is_mps
        t, c = phi_k.shape
        t2, d = v.shape
        assert t == t2 and state.shape == (c, d) and norm_state.shape == (c,)
        self.lib.update_state_phi2(
            phi_k.contiguous(),
            v.contiguous(),
            state,
            norm_state,
            float(decay),
            int(t),
            int(c),
            int(d),
            threads=(c, d, 1),
        )

    def update_bh(self, phi_k_bh: torch.Tensor, v_bh: torch.Tensor, state_bh: torch.Tensor, norm_state_bh: torch.Tensor, decay_bh: torch.Tensor) -> None:
        # Shapes: phi_k_bh [BH, L, C], v_bh [BH, L, D], state_bh [BH, C, D], norm_state_bh [BH, C], decay_bh [BH]
        assert all(t.is_mps for t in (phi_k_bh, v_bh, state_bh, norm_state_bh, decay_bh))
        BH, L, C = phi_k_bh.shape
        BH2, L2, D = v_bh.shape
        assert BH == BH2 and L == L2 and state_bh.shape == (BH, C, D) and norm_state_bh.shape == (BH, C) and decay_bh.shape == (BH,)
        # Tile over BH to reduce kernel workload per launch
        # Allow env override for BH tiling (default retains previous 8)
        bh_env = int(os.environ.get('MPS_BH_BLOCK', '0'))
        bh_block = (bh_env if bh_env > 0 else (8 if BH >= 8 else BH))
        for start in range(0, BH, bh_block):
            end = min(start + bh_block, BH)
            sub_phi = phi_k_bh[start:end].contiguous()
            sub_v = v_bh[start:end].contiguous()
            sub_state = state_bh[start:end]
            sub_norm = norm_state_bh[start:end]
            sub_decay = decay_bh[start:end]
            sub_BH = end - start
            self.lib_bh.update_state_phi2_bh(
                sub_phi,
                sub_v,
                sub_state,
                sub_norm,
                sub_decay,
                int(sub_BH), int(L), int(C), int(D),
                threads=(sub_BH * C, 1, 1),
            )


class TSPowIntra:
    """Metal-backed tile intra-chunk compute: local_out and local_norm for causal φ2 attention."""

    def __init__(self):
        self.lib = compile_ts_pow_intra()

    def run(self, phi_q_bh: torch.Tensor, phi_k_bh: torch.Tensor, v_bh: torch.Tensor, out_bh: torch.Tensor, norm_bh: torch.Tensor) -> None:
        # Inputs: [BH,L,C], [BH,L,C], [BH,L,D]; outputs preallocated: [BH,L,D], [BH,L]
        assert all(t.is_mps for t in (phi_q_bh, phi_k_bh, v_bh, out_bh, norm_bh))
        BH, L, C = phi_q_bh.shape
        BH2, L2, C2 = phi_k_bh.shape
        BH3, L3, D = v_bh.shape
        assert BH == BH2 == BH3 and L == L2 == L3 and C == C2
        # Tile over BH to reduce per-launch workload
        # Allow env override for BH tiling (default retains previous 8)
        bh_env = int(os.environ.get('MPS_BH_BLOCK', '0'))
        bh_block = (bh_env if bh_env > 0 else (8 if BH >= 8 else BH))
        for start in range(0, BH, bh_block):
            end = min(start + bh_block, BH)
            sub_phi_q = phi_q_bh[start:end].contiguous()
            sub_phi_k = phi_k_bh[start:end].contiguous()
            sub_v = v_bh[start:end].contiguous()
            sub_out = out_bh[start:end]
            sub_norm = norm_bh[start:end]
            sub_BH = end - start
            self.lib.ts_pow_intra(
                sub_phi_q,
                sub_phi_k,
                sub_v,
                sub_out,
                sub_norm,
                int(sub_BH), int(L), int(C), int(D),
                threads=(sub_BH * L, 1, 1),
            )


class QKPow2IntraFused:
    """Metal-backed fused squared-dot intra-chunk compute using raw q,k (no explicit φ).

    Computes local_out and local_norm for causal attention with p=2: weight = (q·k)^2.
    """

    def __init__(self):
        self.lib = compile_qk_pow2_intra_fused()

    def run(self, q_bh: torch.Tensor, k_bh: torch.Tensor, v_bh: torch.Tensor, out_bh: torch.Tensor, norm_bh: torch.Tensor) -> None:
        # q_bh,k_bh: [BH,L,D], v_bh: [BH,L,Dv]
        assert all(t.is_mps for t in (q_bh, k_bh, v_bh, out_bh, norm_bh))
        BH, L, D = q_bh.shape
        BH2, L2, Dk = k_bh.shape
        BH3, L3, Dv = v_bh.shape
        assert BH == BH2 == BH3 and L == L2 == L3 and D == Dk

        # Tile by BH if requested via env
        bh_env = int(os.environ.get('MPS_BH_BLOCK', '0'))
        bh_block = (bh_env if bh_env > 0 else (8 if BH >= 8 else BH))
        for start in range(0, BH, bh_block):
            end = min(start + bh_block, BH)
            sub_q = q_bh[start:end].contiguous()
            sub_k = k_bh[start:end].contiguous()
            sub_v = v_bh[start:end].contiguous()
            sub_out = out_bh[start:end]
            sub_norm = norm_bh[start:end]
            sub_BH = end - start
            self.lib.qk_pow2_intra_fused(
                sub_q,
                sub_k,
                sub_v,
                sub_out,
                sub_norm,
                int(sub_BH), int(L), int(D), int(Dv),
                threads=(sub_BH * L, 1, 1),
            )


class QueryStatePhi2:
    """Metal-backed inter-chunk query x [state, norm_state] compute for φ2 attention.

    Computes out_bh and out_norm_bh without Python-side matmul/einsum.
    """

    def __init__(self):
        self.lib = compile_query_state_phi2()

    def run(
        self,
        phi_q_bh: torch.Tensor,     # [BH, L, C]
        state_bh: torch.Tensor,      # [BH, C, D]
        norm_state_bh: torch.Tensor, # [BH, C]
        out_bh: torch.Tensor,        # [BH, L, D], preallocated
        out_norm_bh: torch.Tensor,   # [BH, L], preallocated
    ) -> None:
        assert all(t.is_mps for t in (phi_q_bh, state_bh, norm_state_bh, out_bh, out_norm_bh))
        BH, L, C = phi_q_bh.shape
        BHs, Cs, D = state_bh.shape
        assert BH == BHs and C == Cs and out_bh.shape == (BH, L, D)
        assert norm_state_bh.shape == (BH, C) and out_norm_bh.shape == (BH, L)

        # Optional tiling via env var to influence per-launch workload; we keep 1D grid
        # and let the driver schedule; slicing by blocks can help on some shapes.
        bh_block = int(os.environ.get('MPS_BH_BLOCK', '0'))
        if bh_block and bh_block > 0 and BH > bh_block:
            for start in range(0, BH, bh_block):
                end = min(start + bh_block, BH)
                sub_phi_q = phi_q_bh[start:end].contiguous()
                sub_state = state_bh[start:end]
                sub_norm  = norm_state_bh[start:end]
                sub_out   = out_bh[start:end]
                sub_outn  = out_norm_bh[start:end]
                sub_BH = end - start
                self.lib.query_state_phi2(
                    sub_phi_q,
                    sub_state,
                    sub_norm,
                    sub_out,
                    sub_outn,
                    int(sub_BH), int(L), int(C), int(D),
                    threads=(sub_BH * L, 1, 1),
                )
        else:
            self.lib.query_state_phi2(
                phi_q_bh.contiguous(),
                state_bh,
                norm_state_bh,
                out_bh,
                out_norm_bh,
                int(BH), int(L), int(C), int(D),
                threads=(BH * L, 1, 1),
            )


class DiscumSumState:
    """Metal-backed discounted cumulative sum over chunked (state,norm_state).

    Inputs:
      - state_in:  [N, BH, C, D]
      - norm_in:   [N, BH, C]
      - lambda_nbh:[N, BH] per-chunk decay
    Outputs (preallocated):
      - state_out: [N, BH, C, D]
      - norm_out:  [N, BH, C]
    """

    def __init__(self):
        self.lib = compile_discumsum_state()

    def run(
        self,
        state_in: torch.Tensor,
        norm_in: torch.Tensor,
        lambda_nbh: torch.Tensor,
        state_out: torch.Tensor,
        norm_out: torch.Tensor,
    ) -> None:
        assert all(t.is_mps for t in (state_in, norm_in, lambda_nbh, state_out, norm_out))
        assert state_in.dtype == torch.float32 and norm_in.dtype == torch.float32
        N, BH, C, D = state_in.shape
        assert norm_in.shape == (N, BH, C)
        assert lambda_nbh.shape == (N, BH)
        assert state_out.shape == (N, BH, C, D)
        assert norm_out.shape == (N, BH, C)

        # Flatten to match kernel's expected contiguous layout
        self.lib.discumsum_state(
            state_in.contiguous(),
            norm_in.contiguous(),
            lambda_nbh.contiguous(),
            state_out,
            norm_out,
            int(N), int(BH), int(C), int(D),
            threads=(BH * C, 1, 1),
        )


class FusedUpdateState:
    def __init__(self, head_dim: int, d_tile: int = 32, normalized: bool = True):
        self.phi = Phi2FeatureMap(head_dim, normalized)
        self.lib = compile_fused_update_state_phi2_bh(head_dim, d_tile)

    def update_bh(self, k_bh: torch.Tensor, v_bh: torch.Tensor, state_bh: torch.Tensor, norm_bh: torch.Tensor, decay_bh: torch.Tensor) -> None:
        assert all(t.is_mps for t in (k_bh, v_bh, state_bh, norm_bh, decay_bh))
        BH, L, D = k_bh.shape
        BHv, Lv, Dv = v_bh.shape
        assert BH == BHv and L == Lv and D == Dv
        C = self.phi.out_dim
        assert state_bh.shape == (BH, C, D) and norm_bh.shape == (BH, C)
        if decay_bh.dim() == 0:
            decay_bh = decay_bh.expand(BH)
        assert decay_bh.shape == (BH,)
        self.lib.fused_update_state_phi2_bh(
            k_bh.contiguous(),
            v_bh.contiguous(),
            state_bh,
            norm_bh,
            decay_bh.contiguous(),
            int(BH), int(L), int(D), int(C),
            threads=(BH * C, 1, 1)
        )


class FusedQueryState:
    def __init__(self, head_dim: int, d_tile: int = 32, normalized: bool = True):
        self.phi = Phi2FeatureMap(head_dim, normalized)
        self.lib = compile_fused_query_state_phi2_bh(head_dim, d_tile)

    def run(self, q_bh: torch.Tensor, state_bh: torch.Tensor, norm_bh: torch.Tensor, out_bh: torch.Tensor, out_norm_bh: torch.Tensor) -> None:
        assert all(t.is_mps for t in (q_bh, state_bh, norm_bh, out_bh, out_norm_bh))
        BH, L, D = q_bh.shape
        C = self.phi.out_dim
        assert state_bh.shape == (BH, C, D) and norm_bh.shape == (BH, C)
        assert out_bh.shape == (BH, L, D) and out_norm_bh.shape == (BH, L)
        self.lib.fused_query_state_phi2_bh(
            q_bh.contiguous(),
            state_bh,
            norm_bh,
            out_bh,
            out_norm_bh,
            int(BH), int(L), int(D), int(C),
            threads=(BH * L, 1, 1)
        )


class IntraInterFusedChunk:
    def __init__(self, head_dim: int, d_tile: int = 32, normalized: bool = True):
        self.phi = Phi2FeatureMap(head_dim, normalized)
        self.lib = compile_intra_inter_fused_chunk_bh(head_dim, d_tile)

    def run(self, q_bh: torch.Tensor, k_bh: torch.Tensor, v_bh: torch.Tensor, state_bh: torch.Tensor, norm_bh: torch.Tensor, out_bh: torch.Tensor) -> None:
        assert all(t.is_mps for t in (q_bh, k_bh, v_bh, state_bh, norm_bh, out_bh))
        BH, L, D = q_bh.shape
        BHk, Lk, Dk = k_bh.shape
        BHv, Lv, Dv = v_bh.shape
        assert BH == BHk == BHv and L == Lk == Lv and D == Dk == Dv
        C = self.phi.out_dim
        assert state_bh.shape == (BH, C, D) and norm_bh.shape == (BH, C)
        assert out_bh.shape == (BH, L, D)
        self.lib.intra_inter_fused_chunk_bh(
            q_bh.contiguous(),
            k_bh.contiguous(),
            v_bh.contiguous(),
            state_bh,
            norm_bh,
            out_bh,
            int(BH), int(L), int(D), int(C),
            threads=(BH * L, 1, 1)
        )


class FullFusedChunk:
    """End-to-end fused per-chunk pipeline (two-dispatch) for Power Attention p=2 on MPS.

    Semantics per chunk (streaming):
      1) Outputs: computes normalized outputs using both local intra-chunk contribution
         and current inter-chunk state (no φ materialization), equivalent to
         IntraInterFusedChunk.
      2) State update: updates (state, norm_state) with decay and this chunk's (k,v)
         using the fused update kernel without explicit φ materialization.

    This preserves correct ordering on Metal (no grid-wide barrier), while
    maintaining the fusion recommended by the paper via two sequential kernels.
    """

    def __init__(self, head_dim: int, d_tile: int = 32, normalized: bool = True):
        self.head_dim = int(head_dim)
        self.mode = os.environ.get('MPS_FULLFUSED_MODE', 'intra_inter')  # 'intra_inter' (default) or 'compose'
        if self.mode == 'compose':
            self.local = QKPow2IntraFused()
            self.qstate = FusedQueryState(head_dim, d_tile=d_tile, normalized=normalized)
        else:
            self.intra_inter = IntraInterFusedChunk(head_dim, d_tile=d_tile, normalized=normalized)
        self.updater = FusedUpdateState(head_dim, d_tile=d_tile, normalized=normalized)
        self.phi = Phi2FeatureMap(head_dim, normalized)
        self.eps = 1e-6

    def run(
        self,
        q_bh: torch.Tensor,        # [BH, L, D]
        k_bh: torch.Tensor,        # [BH, L, D]
        v_bh: torch.Tensor,        # [BH, L, Dv]
        state_bh: torch.Tensor,    # [BH, C, D]
        norm_bh: torch.Tensor,     # [BH, C]
        decay_bh: torch.Tensor,    # [BH] or scalar tensor
        out_bh: torch.Tensor       # [BH, L, D]
    ) -> None:
        assert all(t.is_mps for t in (q_bh, k_bh, v_bh, state_bh, norm_bh, out_bh)), "All tensors must be on MPS"

        if self.mode == 'compose':
            BH, L, D = q_bh.shape
            # 1) Local intra-chunk: (q·k)^2 fused
            local_out = torch.empty(BH, L, D, device=q_bh.device, dtype=q_bh.dtype)
            local_norm = torch.empty(BH, L, device=q_bh.device, dtype=q_bh.dtype)
            self.local.run(q_bh, k_bh, v_bh, local_out, local_norm)

            # 2) Inter-chunk: query current state via fused qstate
            out_state = torch.empty(BH, L, D, device=q_bh.device, dtype=q_bh.dtype)
            state_norm = torch.empty(BH, L, device=q_bh.device, dtype=q_bh.dtype)
            self.qstate.run(q_bh, state_bh, norm_bh, out_state, state_norm)

            # 3) Combine and write outputs
            denom = (local_norm + state_norm).unsqueeze(-1)
            out_bh.copy_((local_out + out_state) / (denom + self.eps))
        else:
            # Single kernel for intra+inter+normalize directly into out_bh
            self.intra_inter.run(q_bh, k_bh, v_bh, state_bh, norm_bh, out_bh)

        # Optional correctness check for state update only (debug)
        do_check = os.environ.get('MPS_FUSED_CHECK', '0') == '1'
        if do_check:
            # Copy pre-update state
            prev_state = state_bh.clone()
            prev_norm = norm_bh.clone()

        # 2) Update state in-place for next chunk
        self.updater.update_bh(k_bh, v_bh, state_bh, norm_bh, decay_bh)

        if do_check:
            # Verify fused update equals expected matmul(φ_k^T, v) with decay
            BH, L, D = k_bh.shape
            C = self.phi.out_dim
            # Build φ(k) for this chunk via the feature map; reshape hack: [1,BH,L,D]
            k_tmp = k_bh.view(1, BH, L, D)
            phi_k = self.phi.expand(k_tmp).contiguous().view(BH, L, C)
            added_state = torch.matmul(phi_k.transpose(1, 2), v_bh)
            added_norm = phi_k.sum(dim=1)
            if decay_bh.dim() == 0:
                decay_bh = decay_bh.expand(BH)
            expected_state = prev_state * decay_bh.view(BH, 1, 1) + added_state
            expected_norm = prev_norm * decay_bh.view(BH, 1) + added_norm
            # Debug prints with small stats
            ok_s = torch.allclose(state_bh, expected_state, atol=1e-3, rtol=1e-3)
            ok_n = torch.allclose(norm_bh, expected_norm, atol=1e-3, rtol=1e-3)
            if not (ok_s and ok_n):
                max_diff_s = (state_bh - expected_state).abs().max().item()
                max_diff_n = (norm_bh - expected_norm).abs().max().item()
                print(f"[FullFusedChunk][debug] state mismatch: max |Δ|={max_diff_s:.4e}, norm |Δ|={max_diff_n:.4e}")
            else:
                print("[FullFusedChunk][debug] fused update matches expected")
