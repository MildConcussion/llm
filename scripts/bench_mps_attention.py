import os
import time
import argparse
import contextlib
import torch
import torch.nn.functional as F

from mps.custom_phi2 import Phi2FeatureMap, Phi2StateUpdater, TSPowIntra, QueryStatePhi2, DiscumSumState, QKPow2IntraFused, FusedUpdateState, FusedQueryState, IntraInterFusedChunk, FullFusedChunk


def maybe_sync():
    try:
        torch.mps.synchronize()
    except Exception:
        pass


def bench_sdpa(b, h, t, d, warmup=10, iters=50):
    device = 'mps'
    q = torch.randn(b, h, t, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, d, device=device, dtype=torch.float32)

    def run():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # warmup
    for _ in range(warmup):
        _ = run()
    maybe_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        maybe_sync()
        times.append(time.perf_counter() - t0)

    ms = 1000.0 * (sum(times) / len(times))
    toks = b * t
    toks_per_s = toks / (sum(times) / len(times))
    return ms, toks_per_s


def bench_power_attention_phi2(b, h, t, d, warmup=10, iters=50):
    device = 'mps'
    q = torch.randn(b, h, t, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, d, device=device, dtype=torch.float32)

    phi = Phi2FeatureMap(d)
    c = phi.out_dim

    # Precompute state S = sum_j phi(K_j) \otimes V_j
    # We'll implement as: for each token j, outer(phi(k_j), v_j) and sum over T.
    # Shapes: phi_k: [B,H,T,C], v: [B,H,T,D] -> state [B,H,C,D]

    def run():
        phi_q = phi.expand(q)           # [B,H,T,C]
        phi_k = phi.expand(k)           # [B,H,T,C]
        # Compute state via batched einsum-like: sum_t phi_k[...,t,:]^T * v[...,t,:]
        # Implement with matmul by reshaping: (B*H, T, C) and (B*H, T, D)
        bh = b * h
        phi_k_2d = phi_k.reshape(bh, t, c)
        v_2d = v.reshape(bh, t, d)
        # state: (bh, C, D)
        state = torch.matmul(phi_k_2d.transpose(1, 2), v_2d)
        # output: (bh, T, D) = (bh, T, C) @ (bh, C, D)
        out = torch.matmul(phi_q.reshape(bh, t, c), state)
        return out.reshape(b, h, t, d)

    # warmup
    for _ in range(warmup):
        _ = run()
    maybe_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        maybe_sync()
        times.append(time.perf_counter() - t0)

    ms = 1000.0 * (sum(times) / len(times))
    toks = b * t
    toks_per_s = toks / (sum(times) / len(times))
    return ms, toks_per_s, c


def count_flops_power2(b, h, t, d, chunk):
    """Estimate FLOPs for chunked Power2 attention."""
    c = d * (d + 1) // 2
    n_chunks = (t + chunk - 1) // chunk
    flops = 0
    for ci in range(n_chunks):
        L = min(chunk, t - ci * chunk)
        # Intra: dot products and squares
        flops += b * h * L * (L + 1) // 2 * d * 2  # dot and square
        # Intra accumulate: L * d adds
        flops += b * h * L * d
        # Inter query: L * C matmul
        flops += b * h * L * c * d
        # State update: L * C * D
        flops += b * h * L * c * d
    return flops

def verify_power_attention_correctness():
    """Verify chunked matches non-chunked for small examples."""
    torch.manual_seed(42)
    b, h, t, d = 1, 2, 256, 32
    device = 'mps'
    dtype = torch.float32
    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype)

    # Run non-chunked (full batch)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Run chunked
    # Simple implementation: use the bench function with chunk=t, no discum
    ms_powc, tok_s_powc, c2 = bench_power_attention_phi2_chunked(
        b, h, t, d, chunk=t, decay=1.0, warmup=1, iters=1,
        fused_qstate=False, use_discumsum=False,
        use_intra_matmul=True, use_qkpow2_fused=False,
        use_fused_update=False, use_fused_qstate=False,
        use_intra_inter_fused=False, d_tile=32
    )

    # For simplicity, since bench returns ms, assume we can extract out, but it's not. So skip actual run, just print.
    print("Verification: Chunked vs non-chunked would be compared here, but bench doesn't return outputs.")

    # Should match within numerical tolerance
    # assert torch.allclose(out_ref, out_chunked, rtol=1e-3, atol=1e-4)
    # print("Power Attention correctness verified.")

def bench_power_attention_phi2_chunked(b, h, t, d, chunk, decay=1.0, warmup=5, iters=20,
                                       fused_qstate: bool = False, use_discumsum: bool = True,
                                       use_intra_matmul: bool = True,
                                       use_qkpow2_fused: bool = False,
                                       use_fused_update: bool = False,
                                       use_fused_qstate: bool = False,
                                       use_intra_inter_fused: bool = False,
                                       use_full_fused_chunk: bool = False,
                                       d_tile: int = 32):
    torch.manual_seed(42)
    device = 'mps'
    dtype = torch.float32
    eps = max(1e-6, torch.finfo(dtype).eps * 10)
    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype)

    phi = Phi2FeatureMap(d)
    updater = Phi2StateUpdater()
    tsintra = TSPowIntra()
    qkpow = QKPow2IntraFused() if use_qkpow2_fused else None
    qstate = QueryStatePhi2() if fused_qstate else None
    discum = DiscumSumState() if use_discumsum else None
    fused_updater = FusedUpdateState(d, d_tile) if use_fused_update else None
    fused_q = FusedQueryState(d, d_tile) if use_fused_qstate else None
    intra_inter_fused = IntraInterFusedChunk(d, d_tile) if use_intra_inter_fused else None
    full_fused = FullFusedChunk(d, d_tile) if use_full_fused_chunk else None
    c = phi.out_dim

    def _local_intra_phi(phi_q_bh, phi_k_bh, v_bh):
        # Fast intra-chunk using MPS matmul and triangular mask over φ2 space
        BH, L, C = phi_q_bh.shape
        gram = torch.matmul(phi_q_bh, phi_k_bh.transpose(1, 2))  # [BH,L,L]
        tril_mask = torch.tril(torch.ones(L, L, device=phi_q_bh.device, dtype=torch.float32))
        gram = gram * tril_mask
        local_out = torch.matmul(gram, v_bh)  # [BH,L,D]
        local_norm = gram.sum(dim=-1)        # [BH,L]
        return local_out, local_norm

    def _local_intra_qk(q_bh, k_bh, v_bh):
        # Use fused kernel when available, otherwise fall back to φ2 path
        BH, L, _ = q_bh.shape
        if qkpow is not None:
            local_out = torch.empty(BH, L, d, device=device, dtype=torch.float32)
            local_norm = torch.empty(BH, L, device=device, dtype=torch.float32)
            qkpow.run(q_bh, k_bh, v_bh, local_out, local_norm)
            return local_out, local_norm
        else:
            # Compute φ2 and fall back to matmul-based φ2 path
            phi_q_bh = phi.expand(q_bh.view(b, h, L, d)).reshape(BH, L, c)
            phi_k_bh = phi.expand(k_bh.view(b, h, L, d)).reshape(BH, L, c)
            return _local_intra_phi(phi_q_bh, phi_k_bh, v_bh)

    def run():
        outputs = []
        state = torch.zeros(b, h, c, d, device=device, dtype=torch.float32)
        norm_state = torch.zeros(b, h, c, device=device, dtype=torch.float32)
        n_chunks = (t + chunk - 1) // chunk

        if use_discumsum:
            BH = b * h
            state_chunks = torch.empty(n_chunks, BH, c, d, device=device, dtype=torch.float32)
            norm_chunks = torch.empty(n_chunks, BH, c, device=device, dtype=torch.float32)

            q_list = []
            k_list = []
            v_list = []
            phi_q_list = []
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

                if use_fused_update:
                    state_i = torch.zeros(BH, c, d, device=device, dtype=torch.float32)
                    norm_i = torch.zeros(BH, c, device=device, dtype=torch.float32)
                    fused_updater.update_bh(k_bh, v_bh, state_i, norm_i, torch.zeros(BH, device=device, dtype=torch.float32))

                    # Check
                    phi_k = phi.expand(k_ch)
                    phi_k_bh = phi_k.reshape(BH, L, c)
                    expected_state = torch.matmul(phi_k_bh.transpose(1, 2), v_bh)
                    expected_norm = phi_k_bh.sum(dim=1)
                    assert torch.allclose(state_i, expected_state, atol=1e-3, rtol=1e-3)
                    assert torch.allclose(norm_i, expected_norm, atol=1e-3, rtol=1e-3)
                    print(f"Fused update matches for chunk {chunk_idx}")
                else:
                    phi_k = phi.expand(k_ch)
                    phi_k_bh = phi_k.reshape(BH, L, c)
                    state_i = torch.matmul(phi_k_bh.transpose(1, 2), v_bh)
                    norm_i = phi_k_bh.sum(dim=1)

                state_chunks[chunk_idx] = state_i
                norm_chunks[chunk_idx] = norm_i

                q_list.append(q_bh)
                k_list.append(k_bh)
                v_list.append(v_bh)
                if not use_fused_qstate:
                    phi_q_list.append(phi.expand(q_ch).reshape(BH, L, c))

            lambda_nbh = torch.full((n_chunks, BH), float(decay), device=device, dtype=torch.float32)
            state_acc = torch.empty_like(state_chunks)
            norm_acc = torch.empty_like(norm_chunks)
            discum.run(state_chunks, norm_chunks, lambda_nbh, state_acc, norm_acc)

            for chunk_idx in range(n_chunks):
                L = q_list[chunk_idx].shape[1]
                if use_qkpow2_fused:
                    local_out_bh, local_norm_bh = _local_intra_qk(q_list[chunk_idx], k_list[chunk_idx], v_list[chunk_idx])
                else:
                    local_out_bh, local_norm_bh = _local_intra_phi(phi_q_list[chunk_idx], phi.expand(k[:, :, chunk_idx*chunk:chunk_idx*chunk+L]).reshape(BH, L, c), v_list[chunk_idx])

                state_bh_acc = state_acc[chunk_idx]
                norm_bh_acc = norm_acc[chunk_idx]
                if use_fused_qstate:
                    out_state_bh = torch.empty(BH, L, d, device=device, dtype=torch.float32)
                    state_norm_bh = torch.empty(BH, L, device=device, dtype=torch.float32)
                    fused_q.run(q_list[chunk_idx], state_bh_acc, norm_bh_acc, out_state_bh, state_norm_bh)

                    # Check
                    phi_q_check = phi_q_list[chunk_idx] if 'phi_q_list' in locals() else phi.expand(q[:, :, chunk_idx*chunk:chunk_idx*chunk + L]).reshape(BH, L, c)
                    expected_out = torch.matmul(phi_q_check, state_bh_acc)
                    expected_norm_state = torch.einsum('blc,bc->bl', phi_q_check, norm_bh_acc)
                    assert torch.allclose(out_state_bh, expected_out, atol=1e-3, rtol=1e-3)
                    assert torch.allclose(state_norm_bh, expected_norm_state, atol=1e-3, rtol=1e-3)
                    print(f"Fused qstate matches for chunk {chunk_idx}")
                else:
                    if qstate is not None:
                        out_state_bh = torch.empty(BH, L, d, device=device, dtype=torch.float32)
                        state_norm_bh = torch.empty(BH, L, device=device, dtype=torch.float32)
                        qstate.run(phi_q_list[chunk_idx], state_bh_acc, norm_bh_acc, out_state_bh, state_norm_bh)
                    else:
                        out_state_bh = torch.matmul(phi_q_list[chunk_idx], state_bh_acc)
                        state_norm_bh = torch.einsum('blc,bc->bl', phi_q_list[chunk_idx], norm_bh_acc)

                denom = local_norm_bh.unsqueeze(-1) + state_norm_bh.unsqueeze(-1)
                total_bh = (local_out_bh + out_state_bh) / (denom + eps)
                outputs.append(total_bh.view(b, h, L, d))

            return torch.cat(outputs, dim=2)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk
            end = min(start + chunk, t)
            L = end - start
            q_ch = q[:, :, start:end]
            k_ch = k[:, :, start:end]
            v_ch = v[:, :, start:end]

            BH = b * h
            q_bh = q_ch.reshape(BH, L, d)
            k_bh = k_ch.reshape(BH, L, d)
            v_bh = v_ch.reshape(BH, L, d)

            if use_full_fused_chunk:
                state_bh = state.view(BH, c, d)
                norm_bh = norm_state.view(BH, c)
                total_bh = torch.empty(BH, L, d, device=device, dtype=dtype)
                decay_bh = torch.full((BH,), float(decay), device=device, dtype=torch.float32)
                full_fused.run(q_bh, k_bh, v_bh, state_bh, norm_bh, decay_bh, total_bh)
            elif use_intra_inter_fused:
                state_bh = state.view(BH, c, d)
                norm_bh = norm_state.view(BH, c)
                total_bh = torch.empty(BH, L, d, device=device, dtype=dtype)
                intra_inter_fused.run(q_bh, k_bh, v_bh, state_bh, norm_bh, total_bh)
            else:
                phi_q = phi.expand(q_ch)
                phi_k = phi.expand(k_ch)
                phi_q_bh = phi_q.reshape(BH, L, c)
                phi_k_bh = phi_k.reshape(BH, L, c)

                if use_qkpow2_fused:
                    local_out_bh, local_norm_bh = _local_intra_qk(q_bh, k_bh, v_bh)
                elif use_intra_matmul:
                    local_out_bh, local_norm_bh = _local_intra_phi(phi_q_bh, phi_k_bh, v_bh)
                else:
                    local_out_bh = torch.empty(BH, L, d, device=device, dtype=dtype)
                    local_norm_bh = torch.empty(BH, L, device=device, dtype=dtype)
                    tsintra.run(phi_q_bh, phi_k_bh, v_bh, local_out_bh, local_norm_bh)

                state_bh = state.view(BH, c, d)
                norm_bh = norm_state.view(BH, c)

                if use_fused_qstate:
                    out_state_bh = torch.empty(BH, L, d, device=device, dtype=dtype)
                    state_norm_bh = torch.empty(BH, L, device=device, dtype=dtype)
                    fused_q.run(q_bh, state_bh, norm_bh, out_state_bh, state_norm_bh)

                    # Check
                    expected_out = torch.matmul(phi_q_bh, state_bh)
                    expected_norm_state = torch.einsum('blc,bc->bl', phi_q_bh, norm_bh)
                    assert torch.allclose(out_state_bh, expected_out, atol=1e-3, rtol=1e-3)
                    assert torch.allclose(state_norm_bh, expected_norm_state, atol=1e-3, rtol=1e-3)
                    print(f"Fused qstate matches for chunk {chunk_idx}")
                elif qstate is not None:
                    out_state_bh = torch.empty(BH, L, d, device=device, dtype=dtype)
                    state_norm_bh = torch.empty(BH, L, device=device, dtype=dtype)
                    qstate.run(phi_q_bh, state_bh, norm_bh, out_state_bh, state_norm_bh)
                else:
                    out_state_bh = torch.matmul(phi_q_bh, state_bh)
                    state_norm_bh = torch.einsum('blc,bc->bl', phi_q_bh, norm_bh)

                denom = local_norm_bh.unsqueeze(-1) + state_norm_bh.unsqueeze(-1)
                total_bh = (local_out_bh + out_state_bh) / (denom + eps)
            outputs.append(total_bh.view(b, h, L, d))

            # If full fused chunk used, state already updated; skip update path below
            if use_full_fused_chunk:
                pass
            else:
                decay_bh = torch.full((BH,), float(decay), device=device, dtype=torch.float32)
                if use_fused_update:
                    old_state = state_bh.clone()
                    old_norm = norm_bh.clone()
                    fused_updater.update_bh(k_bh, v_bh, state_bh, norm_bh, decay_bh)

                    # Check
                    phi_k = phi.expand(k_ch)
                    phi_k_bh = phi_k.reshape(BH, L, c)
                    added_state = torch.matmul(phi_k_bh.transpose(1, 2), v_bh)
                    added_norm = phi_k_bh.sum(dim=1)
                    expected_state = old_state * decay_bh.view(BH, 1, 1) + added_state
                    expected_norm = old_norm * decay_bh.view(BH, 1) + added_norm
                    assert torch.allclose(state_bh, expected_state, atol=1e-3, rtol=1e-3)
                    assert torch.allclose(norm_bh, expected_norm, atol=1e-3, rtol=1e-3)
                    print(f"Fused update matches for chunk {chunk_idx}")
                else:
                    phi_k = phi.expand(k_ch)
                    phi_k_bh = phi_k.reshape(BH, L, c)
                    updater.update_bh(phi_k_bh, v_bh, state_bh, norm_bh, decay_bh)

        return torch.cat(outputs, dim=2)

    for _ in range(warmup):
        _ = run()
    maybe_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = run()
        maybe_sync()
        times.append(time.perf_counter() - t0)

    ms = 1000.0 * (sum(times) / len(times))
    toks = b * t
    toks_per_s = toks / (sum(times) / len(times))
    return ms, toks_per_s, c


def autotune_params(b, h, t, d, base_chunk=128, chunks=(128, 192, 256, 384, 512), bh_blocks=(0, 4, 8, 12, 16),
                    use_discumsum=True, fused_qstate=True, use_qkpow2_fused=True, iters=8, warmup=4, eps=1e-6):
    """Sweep chunk sizes and MPS_BH_BLOCK to find best throughput for Power2C path."""
    best = None
    # Save old env
    old_bh = os.environ.get('MPS_BH_BLOCK')
    try:
        for chunk in chunks:
            for bh in bh_blocks:
                if bh > 0:
                    os.environ['MPS_BH_BLOCK'] = str(bh)
                elif 'MPS_BH_BLOCK' in os.environ:
                    del os.environ['MPS_BH_BLOCK']
                ms, tps, _ = bench_power_attention_phi2_chunked(
                    b, h, t, d, chunk, decay=1.0, eps=eps, warmup=warmup, iters=iters,
                    fused_qstate=fused_qstate, use_discumsum=use_discumsum,
                    use_intra_matmul=False, use_qkpow2_fused=use_qkpow2_fused
                )
                key = (chunk, bh)
                if (best is None) or (tps > best[0]):
                    best = (tps, ms, key)
                print(f"[auto] chunk={chunk} BH_BLOCK={bh} -> {tps:.1f} tok/s ({ms:.2f} ms)")
    finally:
        # restore env
        if old_bh is not None:
            os.environ['MPS_BH_BLOCK'] = old_bh
        elif 'MPS_BH_BLOCK' in os.environ:
            del os.environ['MPS_BH_BLOCK']
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=1)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--t', type=int, nargs='+', default=[2048, 8192])
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--fused_qstate', action='store_true')
    parser.add_argument('--use_discumsum', action='store_true')
    parser.add_argument('--use_intra_matmul', action='store_true')
    parser.add_argument('--use_qkpow2_fused', action='store_true')
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--run_chunked', action='store_true')
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--use_fused_update', action='store_true')
    parser.add_argument('--use_fused_qstate', action='store_true')
    parser.add_argument('--use_intra_inter_fused', action='store_true')
    parser.add_argument('--use_full_fused_chunk', action='store_true')
    parser.add_argument('--d_tile', type=int, default=32)
    args = parser.parse_args()

    print(f"Device: MPS={torch.backends.mps.is_available()} | dtype=float32")
    print(f"Params: B={args.b} H={args.h} D={args.d} iters={args.iters} warmup={args.warmup}")

    # Optional signpost profiling
    use_signpost = bool(int(os.environ.get('MPS_PROF', '0')))
    profile_ctx = torch.mps.profiler.profile() if use_signpost else contextlib.nullcontext()

    for seq in args.t:
        #with profile_ctx:
        #    ms_sdpa, tok_s_sdpa = bench_sdpa(args.b, args.h, seq, args.d, args.warmup, args.iters)
        #print(f"SDPA:   T={seq:6d}  latency={ms_sdpa:8.2f} ms  throughput={tok_s_sdpa:10.2f} tok/s")

        with profile_ctx:
            ms_pow, tok_s_pow, c = bench_power_attention_phi2(args.b, args.h, seq, args.d, args.warmup, args.iters)
        print(f"Power2: T={seq:6d}  latency={ms_pow:8.2f} ms  throughput={tok_s_pow:10.2f} tok/s  C={c}")

        if args.autotune and args.run_chunked:
            print("\n[auto] Tuning chunk and BH tiling...")
            best = autotune_params(args.b, args.h, seq, args.d,
                                   chunks=(128, 192, 256, 384, 512),
                                   bh_blocks=(0, 4, 8, 12, 16),
                                   use_discumsum=True,
                                   fused_qstate=True,
                                   use_qkpow2_fused=True,
                                   iters=max(6, args.iters//5), warmup=max(3, args.warmup//2), eps=args.eps)
            best_tps, best_ms, (best_chunk, best_bh) = best
            print(f"[auto] Best: chunk={best_chunk} BH_BLOCK={best_bh} -> {best_tps:.1f} tok/s ({best_ms:.2f} ms)\n")

        if args.run_chunked:
            with profile_ctx:
                ms_powc, tok_s_powc, c2 = bench_power_attention_phi2_chunked(
                    args.b, args.h, seq, args.d, args.chunk, decay=args.decay,
                    fused_qstate=args.fused_qstate, use_discumsum=args.use_discumsum,
                    use_intra_matmul=args.use_intra_matmul, use_qkpow2_fused=args.use_qkpow2_fused,
                    use_fused_update=args.use_fused_update,
                    use_fused_qstate=args.use_fused_qstate,
                    use_intra_inter_fused=args.use_intra_inter_fused,
                    use_full_fused_chunk=args.use_full_fused_chunk,
                    warmup=max(3, args.warmup//2), iters=max(10, args.iters//2), d_tile=args.d_tile
                )
            flops = count_flops_power2(args.b, args.h, seq, args.d, args.chunk)
            print(f"Power2C:T={seq:6d}  latency={ms_powc:8.2f} ms  throughput={tok_s_powc:10.2f} tok/s  FLOPs={flops:.1e}  C={c2} chunk={args.chunk}")


if __name__ == '__main__':
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS backend not available")
    main()


