#!/usr/bin/env python3
# scripts/moe_microbench.py

import os
import time
import argparse
import torch
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_MPS_FAST_MATH', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesicap_reso_test import MoEFeedForward as MoE_New  # noqa: E402
from mesicap_reso_test2 import MoEFeedForward as MoE_Orig  # noqa: E402


def run_bench(impl: str, batch_size: int, seq_len: int, emb_dim: int, num_experts: int, k: int, steps: int, dtype: str,
              topk_sorted: int, idx64: int):
    assert torch.backends.mps.is_available(), "MPS backend not available"
    device = torch.device('mps')

    dtype_obj = getattr(torch, dtype)
    cfg = {
        "batch_size": batch_size,
        "context_length": seq_len,
        "emb_dim": emb_dim,
        "hidden_dim": emb_dim,
        "moe_intermediate_size": emb_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": k,
        "dtype": dtype_obj,
        "topk_sorted": bool(topk_sorted),
        "token_index_idx64": bool(idx64),
    }

    if impl == 'orig':
        moe = MoE_Orig(cfg).to(device)
    else:
        moe = MoE_New(cfg).to(device)
    x = torch.randn(batch_size, seq_len, emb_dim, device=device, dtype=dtype_obj)

    # Warmup
    for _ in range(10):
        _ = moe(x)
    torch.mps.synchronize()

    tokens = batch_size * seq_len
    start_t = time.perf_counter()
    for _ in range(steps):
        _ = moe(x)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start_t

    tps = (tokens * steps) / max(elapsed, 1e-9)
    print(f"MoE forward ({impl}): steps={steps}, batch={batch_size}, seq={seq_len}, emb={emb_dim}, experts={num_experts}, k={k}, dtype={dtype}")
    print(f"Total time: {elapsed:.3f}s | avg/step: {elapsed/steps*1000:.2f} ms | tokens/s: {tps:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--impl', type=str, default='new', choices=['new','orig'])
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--emb', type=int, default=128)
    parser.add_argument('--experts', type=int, default=2)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'])
    parser.add_argument('--topk_sorted', type=int, default=1)
    parser.add_argument('--idx64', type=int, default=1)
    args = parser.parse_args()

    run_bench(args.impl, args.batch, args.seq, args.emb, args.experts, args.k, args.steps, args.dtype,
              args.topk_sorted, args.idx64)


