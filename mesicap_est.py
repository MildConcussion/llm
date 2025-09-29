"""
8-Bit XOR Language Model - Complete Implementation
Fast, elegant, vocabulary-free language modeling
"""

import itertools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import json
import contextlib
from tqdm import tqdm
from grokadamw import GrokAdamW
from transformers import AutoTokenizer
from torch.nn import RMSNorm
from abc import ABC, abstractmethod
from datatrove.utils.dataset import DatatroveFolderDataset

import os
import wandb

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
np.random.seed(42)

os.environ['PYTORCH_MPS_FAST_MATH'] = '1'


class DataLoaderBase(ABC):
    def __init__(self, config):
        self.config = config

    def _num_steps(self, num_seqs):
        return math.ceil(num_seqs / self.config.batch_size)

    def _next_batch(self, index, start_pos, end_pos):
        new_index = index + self.config.batch_size

        batch_data = self._get_x_y_tokens(index, new_index)

        index = new_index
        if new_index >= end_pos:
            self._shuffle_new_epoch()
            index = start_pos

        return (*batch_data, index)

    @abstractmethod
    def _get_x_y_tokens(self, start, end):
        pass

    @abstractmethod
    def _shuffle_new_epoch(self):
        pass


class FileDataLoader(DataLoaderBase):
    def __init__(self, config, tokenizer, world_size=1, rank=0, seed=1998):
        self.config = config
        self.tokenizer = tokenizer
        self.token_size = (2 if len(tokenizer) < 65535 else 4)
        self.seed = seed
        self.current_epoch = 0
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")

        self._load_dataset(seed)

        self.num_seqs = len(self.dataset)
        if rank == 0:
            print(f"{'Total tokens':<30} | {self.num_seqs * config.seq_length:,}")

        self.total_train_seqs = math.ceil((1-config.val_size) * self.num_seqs)
        shard_size = self.total_train_seqs // world_size

        self.train_start_idx = rank * shard_size
        self.train_end_idx = (rank+1) * shard_size
        self.train_seqs = self.train_end_idx - self.train_start_idx

        print(f"Shard range rank:{rank:<13} | ({self.train_start_idx},{self.train_end_idx})")

        self.train_index = self.train_start_idx

        self.val_seqs = self.num_seqs - self.total_train_seqs
        self.val_index = self.total_train_seqs

    def next_batch_train(self):
        x, y, pos, self.train_index = self._next_batch(self.train_index, self.train_start_idx, self.train_end_idx)

        return x, y, pos

    def next_batch_val(self):
        x, y, pos, self.val_index = self._next_batch(self.val_index, self.total_train_seqs, self.num_seqs)

        return x, y, pos

    def num_train_steps_per_epoch(self):
        return self._num_steps(self.train_seqs)

    def num_val_steps(self):
        return self._num_steps(self.val_seqs)

    def _get_x_y_tokens(self, start, end):
        batch_input_data = []
        batch_pos_data = []
        for idx in range(start, min(end, self.num_seqs)):
            sample = self.dataset[idx]
            input_ids = sample['input_ids']  # full seq_len + 1
            positions = sample['positions']  # pre-computed positions, seq_len + 1
            batch_input_data.append(input_ids)
            batch_pos_data.append(positions)

        input_ids_t = torch.stack(batch_input_data)  # [batch, seq_len+1]
        pos_t = torch.stack(batch_pos_data)  # [batch, seq_len+1]

        # Slice to x, y, pos (using pre-computed positions directly for efficiency)
        x = input_ids_t[:, :-1]
        y = input_ids_t[:, 1:]
        pos = pos_t[:, :-1]  # Align with x

        return x, y, pos

    def _shuffle_new_epoch(self):
        self.current_epoch += 1
        self._load_dataset(self.seed + self.current_epoch)

    def _load_dataset(self, seed):
        self.dataset = DatatroveFolderDataset(
            folder_path=self.config.tokens_folder,
            seq_len=self.config.max_seq_len,
            token_size=self.token_size,
            recursive=True,
            shuffle=True,
            seed=seed,
            return_positions=True,
            positions_from_eos_token_id=self.tokenizer.token_to_id("<|endoftext|>")
        )


class LRUCache:
    """Simple LRU cache with optional device-aware tensor handling.

    - capacity: max number of items to keep
    - move_tensors_to_device: when True, tensors are moved to the requested device on hit
    """
    def __init__(self, capacity: int = 64, move_tensors_to_device: bool = True):
        from collections import OrderedDict
        self.capacity = int(max(1, capacity))
        self.move_tensors_to_device = bool(move_tensors_to_device)
        self._store = OrderedDict()

    def get(self, key, factory=None, device: torch.device | None = None):
        store = self._store
        if key in store:
            val = store.pop(key)
            # Move to end as MRU
            store[key] = val
            # Optionally move tensors to requested device
            if self.move_tensors_to_device and device is not None and isinstance(val, torch.Tensor):
                if val.device != device:
                    val = val.to(device)
                    store[key] = val
            return val
        # Miss
        if factory is None:
            raise KeyError(f"LRUCache miss for key={key} and no factory provided")
        val = factory()
        store[key] = val
        # Evict LRU if over capacity
        if len(store) > self.capacity:
            store.popitem(last=False)
        return val


class SamplingStrategy:
    """Unified, numerically-stable sampling for top-p and top-h.

    Usage: next_token = sampler.sample(last_logits, sampling='top_p', top_p=0.9, alpha=0.4)
    - last_logits: [B, V]
    - Returns: [B, 1] sampled token ids
    """
    def __init__(self):
        pass

    @staticmethod
    def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        # probs: [..., V]
        eps = 1e-10
        p = torch.clamp(probs, min=eps)
        return -(p * torch.log(p)).sum(dim=-1)

    def _apply_top_h(self, sorted_logits: torch.Tensor, sorted_idx: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
        """Return boolean keep_mask for tokens to keep based on entropy threshold.
        sorted_logits/sorted_idx: [B, V]
        """
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        full_entropy = self._entropy_from_probs(sorted_probs)  # [B]
        threshold = alpha * full_entropy  # [B]

        B, V = sorted_probs.shape
        keep_mask = torch.zeros(B, V, dtype=torch.bool, device=sorted_probs.device)
        keep_mask[:, 0] = True

        # Iteratively include up to top-100 tokens or until entropy threshold is crossed
        max_scan = min(100, V)
        for i in range(1, max_scan):
            p_i = sorted_probs[:, i]
            if (p_i < 1e-10).all():
                break
            keep_mask[:, i] = True
            # Compute entropy of current subset per batch
            # Build masked distribution per batch (normalize)
            masked = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
            denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sub_probs = masked / denom
            sub_entropy = self._entropy_from_probs(sub_probs)  # [B]
            # Where we exceeded threshold, revoke this token
            revoke = sub_entropy > threshold
            if revoke.any():
                keep_mask[revoke, i] = False
                # Stop adding more for those batches; continue scanning others
                # We can't easily short-circuit per-batch here; acceptable overhead for small max_scan
        return keep_mask

    @torch.no_grad()
    def sample(self, last_logits: torch.Tensor, sampling: str = 'top_p', top_p: float = 0.9, alpha: float = 0.4,
               debug: bool = False) -> torch.Tensor:
        """Sample next token ids from last_logits using specified strategy.
        Returns tensor of shape [B, 1]
        """
        # Numerical safety for non-finite
        logits = torch.where(torch.isfinite(last_logits), last_logits, torch.full_like(last_logits, -1e10))

        # Sort once
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_logits = torch.where(torch.isfinite(sorted_logits), sorted_logits, torch.full_like(sorted_logits, -1e10))

        # Strategy
        if sampling == 'top_h':
            keep_mask = self._apply_top_h(sorted_logits, sorted_idx, alpha=alpha)
        else:  # top_p
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumsum <= top_p
            keep_mask[:, 0] = True

        # Apply mask and build probabilities
        filtered_logits = torch.where(keep_mask, sorted_logits, torch.full_like(sorted_logits, -float('inf')))
        probs = F.softmax(filtered_logits, dim=-1)

        # Validate distribution
        if (torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any() or (probs.sum(dim=-1) == 0).any()):
            # Fallback: uniform over bytes + small EOS mass is handled by caller when needed
            # Here, just do argmax as a safe fallback
            next_sorted = torch.argmax(sorted_logits, dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, next_sorted)
            return next_token

        next_token_sorted = torch.multinomial(probs, 1)
        next_token = sorted_idx.gather(-1, next_token_sorted)
        return next_token

def golden_groups(n_layers, n_heads=8, min_group_size=1, max_group_size=3):
    """Golden grouping with bounded group sizes for all layers.

    - Each group's size is in [min_group_size, max_group_size].
    - Number of groups k is the minimum needed to satisfy the max size constraint: k = ceil(n_heads / max_group_size).
    - Sizes per layer are based on golden weights and allocated via largest remainder within capacity, then permuted per layer using φ-phase.
    - Groups are contiguous and anchored at head index 0 (no rotation), but which heads get larger chunks alternates across layers.
    """
    φ = (1 + np.sqrt(5)) / 2
    golden_conjugate = φ - 1.0  # ≈ 0.618...

    if n_layers <= 0:
        return []

    # Determine feasible k (number of groups)
    min_gs = int(max(1, min_group_size))
    max_gs = int(max(1, max_group_size))
    if min_gs > max_gs:
        min_gs, max_gs = max_gs, min_gs
    k = int(np.ceil(n_heads / max_gs))

    # Generate normalized golden ratio weights for k groups
    j = np.arange(k, dtype=np.float64)
    weights = φ ** (-j)
    weights = weights / weights.sum()

    # Allocate head sizes using largest-remainder method with capacity constraints
    base_sizes = np.full(k, min_gs, dtype=int)
    remaining = int(n_heads - base_sizes.sum())
    capacities = np.full(k, max_gs - min_gs, dtype=int)

    if remaining > 0:
        # Largest-remainder allocation within capacities
        raw_add = weights * remaining
        add_floor = np.floor(raw_add).astype(int)
        add_floor = np.minimum(add_floor, capacities)
        sizes = base_sizes + add_floor

        # Handle remainders
        leftover = remaining - add_floor.sum()
        if leftover > 0:
            rema = raw_add - add_floor
            order = np.argsort(-rema)
            for idx in order:
                if leftover == 0:
                    break
                if sizes[idx] - base_sizes[idx] < capacities[idx]:
                    sizes[idx] += 1
                    leftover -= 1
    else:
        sizes = base_sizes

    # Adjust sizes to exactly sum to n_heads while respecting bounds
    sizes = np.clip(sizes, min_gs, max_gs)
    diff = int(n_heads - sizes.sum())
    if diff != 0:
        direction = 1 if diff > 0 else -1
        steps = abs(diff)
        # Use remainders to guide distribution
        rema = (weights * n_heads) - np.floor(weights * n_heads)
        order = np.argsort(-rema) if direction > 0 else np.argsort(rema)

        for _ in range(steps):
            for idx in order:
                if direction > 0 and sizes[idx] < max_gs:
                    sizes[idx] += 1
                    break
                elif direction < 0 and sizes[idx] > min_gs:
                    sizes[idx] -= 1
                    break

    # Build per-layer groups by permuting size order using φ-phase
    groups_per_layer = []
    head_ring = np.arange(n_heads, dtype=int)

    for i in range(n_layers):
        # Permute sizes across layers using golden ratio phase
        phase = (i * golden_conjugate) % 1.0
        positions = np.arange(len(sizes), dtype=np.float64)
        keys = np.mod(positions * golden_conjugate + phase, 1.0)
        pos_order = np.argsort(keys)

        # Place sizes (largest first) into permuted positions
        sizes_sorted = np.sort(sizes)[::-1]
        sized_positions = np.empty_like(sizes)
        for rank, pos in enumerate(pos_order):
            sized_positions[pos] = sizes_sorted[rank]

        # Form contiguous groups anchored at head index 0
        groups = []
        start = 0
        for sz in sized_positions:
            if sz > 0:
                groups.append(head_ring[start:start+sz].tolist())
                start += sz
        groups_per_layer.append(groups)

    return groups_per_layer


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"] * 8, dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"] * 8, dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"] * 8, cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        hidden = torch.nn.functional.silu(
                self.fc1(x)
            ) * self.fc2(x)
        return self.fc3(hidden)


def create_moe_kernels():
    kernel_source = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    // Launch with a 1D grid sized to num_tokens * k by passing topk_indices as the first buffer.
    kernel void dispatch_tokens_to_experts(
        device const int* topk_indices_flat [[buffer(0)]], // length: num_tokens * k
        device const float* topk_probs_flat [[buffer(1)]], // length: num_tokens * k
        device const float* input [[buffer(2)]],            // length: num_tokens * emb_dim
        device float* expert_inputs [[buffer(3)]],          // [num_experts, max_tokens_per_expert, emb_dim]
        device float* expert_weights [[buffer(4)]],         // [num_experts, max_tokens_per_expert]
        device int* expert_token_indices [[buffer(5)]],     // [num_experts, max_tokens_per_expert]
        device atomic_int* expert_counts [[buffer(6)]],     // [num_experts]
        constant int& num_experts [[buffer(7)]],
        constant int& emb_dim [[buffer(8)]],
        constant int& k [[buffer(9)]],
        constant int& max_tokens_per_expert [[buffer(10)]],
        uint idx [[thread_position_in_grid]])
    {
        // Map 1D thread index into (token_idx, expert_slot)
        int token_idx = int(idx / (uint)k);
        int expert_slot = int(idx % (uint)k);

        // Load routing
        int expert_id = topk_indices_flat[idx];
        float prob = topk_probs_flat[idx];

        if (expert_slot >= k || expert_id < 0 || expert_id >= num_experts) {
            return;
        }

        // Atomic increment to get position in expert's buffer
        int pos = atomic_fetch_add_explicit(&expert_counts[expert_id], 1, memory_order_relaxed);
        if (pos >= max_tokens_per_expert) {
            // Overflow: drop this assignment
            return;
        }

        // Copy token embedding and metadata
        uint expert_offset = uint(expert_id) * uint(max_tokens_per_expert) * uint(emb_dim);
        uint input_offset = uint(token_idx) * uint(emb_dim);

        // Copy the embedding vector (vectorized when possible)
        uint dst_base = expert_offset + uint(pos) * uint(emb_dim);

        if ((emb_dim & 3u) == 0u) {
            // Vectorized path using float4 when emb_dim is divisible by 4
            uint emb_dim4 = uint(emb_dim) >> 2;
            const device float4* in4 = reinterpret_cast<const device float4*>(input + input_offset);
            device float4* out4 = reinterpret_cast<device float4*>(expert_inputs + dst_base);
            for (uint d4 = 0; d4 < emb_dim4; d4++) {
                out4[d4] = in4[d4];
            }
        } else {
            // Fallback scalar copy
            for (uint d = 0; d < (uint)emb_dim; d++) {
                expert_inputs[dst_base + d] = input[input_offset + d];
            }
        }
        // Write weight and original token index once per dispatched pair
        expert_weights[uint(expert_id) * uint(max_tokens_per_expert) + uint(pos)] = prob;
        expert_token_indices[uint(expert_id) * uint(max_tokens_per_expert) + uint(pos)] = token_idx;
    }
    """
    return torch.mps.compile_shader(kernel_source)


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]
        self.topk_sorted = bool(cfg.get("topk_sorted", True))
        self._token_index_dtype = torch.int64 if bool(cfg.get("token_index_idx64", False)) else torch.int32
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        self.fc1 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc2 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_intermediate_size"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])

        # Pre-allocate buffers for M2 Ultra's unified memory
        # Capacity per expert needs to handle worst-case skew where all tokens route to one expert.
        # Use num_tokens * k for safe capacity.
        max_tokens = cfg["batch_size"] * cfg["context_length"]
        self.max_tokens_per_expert = int(max_tokens * self.num_experts_per_tok)
        self.register_buffer('expert_inputs',
            torch.zeros(self.num_experts, self.max_tokens_per_expert, self.emb_dim, dtype=cfg["dtype"]))
        self.register_buffer('expert_weights',
            torch.zeros(self.num_experts, self.max_tokens_per_expert, dtype=cfg["dtype"]))
        self.register_buffer('expert_token_indices',
            torch.zeros(self.num_experts, self.max_tokens_per_expert, dtype=self._token_index_dtype))
        self.register_buffer('expert_counts',
            torch.zeros(self.num_experts, dtype=torch.int32))

        # Compile Metal kernels
        self.metal_kernels = create_moe_kernels()
        self._dbg_step = 0

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Use MPS-optimized operations
        with torch.mps.profiler.profile():
            # Gate scoring - keep on MPS
            scores = self.gate(x)

            # Top-k on MPS (optimized for M2)
            topk_scores, topk_indices = torch.topk(
                scores, self.num_experts_per_tok, dim=-1,
                sorted=self.topk_sorted
            )
            topk_probs = torch.softmax(topk_scores, dim=-1)

            # Reset counts and (optionally) weights
            self.expert_counts.zero_()

            # Flatten views
            num_tokens = batch * seq_len
            x_flat = x.reshape(num_tokens, self.emb_dim).contiguous()
            topk_indices_flat = topk_indices.reshape(-1).to(torch.int32).contiguous()
            topk_probs_flat = topk_probs.reshape(-1).contiguous()

            # Dispatch tokens using Metal kernel with 1D grid sized by first argument
            self.metal_kernels.dispatch_tokens_to_experts(
                topk_indices_flat,
                topk_probs_flat,
                x_flat,
                self.expert_inputs,
                self.expert_weights,
                self.expert_token_indices,
                self.expert_counts,
                self.num_experts,
                self.emb_dim,
                self.num_experts_per_tok,
                self.max_tokens_per_expert,
            )

            # Simple, correct recombine on-device without CPU sync; single index_add_
            out_flat = torch.zeros(num_tokens, self.emb_dim, device=x.device, dtype=x.dtype)

            # Clamp counts to capacity to compensate for dropped overflows in the kernel
            if (self.expert_counts > self.max_tokens_per_expert).any():
                print("[MoE-Metal][warn] Expert overflow: clamping counts to capacity")
                self.expert_counts.clamp_max_(self.max_tokens_per_expert)

            # Use contiguous slices per expert (fast on MPS) with a single later index_add_
            counts_cpu = self.expert_counts.detach().cpu().tolist()
            all_outs = []
            all_token_idx = []
            all_weights = []

            for expert_id, n_tok in enumerate(counts_cpu):
                n_tok = int(n_tok)
                if n_tok <= 0:
                    continue
                expert_in = self.expert_inputs[expert_id, :n_tok, :]
                weights = self.expert_weights[expert_id, :n_tok]
                token_idx = self.expert_token_indices[expert_id, :n_tok].long()

                hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_in)) * self.fc2[expert_id](expert_in)
                expert_out = self.fc3[expert_id](hidden)

                all_outs.append(expert_out)
                all_token_idx.append(token_idx)
                all_weights.append(weights)

            if all_outs:
                cat_out = torch.cat(all_outs, dim=0)
                cat_idx = torch.cat(all_token_idx, dim=0)
                cat_w = torch.cat(all_weights, dim=0).unsqueeze(-1)
                out_flat.index_add_(0, cat_idx, cat_out * cat_w)

            # Optional gated debug (rare CPU sync by design)
            # self._dbg_step += 1
            # if os.getenv('MPS_MOE_DEBUG') and (self._dbg_step % int(os.getenv('MPS_MOE_DEBUG_EVERY', '100')) == 0):
            #     counts_list = self.expert_counts.detach().cpu().tolist()
            #     total_assigned = sum(int(c) for c in counts_list)
            #     print(f"[MoE-Metal][dbg] counts[:min(8,E)]={counts_list[:min(8, len(counts_list))]} sum={total_assigned} cap={self.max_tokens_per_expert}")

            outputs = out_flat.view(batch, seq_len, self.emb_dim)

        return outputs

    def _process_experts_parallel(self):
        # Group experts by similar token counts for better GPU utilization
        counts = self.expert_counts.cpu().numpy()
        expert_groups = self._group_experts_by_load(counts)

        outputs = []
        for group in expert_groups:
            # Process similar-sized batches together
            group_outputs = torch.nn.parallel.parallel_apply(
                [self._expert_forward(i) for i in group],
                [self.expert_inputs[i, :counts[i]] for i in group]
            )
            outputs.extend(group_outputs)

        return self._combine_outputs(outputs)

    def _expert_forward(self, expert_id):
        def forward(x):
            # Fused operations for M2 Ultra
            hidden = torch.nn.functional.silu(
                self.fc1[expert_id](x)
            ) * self.fc2[expert_id](x)
            return self.fc3[expert_id](hidden)
        return forward


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # Avoid splitting and concatenating
    batch_size, num_heads, seq_len, head_dim = x.shape

    # Use complex number rotation (much faster)
    x_complex = x.float().reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_complex)

    cos = cos[:seq_len, :head_dim // 2].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :head_dim // 2].unsqueeze(0).unsqueeze(0)
    freqs_complex = torch.complex(cos, sin)

    x_rotated = x_complex * freqs_complex
    x_rotated = torch.view_as_real(x_rotated).reshape(batch_size, num_heads, seq_len, head_dim)

    return x_rotated.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # Head-wise sigmoid gate
        self.W_gate = nn.Linear(d_in, num_heads, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None


    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention (SDPA fused path on MPS)
        # attn_out = mps_power_attention(
        attn_out = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            is_causal=True,
        )

        # Apply head-wise sigmoid gate
        gate = torch.sigmoid(self.W_gate(x))  # (b, num_tokens, num_heads)
        gate = gate.transpose(1, 2).unsqueeze(-1)  # (b, num_heads, num_tokens, 1)
        attn_out = attn_out * gate

        context = attn_out.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x

class MesicapLM(nn.Module):
    def __init__(self, cfg, encoder):
        super().__init__()
        self.encoder = encoder
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock({**cfg, "num_experts": 0}) if i == 0 else TransformerBlock(cfg) for i in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    @torch.no_grad()
    def generate(self, prompt="", max_len=100, use_cache=True, temp=1.0, sampling='top_p', top_p=0.9, alpha=0.4, debug=False):

        self.eval()
        device = next(self.parameters()).device
        sampler = SamplingStrategy()

        input_token_ids = self.encoder.encode(prompt)

        if len(input_token_ids) > 0 and input_token_ids[-1] == self.encoder.END:
            input_token_ids = input_token_ids[:-1]
        x = input_token_ids.detach().clone().unsqueeze(0).to(device, non_blocking=True)

        # Simple generation without cache
        for _ in range(max_len):
            if x.size(1) > self.cfg["context_length"]:
                x = x[:, -self.cfg["context_length"]:]

            logits = self(x)[:, -1] / max(temp, 1e-6)
            next_token = self._sample_next_token(logits, sampler, sampling, top_p, alpha, debug)

            if next_token.item() == self.encoder.END:
                break
            x = torch.cat([x, next_token], dim=1)


        return self.encoder.decode(x[0].cpu().numpy())

    def _sample_next_token(self, logits, sampler, sampling, top_p, alpha, debug):
        """Helper to sample next token with special token masking and fallback."""
        # Mask specials
        logits[..., self.encoder.START] = -float('inf')
        logits[..., self.encoder.PAD] = -float('inf')
        if hasattr(self.encoder, 'IM_START'):
            logits[..., self.encoder.IM_START] = -float('inf')
        if hasattr(self.encoder, 'IM_END'):
            logits[..., self.encoder.IM_END] = -float('inf')

        if not torch.isfinite(logits).any():
            logits = torch.zeros_like(logits)
            logits[..., :256] = 1.0
            logits[..., self.encoder.END] = 1.0
            logits[..., self.encoder.START] = -float('inf')
            logits[..., self.encoder.PAD] = -float('inf')


        next_token = sampler.sample(logits, sampling=sampling, top_p=top_p, alpha=alpha, debug=debug)

        return next_token


# ============= TRAINING =============

class StreamingGrokMetrics:
    """Minimal, robust grokking detector using scale-free metrics with smoothing."""

    def __init__(self, alpha_slow=0.99, alpha_fast=0.9,
                 k_loss: float = 3.0, k_perp: float = 2.0, k_improve: float = 5.0,
                 signal_smooth_alpha: float = 0.9,
                 train_fast_weight_threshold: float = 0.2, train_fast_weight_max_penalty: float = 0.7):
        # EMA timescales
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)

        # Hyperparameters
        self.k_loss = float(k_loss)
        self.k_perp = float(k_perp)
        self.k_improve = float(k_improve)
        self.signal_smooth_alpha = float(signal_smooth_alpha)
        self.train_fast_weight_threshold = float(train_fast_weight_threshold)
        self.train_fast_weight_max_penalty = float(train_fast_weight_max_penalty)

        # Slow/Fast EMAs for train/eval loss
        self.train_slow = None
        self.eval_slow = None
        self.train_fast = None
        self.eval_fast = None

        # Perplexity tracking (EMA) and best tracking (both hard min and slow EMA of best)
        self.perp_ema = None
        self.best_perp_seen = float('inf')
        self.best_perp_ema = None  # initialized on first update

        # Improvement (relative change of eval ema) with momentum
        self.prev_eval_ema = None
        self.improvement_rate = 0.0

        # Final signal smoothing
        self.signal_ema = None

    def update_train(self, loss: float):
        if self.train_slow is None:
            self.train_slow = float(loss)
            self.train_fast = float(loss)
        else:
            self.train_slow = self.alpha_slow * self.train_slow + (1 - self.alpha_slow) * float(loss)
            self.train_fast = self.alpha_fast * self.train_fast + (1 - self.alpha_fast) * float(loss)

    def update_eval(self, loss: float, perplexity: float):
        # Update eval loss EMAs
        if self.eval_slow is None:
            self.eval_slow = float(loss)
            self.eval_fast = float(loss)
            self.prev_eval_ema = float(loss)
            # Perplexity EMA initializes from provided perplexity
            self.perp_ema = float(perplexity)
            self.best_perp_seen = float(perplexity)
            self.best_perp_ema = float(perplexity)
        else:
            self.eval_slow = self.alpha_slow * self.eval_slow + (1 - self.alpha_slow) * float(loss)
            self.eval_fast = self.alpha_fast * self.eval_fast + (1 - self.alpha_fast) * float(loss)

            # Relative improvement rate (frequency-invariant)
            if self.prev_eval_ema is not None and self.prev_eval_ema > 0:
                relative_change = (self.prev_eval_ema - self.eval_slow) / self.prev_eval_ema
                self.improvement_rate = 0.9 * self.improvement_rate + 0.1 * relative_change
            self.prev_eval_ema = self.eval_slow

            # Perplexity EMA from provided value
            self.perp_ema = self.alpha_slow * self.perp_ema + (1 - self.alpha_slow) * float(perplexity)

            # Best tracking: hard min and slow EMA toward best
            self.best_perp_seen = min(self.best_perp_seen, self.perp_ema)
            best_beta = 0.99
            self.best_perp_ema = best_beta * self.best_perp_ema + (1 - best_beta) * self.best_perp_seen

    def get_signal(self) -> float:
        """Scale-free grokking signal based on ratios and relative improvements."""
        if self.train_slow is None or self.eval_slow is None:
            return 0.0

        if self.train_slow > 0.2:
            return 0.0

        eps = 1e-8

        # Loss ratio signal (eval vs train)
        loss_ratio = self.eval_slow / max(self.train_slow, eps)
        loss_gap_signal = max(0.0, min(1.0, (loss_ratio - 1.0) / max(self.k_loss, eps)))

        # Perplexity ratio signal (current vs best-ema)
        perp_signal = 0.0
        if self.perp_ema is not None and self.best_perp_ema is not None and self.best_perp_ema > 0:
            perp_ratio = self.perp_ema / max(self.best_perp_ema, eps)
            perp_signal = max(0.0, min(1.0, (perp_ratio - 1.0) / max(self.k_perp, eps)))

        # Improvement bonus (relative rate)
        improvement_bonus = max(0.0, self.improvement_rate * self.k_improve)

        # Fast vs slow divergence (positive when fast < slow during improvements)
        divergence = 0.0
        if self.eval_fast is not None and self.eval_slow > 0:
            divergence = max(0.0, (self.eval_slow - self.eval_fast) / max(self.eval_slow, eps))

        # Combine and smooth
        base_signal = 0.6 * loss_gap_signal + 0.4 * perp_signal
        dynamic_signal = min(0.5, improvement_bonus + 2.0 * divergence)
        signal_raw = min(1.0, base_signal + dynamic_signal)

        # Apply train_fast weight penalty: higher signal gets lower weight when train_fast > threshold
        if self.train_fast is not None:
            if self.train_fast > self.train_fast_weight_threshold:
                # Calculate penalty factor: 1.0 at threshold, decreasing to max_penalty at train_fast >= 1.0
                penalty_range = max(0.0, self.train_fast - self.train_fast_weight_threshold)
                max_penalty_range = 1.0 - self.train_fast_weight_threshold
                penalty_factor = 1.0 - (penalty_range / max_penalty_range) * self.train_fast_weight_max_penalty
                penalty_factor = max(0.0, min(1.0, penalty_factor))
                signal_raw *= penalty_factor

        if self.signal_ema is None:
            self.signal_ema = signal_raw
        else:
            a = self.signal_smooth_alpha
            self.signal_ema = a * self.signal_ema + (1 - a) * signal_raw

        return float(self.signal_ema)


class MultiGrokOptimizer:
    """Single unified optimizer managing multiple component-specific optimizers."""

    COMPONENT_CONFIG = {
        'embeddings': {'lr_scale': 1.0, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'attention': {'lr_scale': 0.8, 'weight_decay': 0.01, 'optimizer': 'grokadamw', 'grok': True, 'gradient_clipping': 0.25},
        'ffn': {'lr_scale': 1.2, 'weight_decay': 0.01, 'optimizer': 'grokadamw', 'grok': True, 'grok_scale': 1.2, 'gradient_clipping': 1.0},
        'biases': {'lr_scale': 2.0, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'layer_norm': {'lr_scale': 1.5, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'output': {'lr_scale': 0.3, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False, 'gradient_clipping': 0.1},
    }

    def __init__(self, model, base_lr=3e-4, weight_decay=0.01, gradient_clipping=1.0):
        self.model = model
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping

        # Extract number of layers from model
        self.n_layers = len(self.model.trf_blocks)

        # Regex patterns for parameter parsing
        import re
        self.patterns = {
            'layer_idx': re.compile(r'trf_blocks\.(\d+)\.')
        }

        # Single pass parameter classification
        self.param_groups = self._classify_parameters()

        # Create optimizers
        self.optimizers = {}
        self.grok_signals = {}
        self._create_optimizers()

        # Unified gradient clipping handle
        self.all_params = [p for group in self.param_groups.values() for p in group['params']]

        # Map parameters to layer indices once
        self._param_layer_map = self._build_param_layer_map()

    def _build_param_layer_map(self):
        """Build parameter -> layer index mapping."""
        param_to_layer = {}
        for name, param in self.model.named_parameters():
            if 'trf_blocks.' in name:
                match = self.patterns['layer_idx'].search(name)
                if match:
                    layer_idx = int(match.group(1))
                    param_to_layer[id(param)] = layer_idx
        return param_to_layer

    def _classify_parameters(self):
        """Single pass to classify all parameters by optimal component grouping."""
        groups = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Priority-based component classification (biases override everything)
            if name.endswith('.bias'):
                component = 'biases'
            elif 'norm' in name or 'scale' in name or 'shift' in name:
                component = 'layer_norm'
            elif any(x in name for x in ['W_query', 'W_key', 'W_value', 'out_proj']):
                component = 'attention'
            elif any(x in name for x in ['fc1', 'fc2', 'fc3', 'ffn']):
                component = 'ffn'
            elif 'out' in name and '_proj' not in name:
                component = 'output'
            elif 'embedding' in name or 'offset_embeddings' in name or 'rope' in name:
                component = 'embeddings'
            else:
                # Default to embeddings for unmatched parameters (like position embeddings, etc.)
                component = 'embeddings'

            # Get component config
            config = self.COMPONENT_CONFIG[component]

            # Use weight_decay from config (no longer needs_decay logic)
            weight_decay = config['weight_decay']

            # Create key (component only, since weight_decay is now per-component)
            key = component

            if key not in groups:
                groups[key] = {
                    'params': [],
                    'weight_decay': weight_decay,
                    'component': component,
                    'config': config
                }
            groups[key]['params'].append(param)

        return groups

    def _create_optimizers(self):
        """Create optimizers for each component type."""
        # Create one optimizer per component
        for component, group in self.param_groups.items():
            config = group['config']
            lr = self.base_lr * config['lr_scale']
            weight_decay = group['weight_decay']

            # Single param group per component with component-specific weight_decay
            param_groups = [{
                'params': group['params'],
                'weight_decay': weight_decay
            }]

            if config['optimizer'] == 'grokadamw' and config['grok']:
                # Layer-aware grok signal
                signal_fn = self._create_grok_signal(component)
                self.grok_signals[component] = signal_fn
                self.optimizers[component] = GrokAdamW(
                    param_groups,
                    lr=lr,
                    grokking_signal_fns=[signal_fn],
                    gradient_clipping=0,  # Handled externally for component-specific clipping
                    betas=(0.9, 0.999)
                )
            elif config['optimizer'] == 'adamw':
                self.optimizers[component] = torch.optim.AdamW(
                    param_groups,
                    lr=lr,
                    betas=(0.8, 0.95)
                )
            else:  # adam
                self.optimizers[component] = torch.optim.Adam(
                    param_groups,
                    lr=lr,
                    betas=(0.9, 0.98)
                )

    def _create_grok_signal(self, component):
        """Component-specific grok signal factory."""
        if component == 'ffn':
            # FFN gets stronger signal for memory grokking
            return lambda: self.base_grok_signal() * 1.2
        elif component == 'attention':
            # Attention gets standard signal
            return lambda: self.base_grok_signal()
        else:
            return lambda: 0.0

    def base_grok_signal(self):
        """Get base grokking signal from trainer metrics."""
        # This will be injected from trainer
        return getattr(self, '_current_grok_signal', 0.0)

    def set_grok_signal(self, signal: float, collect_grads: bool = False):
        """Set base grok signal with optional gradient collection."""
        # Ensure base_grok_signal() reads the latest value
        self._current_grok_signal = signal
        self._collect_grads = collect_grads and signal > 0.1

    @torch.no_grad()
    def step(self):
        """Single step for all optimizers with component-specific gradient clipping."""
        # Apply component-specific gradient clipping
        for component, opt in self.optimizers.items():
            config = self.COMPONENT_CONFIG[component]
            clipping_threshold = config.get('gradient_clipping', 0)

            if clipping_threshold > 0:
                # Get all parameters for this component's optimizer
                params = []
                for group in opt.param_groups:
                    params.extend(group['params'])
                if params:
                    torch.nn.utils.clip_grad_norm_(params, clipping_threshold)

        # Step all optimizers
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none=True):
        """Zero gradients for all optimizers."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Unified state dict."""
        return {
            f'{component}_opt': opt.state_dict()
            for component, opt in self.optimizers.items()
        }

    def load_state_dict(self, state_dict):
        """Load unified state dict."""
        for component, opt in self.optimizers.items():
            key = f'{component}_opt'
            if key in state_dict:
                opt.load_state_dict(state_dict[key])


class Trainer:
    """Efficient trainer with mixed precision and gradient accumulation."""

    def __init__(self, model: MesicapLM, lr=3e-4, warmup_steps=1000,
                 weight_decay=0.1, grad_accum_steps=1, device='cuda',
                 total_steps: int | None = None, ema_alpha=0.99,
                 gradient_clipping: float = 1.0,
                 quick_eval_k: int = 4):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel params - Total: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")

        print("\nModel architecture:")
        print(model)

        output = model.generate("The ", max_len=20)
        print(f"\nGenerated: {output}")

        self.metrics = StreamingGrokMetrics(alpha_slow=ema_alpha)
        self.opt = MultiGrokOptimizer(model, base_lr=lr, weight_decay=0.1)

        # OneCycle schedule with proper total steps and warmup fraction
        total_steps = total_steps or 10000

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt.optimizers['attention'],  # Use main optimizer for scheduling
            max_lr=lr,
            total_steps=total_steps,
            pct_start=min(warmup_steps/total_steps, 0.9)
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        self.grad_collect_interval = 50
        self.step = 0

        # Mixed precision forward (safe across devices)
        if self.device == 'cuda':
            autocast_ctx = torch.amp.autocast(device_type='cuda')
        elif self.device == 'cpu':
            autocast_ctx = torch.amp.autocast(device_type='cpu')
        else:
            # MPS or other devices: disable autocast by default for stability
            autocast_ctx = contextlib.nullcontext()

        self.autocast_ctx = autocast_ctx

        # Mini-eval averaging to reduce variance
        self.quick_eval_k = max(1, int(quick_eval_k))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: int = None) -> tuple[float, float]:
        """Fast evaluation with mixed precision.
        Returns: (avg_loss, perplexity)
        """
        self.model.eval()
        losses = []
        total_tokens = 0

        # Limit evaluation batches for speed
        eval_batches = enumerate(dataloader)
        if max_batches:
            eval_batches = itertools.islice(eval_batches, max_batches)

        for i, batch in eval_batches:
            batch = batch.to(self.device, non_blocking=True)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with self.autocast_ctx:
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.model.encoder.PAD,
                    reduction='none'
                )

                # Track valid tokens for accurate perplexity
                valid_mask = targets.reshape(-1) != self.model.encoder.PAD

                valid_loss = loss[valid_mask]

                if valid_loss.numel() > 0:
                    losses.append(valid_loss.mean().item())
                    total_tokens += valid_loss.numel()

        self.model.train()

        if not losses:
            return float('inf'), float('inf')

        avg_loss = np.mean(losses)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow

        return avg_loss, perplexity

    def update_metrics(self, train_loss: float, eval_loss: float, perplexity: float):
        """Update tracked metrics for grokking signal."""
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.perplexity = perplexity
        self.best_perplexity = min(self.best_perplexity, perplexity)

    def _unpack_batch(self, batch):
        """Unpack batch tuple (tokens, loss_mask) or return tensor directly."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            tokens, loss_mask = batch
            return tokens.to(self.device), loss_mask.to(self.device, non_blocking=True)
        else:
            return batch.to(self.device, non_blocking=True), None

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step with mixed precision."""
        batch, loss_mask = self._unpack_batch(batch)

        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with self.autocast_ctx:
            logits = self.model(inputs)
            if loss_mask is not None:
                # Use target-aligned mask: positions predict next token
                mask = loss_mask[:, 1:]
                per_pos_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.encoder.PAD,
                    reduction='none'
                ).view_as(mask)
                # Count only masked, non-PAD
                valid = (targets != self.encoder.PAD) & (mask > 0.5)
                if valid.any():
                    loss = (per_pos_loss[valid]).mean()
                else:
                    # Fallback to standard loss if no masked targets present
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=self.encoder.PAD
                    )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.encoder.PAD
                )

            # Scale for gradient accumulation
            loss = loss / self.grad_accum_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update streaming train loss immediately
        actual_loss = loss.item() * self.grad_accum_steps
        self.metrics.update_train(actual_loss)

        # Optimizer step
        # self.opt.set_grok_signal(self.metrics.get_signal())

        # Single step for all optimizers
        if (self.step + 1) % self.grad_accum_steps == 0:
            signal = self.metrics.get_signal()
            collect_grads = (self.step % self.grad_collect_interval == 0)
            self.opt.set_grok_signal(signal, collect_grads=collect_grads)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

        self.step += 1
        return loss.item()

    @torch.no_grad()
    def eval_step(self, batch: torch.Tensor) -> tuple[float, int]:
        """Single evaluation step - returns loss and valid token count."""
        batch, loss_mask = self._unpack_batch(batch)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with self.autocast_ctx:
            logits = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.encoder.PAD,
                reduction='none'
            )

            valid_mask = targets.reshape(-1) != self.encoder.PAD
            if loss_mask is not None:
                # Align to targets (next token)
                m = loss_mask[:, 1:].reshape(-1) > 0.5
                valid_mask = valid_mask & m
            valid_loss = loss[valid_mask]

            if valid_loss.numel() > 0:
                return valid_loss.mean().item(), valid_loss.numel()
            return 0.0, 0

    def quick_eval_update(self, val_batch: torch.Tensor):
        """Fast single-batch evaluation for streaming metrics (backward compatible)."""
        self.model.eval()
        loss, n_tokens = self.eval_step(val_batch)
        if n_tokens > 0:
            perplexity = math.exp(min(loss, 20))
            self.metrics.update_eval(loss, perplexity)
        self.model.train()

    def quick_eval_update_many(self, batches: List[torch.Tensor]):
        """Evaluate on multiple mini-batches and update metrics with weighted average."""
        self.model.eval()
        total_loss_times_tokens = 0.0
        total_tokens = 0
        for b in batches:
            loss, n_tokens = self.eval_step(b)
            if n_tokens > 0:
                total_loss_times_tokens += float(loss) * int(n_tokens)
                total_tokens += int(n_tokens)
        if total_tokens > 0:
            avg_loss = total_loss_times_tokens / max(1, total_tokens)
            perplexity = math.exp(min(avg_loss, 20))
            self.metrics.update_eval(avg_loss, perplexity)
        self.model.train()

    @property
    def encoder(self):
        return self.model.encoder

# ============= UTILS =============

def save_model(model: MesicapLM, path: Union[str, Path], config: Dict[str, Any] = None, checkpoint: bool = False, epoch: int = 0):
    """Save model and config - updated for RoPE model."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # Save config
    allowed_keys = {'d_model', 'n_heads', 'n_layers', 'max_len', 'rope_base'}
    default_config = {
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': len(model.trf_blocks),
        'max_len': model.rope.max_seq_len,
        'rope_base': model.rope.base,
    }
    # Filter incoming config to contain only model constructor keys; fill missing from defaults
    if config is not None:
        filtered = {k: v for k, v in config.items() if k in allowed_keys}
        for k in allowed_keys:
            if k not in filtered:
                filtered[k] = default_config[k]
        config = filtered
    else:
        config = default_config

    with open(path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save weights
    if checkpoint:
        torch.save(model.state_dict(), path / f'model_checkpoint_{epoch}.pt')
    else:
        torch.save(model.state_dict(), path / 'model.pt')

def load_model(path: Union[str, Path], device='cuda') -> MesicapLM:
    """Load model from checkpoint."""
    path = Path(path)

    # Load config
    with open(path / 'config.json', 'r') as f:
        loaded = json.load(f)
        # Be robust to extra metadata keys in older checkpoints
        allowed_keys = {'d_model', 'n_heads', 'n_layers', 'max_len', 'rope_base'}
        config = {k: v for k, v in loaded.items() if k in allowed_keys}

    # Create model
    model = MesicapLM(**config)

    # Load weights
    model.load_state_dict(torch.load(path / 'model.pt', map_location=device))

    return model.to(device)

# ============= MAIN TRAINING LOOP =============

class DatasetBuilder:
    """Handles dataset and DataLoader creation for training and validation."""

    def __init__(self, device: str):
        self.device = device

    def _get_dataloader_kwargs(self, for_eval: bool = False):
        """Get device-optimized DataLoader arguments."""
        if self.device == 'cuda':
            workers = min(8, (os.cpu_count() or 8))
            return dict(num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, shuffle=not for_eval)
        elif self.device == 'mps':
            return dict(num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None, shuffle=not for_eval)
        else:
            workers = min(4, (os.cpu_count() or 4))
            return dict(num_workers=workers, pin_memory=False, persistent_workers=True, prefetch_factor=2, shuffle=not for_eval)

    def _build_packed_dataset(self, roots: List[Union[str, Path]], seq_length: int, mixing_policy: str, mixing_weights: Optional[List[float]]):
        """Build packed dataset from multiple roots."""
        ds_per_root = []
        for r in roots:
            shard_dirs = discover_shards(str(r))
            if not shard_dirs:
                continue
            shard_datasets = [
                PackedTriadicShardDataset(sd, seq_length=seq_length, stride=seq_length//2, pad_id=TriadicEncoder.PAD)
                for sd in shard_dirs
            ]
            if len(shard_datasets) == 1:
                ds_per_root.append(shard_datasets[0])
            else:
                ds_per_root.append(torch.utils.data.ConcatDataset(shard_datasets))

        if not ds_per_root:
            raise ValueError("No shards found in provided packed_dirs")

        if len(ds_per_root) == 1:
            return ds_per_root[0]

        return MixedPackedTriadicDataset(ds_per_root, policy=mixing_policy, weights=mixing_weights)

    def create_datasets_and_loaders(self, encoder, data_path, val_path, packed_dirs, val_packed_dirs,
                                   seq_length, batch_size, mixing_policy, mixing_weights):
        """Create training and validation datasets/loaders for non-curriculum training."""
        use_packed = False
        train_dataset = None
        val_dataset = None

        if packed_dirs is not None and len(packed_dirs) > 0:
            train_dataset = self._build_packed_dataset([str(p) for p in packed_dirs], seq_length, mixing_policy, mixing_weights)
            use_packed = True
            print(f"Using packed datasets for training. policy={mixing_policy}, weights={mixing_weights}")
        else:
            # Auto-detect packed shards under data_path directory
            dp = Path(data_path)
            if dp.is_dir():
                shard_dirs = discover_shards(str(dp))
                if shard_dirs:
                    train_dataset = self._build_packed_dataset([dp], seq_length, mixing_policy, mixing_weights)
                    use_packed = True
                    print(f"Auto-detected packed shards under {dp}")

        if train_dataset is None:
            # Fallback to simple text dataset
            train_dataset = XORDataset(data_path, seq_length, encoder=encoder)

        train_loader = DataLoader(train_dataset, batch_size, **self._get_dataloader_kwargs(for_eval=False))

        # Validation dataset
        if val_packed_dirs is not None and len(val_packed_dirs) > 0:
            val_dataset = self._build_packed_dataset([str(p) for p in val_packed_dirs], seq_length, mixing_policy, mixing_weights)
        elif val_path is not None:
            vp = Path(val_path)
            if vp.is_dir() and discover_shards(str(vp)):
                val_dataset = self._build_packed_dataset([vp], seq_length, mixing_policy, mixing_weights)
            else:
                val_dataset = XORDataset(val_path, seq_length, encoder=encoder)
        else:
            if use_packed:
                # Heuristic: reuse train roots for eval with full-stride windows (no shuffle in loader)
                if packed_dirs is not None and len(packed_dirs) > 0:
                    val_dataset = self._build_packed_dataset([str(p) for p in packed_dirs], seq_length, mixing_policy, mixing_weights)
                else:
                    val_dataset = self._build_packed_dataset([dp], seq_length, mixing_policy, mixing_weights)
            else:
                # Use 10% of training data with different stride for pseudo-validation
                val_dataset = XORDataset(data_path, seq_length, stride=seq_length, encoder=encoder)

        val_loader = DataLoader(val_dataset, batch_size, **self._get_dataloader_kwargs(for_eval=True))

        return train_loader, val_loader

    def create_curriculum_datasets_and_loaders(self, encoder, curriculum_stage: Dict[str, Any], seq_length: int, batch_size: int,
                                              mixing_policy: str, mixing_weights: Optional[List[float]]):
        """Create datasets and loaders for a curriculum stage."""
        stage_name = curriculum_stage.get('name', 'stage')
        stage_steps = curriculum_stage.get('steps', None)
        roots = curriculum_stage.get('packed_roots') or curriculum_stage.get('packed_dirs') or []
        roots = [str(p) for p in roots]
        if not roots:
            raise ValueError(f"Curriculum stage '{stage_name}' requires 'packed_roots' or 'packed_dirs'")

        # Ensure train/val split exists (create if missing) for each dataset root
        val_ratio = float(curriculum_stage.get('val_ratio', 0.05))
        split_pairs = []
        for r in roots:
            tr_dir, va_dir = ensure_train_val_split(r, val_ratio=val_ratio, seed=42)
            split_pairs.append((tr_dir, va_dir))

        # Build mixed datasets across all roots for this stage
        if stage_steps is not None:
            # Estimate windows needed for training and small validation
            # Each step consumes one batch; each batch uses `batch_size` windows
            train_windows_needed = int(stage_steps) * int(batch_size)
            # For validation: approximate 100 batches like quick eval upper bound
            val_batches = 100
            val_windows_needed = val_batches * int(batch_size)
            print(f"[stage] {stage_name}: capping dataset selection -> train_windows~{train_windows_needed}, val_windows~{val_windows_needed}")
            stage_train_ds = build_capped_mixed_dataset(
                roots, split='train', seq_length=seq_length, pad_id=TriadicEncoder.PAD,
                windows_needed=train_windows_needed, policy=mixing_policy, weights=mixing_weights
            )
            stage_val_ds = build_capped_mixed_dataset(
                roots, split='val', seq_length=seq_length, pad_id=TriadicEncoder.PAD,
                windows_needed=val_windows_needed, policy=mixing_policy, weights=mixing_weights
            )
        else:
            stage_train_ds = build_mixed_dataset(roots, split='train', seq_length=seq_length, pad_id=TriadicEncoder.PAD,
                                                 policy=mixing_policy, weights=mixing_weights)
            stage_val_ds = build_mixed_dataset(roots, split='val', seq_length=seq_length, pad_id=TriadicEncoder.PAD,
                                               policy=mixing_policy, weights=mixing_weights)

        # Debug counts for visibility
        print(f"[stage] {stage_name}: roots={len(roots)}; building DataLoaders")

        stage_train_loader = DataLoader(stage_train_ds, batch_size, **self._get_dataloader_kwargs(for_eval=False))
        stage_val_loader = DataLoader(stage_val_ds, batch_size, **self._get_dataloader_kwargs(for_eval=True))

        return stage_train_loader, stage_val_loader


class CurriculumManager:
    """Manages curriculum learning stages and their execution."""

    def __init__(self, dataset_builder: DatasetBuilder, seq_length: int, batch_size: int,
                 mixing_policy: str, mixing_weights: Optional[List[float]]):
        self.dataset_builder = dataset_builder
        self.encoder = None  # will be set by constructor in train()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.mixing_policy = mixing_policy
        self.mixing_weights = mixing_weights

    def run_curriculum(self, curriculum: List[Dict[str, Any]], stage_runner):
        """Run all curriculum stages."""
        for stage in curriculum:
            stage_name = stage.get('name', 'stage')
            stage_epochs = int(stage.get('epochs', 1))
            stage_lr = float(stage.get('lr', 3e-4))
            stage_steps = stage.get('steps', None)
            if stage_steps is not None:
                stage_steps = int(stage_steps)
                if stage_steps <= 0:
                    raise ValueError(f"Curriculum stage '{stage_name}' has invalid steps={stage_steps}; must be > 0")
                print(f"[stage] {stage_name}: limiting to {stage_steps} steps")

            # Create datasets and loaders for this stage
            stage_train_loader, stage_val_loader = self.dataset_builder.create_curriculum_datasets_and_loaders(
                self.encoder, stage, self.seq_length, self.batch_size, self.mixing_policy, self.mixing_weights
            )

            # Run the stage
            stage_runner.run_stage(stage_name, stage_train_loader, stage_val_loader, stage_epochs, stage_lr, stage_steps)


class StageRunner:
    """Handles execution of individual training stages."""

    def __init__(self, model: MesicapLM, device: str, eval_interval: int, ema_alpha: float,
                 test_prompt, model_path: Union[str, Path], run):
        self.model = model
        self.device = device
        self.eval_interval = eval_interval
        self.ema_alpha = ema_alpha
        self.test_prompt = test_prompt
        self.model_path = model_path
        self.run = run

    def _setup_stage_freezing(self, stage_name: str):
        """Apply stage-specific parameter freezing."""
        if stage_name == 'school':
            # Freeze lower layers and bit projection
            n_freeze = int(len(self.model.trf_blocks) * 0.8)  # Freeze 60% of layers

            # Freeze input projection stack (align with current module names)
            for p in self.model.bit_extract.parameters():
                p.requires_grad = False
            for p in self.model.embed_proj.parameters():
                p.requires_grad = False

            # Freeze RoPE (positional encoding)
            for param in self.model.rope.parameters():
                param.requires_grad = False

            # Freeze lower transformer layers
            for i in range(n_freeze):
                for param in self.model.trf_blocks[i].parameters():
                    param.requires_grad = False

    def run_stage(self, stage_name: str, train_loader, val_loader, epochs: int, lr: float, steps: Optional[int] = None):
        """Run a single training stage."""
        steps_per_epoch = max(len(train_loader), 1)
        planned_total = steps_per_epoch * epochs
        effective_total = planned_total if steps is None else min(planned_total, int(steps))
        print(f"[stage] {stage_name}: steps_per_epoch={steps_per_epoch}, planned_total={planned_total}, effective_total={effective_total}")

        # Apply stage-specific modifications
        self._setup_stage_freezing(stage_name)

        # Create trainer for this stage
        trainer = Trainer(
            self.model,
            lr=lr,
            device=self.device,
            total_steps=effective_total,
            warmup_steps=min(1000, effective_total // 10),
            ema_alpha=self.ema_alpha,
            quick_eval_k=8
        )

        val_iter = itertools.cycle(val_loader)
        global_step = 0
        reached_cap = False

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{epochs}")
            for batch_idx, train_batch in enumerate(pbar):
                loss = trainer.train_step(train_batch)
                global_step += 1

                # Quick eval update
                if epoch > 2 and global_step % self.eval_interval == 0:
                    if trainer.quick_eval_k <= 1:
                        val_batch = next(val_iter)
                        trainer.quick_eval_update(val_batch)
                    else:
                        batches = list(itertools.islice(val_iter, int(trainer.quick_eval_k)))
                        trainer.quick_eval_update_many(batches)

                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'train/slow': f"{trainer.metrics.train_slow:.4f}" if trainer.metrics.train_slow else "N/A",
                    'train/fast': f"{trainer.metrics.train_fast:.4f}" if trainer.metrics.train_fast else "N/A",
                    'eval/slow': f"{trainer.metrics.eval_slow:.4f}" if trainer.metrics.eval_slow else "N/A",
                    'eval/fast': f"{trainer.metrics.eval_fast:.4f}" if trainer.metrics.eval_fast else "N/A",
                    'perp': f"{trainer.metrics.perp_ema:.2f}" if trainer.metrics.perp_ema else "N/A",
                    'grok': f"{trainer.metrics.get_signal():.3f}"
                })

                self.run.log({
                    'train/loss': loss,
                    'train/ema_slow': trainer.metrics.train_slow,
                    'train/ema_fast': trainer.metrics.train_fast,
                    'validation/ema_slow': trainer.metrics.eval_slow,
                    'validation/ema_fast': trainer.metrics.eval_fast,
                    'perplexity/ema': trainer.metrics.perp_ema,
                    'grok': trainer.metrics.get_signal()
                })

                if global_step % 1000 == 0:
                    self.model.eval()
                    sample = self.model.generate(self.test_prompt, max_len=60)
                    print(f"\nSample ({stage_name}): {sample}\n")
                    self.model.train()

                if steps is not None and global_step >= int(steps):
                    print(f"[stage] {stage_name}: reached step cap ({global_step}/{int(steps)}); ending stage after summary.")
                    reached_cap = True
                    break

            # Full eval snapshot
            self.model.eval()
            eval_losses = []
            for i, batch in enumerate(itertools.islice(val_loader, 100)):
                loss, n_tokens = trainer.eval_step(batch)
                if n_tokens > 0:
                    eval_losses.append(loss)
            if eval_losses:
                avg_eval_loss = np.mean(eval_losses)
                epoch_perplexity = math.exp(min(avg_eval_loss, 20))
                print(f"\n{stage_name} Epoch {epoch+1} Summary:")
                print(f"  Streaming - Train EMA: {trainer.metrics.train_slow:.4f}, "
                      f"Eval EMA: {trainer.metrics.eval_slow:.4f}, "
                      f"Perp EMA: {trainer.metrics.perp_ema:.2f}")
                print(f"  Full Eval - Loss: {avg_eval_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
                print(f"  Grokking Signal: {trainer.metrics.get_signal():.3f}")
                self.run.log({
                    'validation/loss': avg_eval_loss,
                    'validation/perplexity': epoch_perplexity,
                    'validation/grok': trainer.metrics.get_signal()
                })

            # Save checkpoint
            save_model(self.model, self.model_path, {
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': len(self.model.trf_blocks),
                'max_len': self.model.rope.max_seq_len,
                'rope_base': self.model.rope.base,
                'best_perp_seen': trainer.metrics.best_perp_seen,
                'stage': stage_name,
                'epoch': epoch + 1
            }, checkpoint=True, epoch=epoch + 1)

            self.model.eval()
            sample = self.model.generate(self.test_prompt, max_len=60)
            print(f"\nSample ({stage_name}): {sample}\n")

            if reached_cap:
                break


def train(
    data_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,  # Optional separate validation set
    model_path: Union[str, Path] = "xor_model",
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    seq_length: int = 512,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 3e-4,
    rope_base: int = 10000,
    eval_interval: int = 10,  # Eval every N training steps
    ema_alpha: float = 0.99,  # EMA decay rate
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    compile_model: bool = False,
    packed_dirs: Optional[List[Union[str, Path]]] = None,  # Multiple packed dataset roots
    val_packed_dirs: Optional[List[Union[str, Path]]] = None,
    mixing_policy: str = 'round_robin',  # 'round_robin' or 'weighted'
    mixing_weights: Optional[List[float]] = None,
    test_prompt: str = "The ",
    curriculum: Optional[List[Dict[str, Any]]] = None,
):
    """Complete training pipeline."""

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="energyx-hologram",
        # Set the wandb project where this run will be logged.
        project="kulles",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "MesicapResonance",
            "epochs": epochs,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "rope_base": rope_base,
        },
    )

    print(f"Training on {device}")

    tokenizer = AutoTokenizer.from_pretrained("./est_superbpe")
    encoder = tokenizer

    MESICAP_CONFIG = {
        "batch_size": batch_size,
        "vocab_size": encoder.vocab_size,
        "context_length": seq_length,
        "emb_dim": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "head_dim": d_model // n_heads,
        "hidden_dim": d_model,
        "qk_norm": False,
        "n_kv_groups": 4,
        "rope_base": rope_base,
        "dtype": torch.float32,
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": d_model,
    }

    # Create model once and reuse across stages
    model = MesicapLM(MESICAP_CONFIG, encoder)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Create helper classes
    dataset_builder = DatasetBuilder(device)
    curriculum_manager = CurriculumManager(dataset_builder, seq_length, batch_size, mixing_policy, mixing_weights)
    curriculum_manager.encoder = encoder
    stage_runner = StageRunner(
        model, device, eval_interval, ema_alpha, test_prompt, model_path, run
    )

    # Run training
    using_curriculum = bool(curriculum and len(curriculum) > 0)
    if using_curriculum:
        curriculum_manager.run_curriculum(curriculum, stage_runner)
    else:
        # Single-stage legacy flow
        train_loader, val_loader = dataset_builder.create_datasets_and_loaders(
            encoder,
            data_path, val_path, packed_dirs, val_packed_dirs,
            seq_length, batch_size, mixing_policy, mixing_weights
        )
        stage_runner.run_stage('main', train_loader, val_loader, epochs, lr)

    run.finish()
    save_model(model, model_path)
    return model


def run_test_generations(model: MesicapLM, input_text: str, max_len: int = 512):
    model.eval()
    output = model.generate(input_text, max_len=max_len)
    print(f"\nGenerated (top-k, temp=1.0):\n{output}\n")
    output = model.generate(input_text, max_len=max_len, temp=0.5)
    print(f"\nGenerated (top-k, temp=0.5):\n{output}\n")

    output = model.generate(input_text, max_len=max_len, sampling='top_h')
    print(f"\nGenerated (top-h, temp=1.0):\n{output}\n")
    output = model.generate(input_text, max_len=max_len, sampling='top_h', temp=0.5)
    print(f"\nGenerated (top-h, temp=0.5):\n{output}\n")

    #output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True)
    #print(f"\nGenerated (chat template, temp=1.0):\n{output}\n")
    #output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True, sampling='top_h')
    #print(f"\nGenerated (chat template, top-h):\n{output}\n")

# ============= EXAMPLE USAGE =============

if __name__ == "__main__":

    test_run = True
    load_and_test = False
    test_run_shakespeare = False

    if load_and_test:
        model = load_model("xor_model", "mps")
        run_test_generations(model, "The world is a cold place.", max_len=512)
        exit()

    if test_run:
        if test_run_shakespeare:
            input_text = """
    ALL:
    Content, content.

    MENENIUS:
    O sir, you are not right: have you not known
    The worthiest men have done't?

    CORIOLANUS:
    """.strip()

            model = train(
                "data/tiny_shakespeare.txt",
                seq_length=512,
                batch_size=8,
                lr=0.0003,
                epochs=15,
                d_model=128,
                n_heads=8,
                n_layers=6,
                rope_base=10000,
                test_prompt="The "
            )

            output = model.generate(input_text, max_len=200)
            print(f"\nGenerated:\n{output}")

        else:
            model = train(
            "datasets/packed",
            model_path="triadic_test",
            seq_length=512,
            eval_interval=25,
            batch_size=32,
            epochs=5,
            d_model=128,
            n_heads=8,
            n_layers=6,
            rope_base=10000,
            test_prompt="The ",
            mixing_policy='round_robin_wrap',
            curriculum=[
                {
                    'name': 'pretrain',
                    'packed_roots': [
                        'datasets/packed/triadic-tiny-stories-512',
                    ],
                    'epochs': 10,
                    #'steps': 1500,
                    'lr': 2e-3
                }
            ]
        )
    else:
        model = train(
            "datasets/packed",
            seq_length=3096,
            batch_size=4,
            epochs=5,
            d_model=512,
            n_heads=8,
            n_layers=8,
            rope_base=10000,
            test_prompt="The ",
            mixing_policy='round_robin_wrap',
            curriculum=[
                {
                    'name': 'pretrain',
                    'packed_roots': [
                        'datasets/packed/tiny-stories',
                        'datasets/packed/orca-pre',
                        'datasets/packed/tiny-lessons',
                        'datasets/packed/tiny-textbooks',
                        'datasets/packed/tiny-superwiki',
                        'datasets/packed/arxiver'
                    ],
                    #'epochs': 1,
                    'steps': 5000,
                    'lr': 3e-4
                },
                {
                    'name': 'school',
                    'packed_roots': [
                        'datasets/packed/orca-inst',
                    ],

                    'steps': 4000,
                    'lr': 5e-5
                }
            ]
        )

    #output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True)
    #print(f"\nGenerated: {output}")

    run_test_generations(model, "The world is a cold place.", max_len=512)
