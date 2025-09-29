"""
8-Bit XOR Language Model - Complete Implementation
Fast, elegant, vocabulary-free language modeling
"""

import contextlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.utils.data import DataLoader
from tqdm import tqdm

from datatrove.utils.dataset import DatatroveFolderDataset
from grokadamw import GrokAdamW
from transformers import AutoTokenizer

import os
import wandb

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
np.random.seed(42)

os.environ['PYTORCH_MPS_FAST_MATH'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class TokenizerAdapter:
    """Unified wrapper around Hugging Face AutoTokenizer loaded from a JSON file."""

    def __init__(self, tokenizer_path: Union[str, Path]):
        self.tokenizer_path = Path(tokenizer_path)
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

        if self.tokenizer_path.is_file():
            pretrained_root = self.tokenizer_path.parent.as_posix()
            tokenizer_file = self.tokenizer_path.name
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_root,
                tokenizer_file=tokenizer_file,
                use_fast=True,
                trust_remote_code=False,
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path.as_posix(),
                use_fast=True,
                trust_remote_code=False,
            )

        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token is not None:
                print("[TokenizerAdapter][debug] Assigning pad_token to eos_token")
                self._tokenizer.pad_token = self._tokenizer.eos_token
            elif self._tokenizer.bos_token is not None:
                print("[TokenizerAdapter][debug] Assigning pad_token to bos_token")
                self._tokenizer.pad_token = self._tokenizer.bos_token
            else:
                raise ValueError("Tokenizer must define at least one of pad/eos/bos tokens.")

        if self._tokenizer.bos_token_id is None and self._tokenizer.cls_token is not None:
            print("[TokenizerAdapter][debug] Using cls_token as bos_token")
            self._tokenizer.bos_token = self._tokenizer.cls_token

        if self._tokenizer.eos_token_id is None and self._tokenizer.sep_token is not None:
            print("[TokenizerAdapter][debug] Using sep_token as eos_token")
            self._tokenizer.eos_token = self._tokenizer.sep_token

        for label in ("pad", "bos", "eos"):
            if getattr(self._tokenizer, f"{label}_token_id") is None:
                raise ValueError(f"Tokenizer missing required {label}_token_id after initialization.")

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)

    @property
    def bos_token_id(self) -> int:
        return int(self._tokenizer.bos_token_id)

    @property
    def eos_token_id(self) -> int:
        return int(self._tokenizer.eos_token_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        token_ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        return self._tokenizer.batch_decode(batch_ids.tolist(), skip_special_tokens=skip_special_tokens)

    def __repr__(self) -> str:
        return f"TokenizerAdapter(path={self.tokenizer_path}, vocab={self.vocab_size})"

# Fast bit operations via lookup tables
_XOR_LUT = np.array([[i ^ j for j in range(256)] for i in range(256)], dtype=np.uint8)
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
_GRAY_LUT = np.array([i ^ (i >> 1) for i in range(256)], dtype=np.uint8)
_GRAY_INV = np.zeros(256, dtype=np.uint8)
for i in range(256):
    _GRAY_INV[_GRAY_LUT[i]] = i



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

    kernel void fused_silu_multiply(
        device const float* gate_src [[buffer(0)]],
        device const float* carrier_src [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant int& total_elems [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= total_elems) {
            return;
        }
        float v_gate = gate_src[gid];
        float v_carrier = carrier_src[gid];
        float sig = 1.0f / (1.0f + exp(-v_gate));
        out[gid] = (v_gate * sig) * v_carrier;
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
        self._fused_silu_kernel = getattr(self.metal_kernels, 'fused_silu_multiply', None)
        self._dbg_step = 0

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Use MPS-optimized operations
        profile_ctx = torch.mps.profiler.profile if not os.getenv("MESICAP_DISABLE_INTERNAL_MPS_PROFILE") else contextlib.nullcontext
        with profile_ctx():
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
            counts = self.expert_counts.clamp_max(self.max_tokens_per_expert)
            active = torch.nonzero(counts > 0, as_tuple=False).flatten()
            if active.numel() > 0:
                active_list = active.tolist()
                counts_list = [int(counts[eid].item()) for eid in active_list]
                max_tokens = max(counts_list)

                if max_tokens > 0:
                    num_active = len(active_list)
                    inputs_tensor = torch.zeros(
                        num_active,
                        max_tokens,
                        self.emb_dim,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    mask = torch.zeros(
                        num_active,
                        max_tokens,
                        device=x.device,
                        dtype=torch.bool,
                    )

                    token_indices = []
                    routing_weights = []

                    for idx, expert_id in enumerate(active_list):
                        n_tok = counts_list[idx]
                        expert_in = self.expert_inputs[expert_id, :n_tok, :]
                        inputs_tensor[idx, :n_tok] = expert_in
                        mask[idx, :n_tok] = True
                        token_indices.append(self.expert_token_indices[expert_id, :n_tok].to(dtype=torch.long))
                        routing_weights.append(self.expert_weights[expert_id, :n_tok])

                    fc1_weight = torch.stack([self.fc1[eid].weight for eid in active_list])
                    fc2_weight = torch.stack([self.fc2[eid].weight for eid in active_list])
                    fc3_weight = torch.stack([self.fc3[eid].weight for eid in active_list])

                    gate_proj = torch.matmul(inputs_tensor, fc1_weight.transpose(-2, -1))
                    carrier_proj = torch.matmul(inputs_tensor, fc2_weight.transpose(-2, -1))

                    hidden_proj = torch.zeros_like(gate_proj)
                    for idx, n_tok in enumerate(counts_list):
                        if n_tok == 0:
                            continue
                        gate_slice = gate_proj[idx, :n_tok].contiguous()
                        carrier_slice = carrier_proj[idx, :n_tok].contiguous()
                        fused_slice = hidden_proj[idx, :n_tok]
                        self._fused_silu_kernel(
                            gate_slice.reshape(-1),
                            carrier_slice.reshape(-1),
                            fused_slice.reshape(-1),
                            gate_slice.numel(),
                        )

                    expert_proj = torch.matmul(hidden_proj, fc3_weight.transpose(-2, -1))

                    expert_outputs = []
                    expert_token_idx = []
                    expert_weight_list = []
                    for idx, n_tok in enumerate(counts_list):
                        if n_tok == 0:
                            continue
                        expert_outputs.append(expert_proj[idx, :n_tok])
                        expert_token_idx.append(token_indices[idx])
                        expert_weight_list.append(routing_weights[idx])

                    if expert_outputs:
                        cat_out = torch.cat(expert_outputs, dim=0)
                        cat_idx = torch.cat(expert_token_idx, dim=0)
                        cat_w = torch.cat(expert_weight_list, dim=0).unsqueeze(-1)
                        out_flat.index_add_(0, cat_idx, cat_out * cat_w)

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

class GatedLinearAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim=None, dtype=None, eps: float = 1e-6):
        super().__init__()
        self.num_heads = int(num_heads)
        if head_dim is None:
            assert d_in % self.num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // self.num_heads
        self.head_dim = int(head_dim)
        self.d_out = self.num_heads * self.head_dim
        self.eps = float(eps)

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        self.W_gate = nn.Linear(d_in, d_in, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):  # noqa: ARG002
        b, seq_len, _ = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = F.relu(q) + self.eps
        k = F.relu(k) + self.eps

        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        v_f = v.to(torch.float32)

        outer = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)
        kv_prefix = torch.cumsum(outer, dim=2)
        k_prefix = torch.cumsum(k_f, dim=2)

        numerator = torch.einsum('bhtd,bhtdm->bhtm', q_f, kv_prefix)
        denominator = torch.sum(q_f * k_prefix, dim=-1, keepdim=True)
        denominator = torch.clamp(denominator, min=self.eps)

        attn = numerator / denominator

        attn = attn.transpose(1, 2).reshape(b, seq_len, self.d_out)
        attn = self.out_proj(attn)

        gate = torch.sigmoid(self.W_gate(x))
        attn = attn * gate

        return attn


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_in, num_heads, window_size: int = 128, head_dim=None, dtype=None, qk_norm=False):
        super().__init__()
        self.num_heads = int(num_heads)
        if head_dim is None:
            assert d_in % self.num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // self.num_heads
        self.head_dim = int(head_dim)
        self.d_out = self.num_heads * self.head_dim
        self.window_size = int(window_size)

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_gate = nn.Linear(d_in, self.num_heads, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        self._mask_base_cache: dict[int, torch.Tensor] = {}
        self._mask_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _get_base_mask(self, seq_len: int) -> torch.Tensor:
        mask = self._mask_base_cache.get(seq_len)
        if mask is None:
            idx = torch.arange(seq_len, dtype=torch.int32)
            distance = idx.unsqueeze(0) - idx.unsqueeze(1)
            allow = (distance <= 0) & (distance >= -self.window_size + 1)
            base = torch.zeros((seq_len, seq_len), dtype=torch.bool)
            base.copy_(allow)
            mask = base
            self._mask_base_cache[seq_len] = mask
        return mask

    def _get_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (seq_len, device, dtype)
        cached = self._mask_cache.get(key)
        if cached is None:
            base = self._get_base_mask(seq_len)
            if base.device != device:
                base = base.to(device, copy=True)
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
            mask = mask.masked_fill(base, 0.0)
            mask = mask.unsqueeze(0).unsqueeze(0)
            self._mask_cache[key] = mask
            return mask
        return cached

    def forward(self, x, mask, cos, sin):  # noqa: ARG002
        b, seq_len, _ = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_mask = self._get_mask(seq_len, x.device, q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
        )

        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            print("[SWA][warn] NaN/Inf detected in attention output")

        gate = torch.sigmoid(self.W_gate(x))
        gate = gate.transpose(1, 2).unsqueeze(-1)
        attn_out = attn_out * gate

        context = attn_out.transpose(1, 2).reshape(b, seq_len, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        if cfg.get("use_gla_first_layer", False) and layer_idx == 0:
            self.att = GatedLinearAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                dtype=cfg["dtype"]
            )
        elif cfg.get("use_swa_layers", False) and layer_idx in cfg.get("swa_layers", []):
            self.att = SlidingWindowAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                dtype=cfg["dtype"],
                window_size=cfg.get("swa_window", 128),
                qk_norm=cfg["qk_norm"],
            )
        else:
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
    def __init__(self, cfg, tokenizer: TokenizerAdapter):
        super().__init__()
        self.tokenizer = tokenizer
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock({**cfg, "num_experts": 0}, layer_idx=0)
                if i == 0
                else TransformerBlock(cfg, layer_idx=i)
                for i in range(cfg["n_layers"])
            ]
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

        input_token_ids = self.tokenizer.encode(prompt)

        if input_token_ids.numel() > 0 and int(input_token_ids[-1]) == self.tokenizer.eos_token_id:
            input_token_ids = input_token_ids[:-1]
        x = input_token_ids.detach().clone().unsqueeze(0).to(device, non_blocking=True)

        # Simple generation without cache
        for _ in range(max_len):
            if x.size(1) > self.cfg["context_length"]:
                x = x[:, -self.cfg["context_length"]:]

            logits = self(x)[:, -1] / max(temp, 1e-6)
            next_token = self._sample_next_token(logits, sampler, sampling, top_p, alpha, debug)

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            x = torch.cat([x, next_token], dim=1)


        return self.tokenizer.decode(x[0].cpu().tolist())

    def _sample_next_token(self, logits, sampler, sampling, top_p, alpha, debug):
        """Helper to sample next token with special token masking and fallback."""
        # Mask specials
        logits[..., self.tokenizer.bos_token_id] = -float('inf')
        logits[..., self.tokenizer.pad_token_id] = -float('inf')

        if not torch.isfinite(logits).any():
            logits = torch.zeros_like(logits)
            logits[..., :256] = 1.0
            logits[..., self.tokenizer.eos_token_id] = 1.0
            logits[..., self.tokenizer.bos_token_id] = -float('inf')
            logits[..., self.tokenizer.pad_token_id] = -float('inf')


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
        'gate': {'lr_scale': 1.0, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
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
            elif any(x in name for x in ['fc1', 'fc2', 'fc3', 'ff']):
                component = 'ffn'
            elif 'W_gate' in name:
                component = 'gate'
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
                    betas=(0.8, 0.95),
                    fused=True
                )
            else:  # adam
                self.optimizers[component] = torch.optim.Adam(
                    param_groups,
                    lr=lr,
                    betas=(0.9, 0.98),
                    fused=True
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
                 weight_decay=0.1, device='cuda',
                 total_steps: int | None = None, ema_alpha=0.99,
                 gradient_clipping: float = 1.0,
                 quick_eval_k: int = 4):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = self.model.tokenizer

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
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='none'
                )

                # Track valid tokens for accurate perplexity
                valid_mask = targets.reshape(-1) != self.tokenizer.pad_token_id

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
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='none'
                ).view_as(mask)
                # Count only masked, non-PAD
                valid = (targets != self.tokenizer.pad_token_id) & (mask > 0.5)
                if valid.any():
                    loss = (per_pos_loss[valid]).mean()
                else:
                    # Fallback to standard loss if no masked targets present
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_detached = loss.detach()
        actual_loss = None

        # Optimizer step
        # self.opt.set_grok_signal(self.metrics.get_signal())

        # Single step for all optimizers
        if (self.step + 1):
            signal = self.metrics.get_signal()
            collect_grads = (self.step % self.grad_collect_interval == 0)
            self.opt.set_grok_signal(signal, collect_grads=collect_grads)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

        if self.device == 'mps':
            torch.mps.synchronize()

        actual_loss = float(loss_detached)
        self.metrics.update_train(actual_loss)

        self.step += 1
        return actual_loss

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
                ignore_index=self.tokenizer.pad_token_id,
                reduction='none'
            )

            valid_mask = targets.reshape(-1) != self.tokenizer.pad_token_id
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
        return self.model.tokenizer

# ============= UTILS =============

def save_model(model: MesicapLM, path: Union[str, Path], config: Dict[str, Any] = None, checkpoint: bool = False, epoch: int = 0):
    """Save model and config - updated for RoPE model."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # Save config
    metadata = {
        'model_config': model.cfg,
        'tokenizer_path': getattr(model.tokenizer, 'tokenizer_path', None),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(path / 'config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save weights
    if checkpoint:
        torch.save(model.state_dict(), path / f'model_checkpoint_{epoch}.pt')
    else:
        torch.save(model.state_dict(), path / 'model.pt')

def load_model(path: Union[str, Path], tokenizer_json: Union[str, Path], device='cuda') -> MesicapLM:
    """Load model from checkpoint."""
    path = Path(path)

    with open(path / 'config.json', 'r') as f:
        metadata = json.load(f)
        cfg = metadata.get('model_config', {})

    tokenizer = TokenizerAdapter(tokenizer_json)
    model = MesicapLM(cfg, tokenizer)
    model.load_state_dict(torch.load(path / 'model.pt', map_location=device))

    return model.to(device)

# ============= MAIN TRAINING LOOP =============

class DatasetBuilder:
    """Constructs DataLoaders backed by DatatroveFolderDataset or simple text fallback."""

    def __init__(self, device: str, tokenizer: TokenizerAdapter, seq_length: int, token_size: int = 2,
                 datatrove_paths_file: Optional[Union[str, Path]] = None):
        self.device = device
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.token_size = int(token_size)
        self.paths_file = Path(datatrove_paths_file) if datatrove_paths_file else None

    def _get_dataloader_kwargs(self, for_eval: bool = False):
        if self.device == 'cuda':
            workers = min(8, (os.cpu_count() or 8))
            return dict(num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, shuffle=not for_eval)
        if self.device == 'mps':
            return dict(num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None, shuffle=not for_eval)
        workers = min(4, (os.cpu_count() or 4))
        return dict(num_workers=workers, pin_memory=False, persistent_workers=True, prefetch_factor=2, shuffle=not for_eval)

    def _ensure_paths_file(self, folder: Union[str, Path]) -> Optional[str]:
        if self.paths_file is None:
            return None
        folder = Path(folder)
        self.paths_file.parent.mkdir(parents=True, exist_ok=True)
        return self.paths_file.as_posix()

    def _build_datatrove_dataset(self, folder: Union[str, Path], shuffle: bool, seed: int) -> DatatroveFolderDataset:
        folder_path = Path(folder).as_posix()
        paths_file = self._ensure_paths_file(folder)
        return DatatroveFolderDataset(
            folder_path=folder_path,
            seq_len=self.seq_length,
            filename_pattern="*.ds",
            recursive=True,
            token_size=self.token_size,
            shuffle=shuffle,
            seed=seed,
            return_positions=False,
            positions_from_eos_token_id=None,
            paths_file=paths_file,
        )

    def _datatrove_collate(self, batch: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        token_tensors = []
        for item in batch:
            input_ids = item['input_ids']
            if isinstance(input_ids, np.ndarray):
                input_ids = torch.from_numpy(input_ids.astype(np.int64))
            elif not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            else:
                input_ids = input_ids.to(dtype=torch.long)
            token_tensors.append(input_ids)

        stacked = torch.stack(token_tensors, dim=0)
        return stacked

    def create_dataloaders(self, train_folder: Union[str, Path], val_folder: Optional[Union[str, Path]] = None,
                           batch_size: int = 32, seed: int = 42) -> tuple[DataLoader, DataLoader]:
        train_ds = self._build_datatrove_dataset(train_folder, shuffle=True, seed=seed)
        val_folder_resolved = val_folder if val_folder is not None else train_folder
        val_ds = self._build_datatrove_dataset(val_folder_resolved, shuffle=False, seed=seed + 1)

        def build_loader(dataset, for_eval: bool):
            if len(dataset) == 0:
                raise ValueError(f"Datatrove dataset at {train_folder if not for_eval else val_folder_resolved} is empty")
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self._datatrove_collate,
                **self._get_dataloader_kwargs(for_eval=for_eval)
            )

        train_loader = build_loader(train_ds, for_eval=False)
        val_loader = build_loader(val_ds, for_eval=True)

        return train_loader, val_loader


class CurriculumManager:
    """Manages curriculum learning stages and their execution."""

    def __init__(self, dataset_builder: DatasetBuilder, seq_length: int, batch_size: int):
        self.dataset_builder = dataset_builder
        self.seq_length = seq_length
        self.batch_size = batch_size

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

            stage_roots = stage.get('train_folder') or stage.get('packed_roots') or stage.get('packed_dirs')
            if stage_roots is None:
                raise ValueError(f"Curriculum stage '{stage_name}' requires 'train_folder' or 'packed_roots'")
            if isinstance(stage_roots, (list, tuple)):
                train_root = stage_roots[0]
                val_root = stage.get('val_folder', stage.get('val_root', stage_roots[-1]))
            else:
                train_root = stage_roots
                val_root = stage.get('val_folder', stage.get('val_root', train_root))

            stage_train_loader, stage_val_loader = self.dataset_builder.create_dataloaders(
                train_folder=train_root,
                val_folder=val_root,
                batch_size=self.batch_size,
                seed=stage.get('seed', 42)
            )

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
        if stage_name != 'school':
            return

        n_freeze = int(len(self.model.trf_blocks) * 0.8)
        print(f"[StageRunner][debug] Freezing token embeddings and first {n_freeze} transformer blocks")

        for param in self.model.tok_emb.parameters():
            param.requires_grad = False

        for idx, block in enumerate(self.model.trf_blocks[:n_freeze]):
            for param in block.parameters():
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
                if global_step % self.eval_interval == 0:
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
                'd_model': self.model.cfg['d_model'],
                'n_heads': self.model.cfg['n_heads'],
                'n_layers': len(self.model.cfg['n_layers']),
                'max_len': self.model.cfg['context_length'],
                'rope_base': self.model.cfg['rope_base'],
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
    train_data_folder: Union[str, Path],
    val_data_folder: Optional[Union[str, Path]] = None,
    tokenizer_json: Union[str, Path] = "tokenizer.json",
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
    token_size: int = 2,
    datatrove_paths_file: Optional[Union[str, Path]] = None,
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

    tokenizer = TokenizerAdapter(tokenizer_json)

    MESICAP_CONFIG = {
        "batch_size": batch_size,
        "vocab_size": tokenizer.vocab_size,
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
        "moe_intermediate_size": d_model * 2,
        "use_gla_first_layer": True,
        "use_swa_layers": False,
        "swa_layers": [1, 2],
        "swa_window": 128,
    }

    # Create model once and reuse across stages
    model = MesicapLM(MESICAP_CONFIG, tokenizer)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Create helper classes
    dataset_builder = DatasetBuilder(device, tokenizer, seq_length, token_size, datatrove_paths_file)
    curriculum_manager = CurriculumManager(dataset_builder, seq_length, batch_size)
    stage_runner = StageRunner(
        model, device, eval_interval, ema_alpha, test_prompt, model_path, run
    )

    # Run training
    using_curriculum = bool(curriculum and len(curriculum) > 0)
    if using_curriculum:
        curriculum_manager.run_curriculum(curriculum, stage_runner)
    else:
        # Single-stage legacy flow
        train_loader, val_loader = dataset_builder.create_dataloaders(
            train_folder=train_data_folder,
            val_folder=val_data_folder,
            batch_size=batch_size,
            seed=42
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
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if load_and_test:
        tokenizer_json = Path("tokenizer.json")
        model = load_model("xor_model", tokenizer_json, device=device)
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
            eval_interval=1000,
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
