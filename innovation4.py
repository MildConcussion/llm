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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import json
import contextlib
from tqdm import tqdm
from grokadamw import GrokAdamW
from xor_packed import PackedXORShardDataset, MixedPackedXORDataset, discover_shards
from xor_packed import ensure_train_val_split, build_mixed_dataset
from xor_packed import build_capped_mixed_dataset
import os
import wandb

torch.set_float32_matmul_precision('high')


def partition_heads_golden(n_heads, n_groups):
    """
    Partition heads into groups using Golden ratio proportions.

    Args:
        n_heads: Number of heads to partition
        n_groups: Number of groups to create

    Returns:
        List of head index groups
    """
    if n_groups == 1:
        return [list(range(n_heads))]

    φ = (1 + np.sqrt(5)) / 2

    # Create golden-ratio weighted partitions (largest-remainder method)
    weights = np.array([φ ** (n_groups - 1 - i) for i in range(n_groups)], dtype=np.float64)
    weights = weights / weights.sum()

    raw = weights * n_heads
    floor_sizes = np.floor(raw).astype(int)
    remainder = raw - floor_sizes

    # Distribute remaining heads to groups with largest fractional parts
    remaining = int(n_heads - floor_sizes.sum())
    if remaining > 0:
        idx = np.argsort(-remainder)
        floor_sizes[idx[:remaining]] += 1

    sizes = floor_sizes

    # Create head index groups
    groups = []
    start = 0
    for size in sizes:
        if size > 0:  # Only add non-empty groups
            groups.append(list(range(start, start + size)))
            start += size

    return groups

# Even more concise version using vectorization
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

    k_min = int(np.ceil(n_heads / max_gs))
    k_max = int(max(1, n_heads // min_gs))
    if k_min > k_max:
        # Infeasible constraints; relax by increasing k to k_min and letting some groups be size > min_gs
        k = k_min
    else:
        k = k_min

    # Base golden weights for k groups (largest first)
    j = np.arange(k, dtype=np.float64)
    base_weights = φ ** (-j)
    base_weights = base_weights / base_weights.sum()

    # Compute a single canonical size vector within [min_gs, max_gs]
    base_sizes = np.full(k, min_gs, dtype=int)
    remaining = int(n_heads - base_sizes.sum())
    capacities = np.full(k, max_gs - min_gs, dtype=int)
    if remaining < 0:
        # Should not happen with k = ceil(n_heads / max_gs), but guard
        raise ValueError("Invalid group size constraints relative to n_heads")

    if remaining > 0:
        # Largest-remainder allocation within capacities
        raw_add = base_weights * remaining
        add_floor = np.floor(raw_add).astype(int)
        # Respect capacities
        add_floor = np.minimum(add_floor, capacities)
        sizes = base_sizes + add_floor
        leftover = remaining - add_floor.sum()
        if leftover > 0:
            rema = raw_add - add_floor
            # Assign remaining heads by descending fractional remainder, honoring capacity
            order = np.argsort(-rema)
            for idx in order:
                if leftover == 0:
                    break
                if sizes[idx] - base_sizes[idx] < capacities[idx]:
                    sizes[idx] += 1
                    leftover -= 1
        else:
            sizes = base_sizes
    else:
        sizes = base_sizes

    # Defensive clamp and final adjustment if rounding drifted
    sizes = np.clip(sizes, min_gs, max_gs)
    diff = int(n_heads - sizes.sum())
    if diff != 0:
        # Add/subtract heads starting from largest-remainder preference while respecting bounds
        direction = 1 if diff > 0 else -1
        steps = abs(diff)
        # Use remainders to guide distribution; if not available, use golden order
        rema = (base_weights * n_heads) - np.floor(base_weights * n_heads)
        order = np.argsort(-rema) if direction > 0 else np.argsort(rema)
        for _ in range(steps):
            for idx in order:
                if direction > 0 and sizes[idx] < max_gs:
                    sizes[idx] += 1
                    break
                if direction < 0 and sizes[idx] > min_gs:
                    sizes[idx] -= 1
                    break

    # Build per-layer groups by permuting size order using φ-phase
    groups_per_layer = []
    for i in range(n_layers):
        phase = (i * golden_conjugate) % 1.0
        positions = np.arange(len(sizes), dtype=np.float64)
        keys = np.mod(positions * golden_conjugate + phase, 1.0)
        pos_order = np.argsort(keys)  # ascending keys define placement order

        # Place sizes (largest first) into permuted positions
        sizes_sorted = np.sort(sizes)[::-1]
        sized_positions = np.empty_like(sizes)
        for rank, pos in enumerate(pos_order):
            sized_positions[pos] = sizes_sorted[rank]

        # Form contiguous groups anchored at head index 0
        head_ring = np.arange(n_heads, dtype=int)
        groups = []
        start = 0
        for sz in sized_positions:
            if sz > 0:
                groups.append(head_ring[start:start+sz].tolist())
                start += sz
        groups_per_layer.append(groups)

    return groups_per_layer


# ============= ENCODER =============

class GrayCodeEncoder:
    """Gray code preserves bit locality."""

    START, EOS, PAD, REGISTER = 256, 257, 258, 259
    # New special tokens for instruction message boundaries
    IM_START, IM_END = 260, 261

    def __init__(self):
        # Precompute Gray code lookup
        self.gray_lut = np.array([i ^ (i >> 1) for i in range(256)], dtype=np.int64)
        self.gray_inv = np.zeros(256, dtype=np.int64)
        for i in range(256):
            self.gray_inv[self.gray_lut[i]] = i

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.array([self.START, self.EOS], dtype=np.int64)

        text_bytes = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8)
        output = np.zeros(len(text_bytes) + 2, dtype=np.int64)

        output[0] = self.START
        output[1:-1] = self.gray_lut[text_bytes]
        output[-1] = self.EOS
        return output

    def decode(self, seq: np.ndarray) -> str:
        mask = (seq < 256)
        gray_bytes = seq[mask]
        original = self.gray_inv[gray_bytes]
        return bytes(original.astype(np.uint8)).decode('utf-8', errors='ignore')

    def __len__(self):
        # bytes (0..255) + START/EOS/PAD/REGISTER/IM_START/IM_END
        return len(self.gray_lut) + 6


# ============= DATASET =============

class XORDataset(Dataset):
    """Efficient dataset for XOR sequences."""

    def __init__(self, data_path: Union[str, Path], seq_length: int = 512,
                 stride: Optional[int] = None):
        self.encoder = GrayCodeEncoder()
        self.seq_length = seq_length
        self.stride = stride or seq_length // 2

        # Load and encode data
        if str(data_path).endswith(('.txt', '.md')):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # Encode entire text
        self.data = self.encoder.encode(text)

        # Calculate number of sequences
        self.n_sequences = max(1, (len(self.data) - self.seq_length) // self.stride + 1)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = min(start + self.seq_length, len(self.data))

        # Get sequence and pad if necessary
        seq = np.zeros(self.seq_length, dtype=np.int64)
        seq[:end-start] = self.data[start:end]

        # Fill rest with PAD
        if end - start < self.seq_length:
            seq[end-start:] = self.encoder.PAD

        return torch.from_numpy(seq)


# ============= MODEL =============

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Fixed version"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for maximum sequence length
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Create cos and sin embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        # Shape: [1, 1, seq_len, dim] for proper broadcasting with [B, H, L, head_dim]
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None, positions: torch.Tensor | None = None):
        """Apply rotary embeddings to queries and keys.
        Input shape: [B, H, L, head_dim]
        If positions is provided (Tensor[B, L]), compute cos/sin per batch using inv_freq.
        """
        if seq_len is None:
            seq_len = q.shape[2]  # Note: shape[2] because input is [B, H, L, head_dim]

        if positions is None:
            # Use precomputed values
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            # Compute per-batch cos/sin from provided positions
            # positions: [B, L] -> freqs: [B, L, dim/2]
            freqs = torch.einsum('bl,d->bld', positions.to(self.inv_freq.dtype), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, dim]
            cos = emb.cos().unsqueeze(1)  # [B, 1, L, dim]
            sin = emb.sin().unsqueeze(1)  # [B, 1, L, dim]

        # Apply rotation using complex number properties
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dtype):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class AsymGQATransformerBlock(nn.Module):
    """Transformer with Asymmetric Grouped-Query Attention"""

    def __init__(self, d_model, n_heads, d_ff, rope,
                 groups=None, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.param_dtype = dtype

        # Asymmetric grouping: list of lists [[0,1,2], [3], [4,5,6,7], ...]
        # If None, use standard MHA
        self.groups = groups or [[i] for i in range(n_heads)]
        self.n_kv_heads = len(self.groups)

        # Create mapping: which KV head does each Q head use?
        self.register_buffer('kv_map', self._create_kv_map())
        # Group aggregation matrix to map per-Q-head features to per-KV-head features
        # Shape: [n_kv_heads, n_heads]; rows average features over heads in the group
        group_agg = torch.zeros(self.n_kv_heads, self.n_heads, dtype=torch.float32)
        for kv_idx, group in enumerate(self.groups):
            if len(group) > 0:
                group_agg[kv_idx, group] = 1.0 / float(len(group))
        self.register_buffer('group_agg', group_agg)

        # Q always full size, K/V based on groups
        self.q_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)

        # FFN and norms unchanged
        self.ffn = FeedForward(d_model, d_ff, dtype)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Per-head additive tau parameters (token + position) inspired by screenshot design
        # Token maps: project head-local features to a scalar per token and head
        self.tau_wq = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        self.tau_wv_kv = nn.Parameter(torch.zeros(self.n_kv_heads, self.head_dim))
        # Positional alpha per head
        self.tau_alpha = nn.Parameter(torch.zeros(self.n_heads))

        nn.init.normal_(self.tau_wq, std=0.02)
        nn.init.normal_(self.tau_wv_kv, std=0.02)
        nn.init.zeros_(self.tau_alpha)

        # Small LRU cache for position logs per device and sequence length
        self._poslog_cache_lru = None
        self._poslog_cache_max = 64

    def _get_pos_log_cached(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return cached log1p(arange(L)) [L] float32 on device."""
        if self._poslog_cache_lru is None:
            from collections import OrderedDict
            self._poslog_cache_lru = OrderedDict()

        key = (str(device), int(seq_len))
        cache = self._poslog_cache_lru
        if key in cache:
            val = cache.pop(key)
            cache[key] = val
            if val.device != device:
                val = val.to(device)
                cache[key] = val
            return val
        # Miss → create
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        val = torch.log1p(pos)
        cache[key] = val
        if len(cache) > self._poslog_cache_max:
            cache.popitem(last=False)
        return val

    def _create_kv_map(self):
        """Create index mapping from Q heads to KV heads"""
        kv_map = torch.zeros(self.n_heads, dtype=torch.long)
        for kv_idx, group in enumerate(self.groups):
            for q_idx in group:
                kv_map[q_idx] = kv_idx
        return kv_map

    def forward(self, x, mask=None, key_padding_mask=None, positions: torch.Tensor | None = None):
        B, L, D = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # Project once
        q_proj = self.q_proj(x_norm)  # [B, L, D]
        k_proj = self.k_proj(x_norm)  # [B, L, n_kv_heads * head_dim]
        v_proj = self.v_proj(x_norm)  # [B, L, n_kv_heads * head_dim]

        # Reshape for attention
        q = q_proj.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K,V
        k = k[:, self.kv_map]
        v = v[:, self.kv_map]

        # Tau computation using already-projected values
        # Single GELU features from q_proj reused for both q and v tau computations
        tok_feat_q = F.gelu(q_proj).reshape(B, L, self.n_heads, self.head_dim)
        tau_tok_q = torch.tanh((tok_feat_q * self.tau_wq).sum(dim=-1))  # [B, L, H]
        # Map per-Q-head features to per-KV-head features via group aggregation
        # grouped_tok_feat: [B, L, n_kv_heads, head_dim]
        grouped_tok_feat = torch.einsum('gh,blhd->blgd', self.group_agg, tok_feat_q)
        tau_tok_v_grouped = torch.tanh((grouped_tok_feat * self.tau_wv_kv).sum(dim=-1))  # [B, L, n_kv_heads]
        tau_tok_v = tau_tok_v_grouped.gather(2, self.kv_map.view(1,1,-1).expand(B,L,-1))  # [B, L, H]

        # Position term (standardized dtype)
        if positions is None:
            pos_log_1d = self._get_pos_log_cached(L, x.device)  # [L]
            pos_log = pos_log_1d.view(1, L).expand(B, L)
        else:
            positions = positions.to(torch.float32)
            pos_log = torch.log1p(positions)
        alpha = torch.sigmoid(self.tau_alpha)
        tau_pos = 1.0 + alpha.view(1, self.n_heads, 1) * pos_log.view(B, 1, L) - 0.5

        # Final taus
        tau_q = (tau_tok_q.transpose(1, 2) + tau_pos).unsqueeze(-1)
        tau_v = (tau_tok_v.transpose(1, 2) + tau_pos).unsqueeze(-1)

        # Apply gating
        q = q * tau_q
        v = v * tau_v

        # Apply RoPE (optionally with custom positions)
        q, k = self.rope(q, k, seq_len=L, positions=positions)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Normalize mask to [B, 1, L, L]
            if mask.dim() == 2:  # [L, L]
                mask_norm = mask[None, None, :, :]
            elif mask.dim() == 3:  # [B, L, L]
                mask_norm = mask[:, None, :, :]
            elif mask.dim() == 4:  # [B, 1, L, L] or [B, H, L, L]
                # If heads dimension present, reduce/assume broadcastable
                mask_norm = mask if mask.size(1) == 1 else mask[:, :1, :, :]
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
            scores.masked_fill_(mask_norm, -float('inf'))
        if key_padding_mask is not None:
            # Mask both rows (queries from PAD) and columns (keys that are PAD)
            scores.masked_fill_(key_padding_mask[:, None, None, :], -float('inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)

        # Residual + FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))

        return x

    def full_pass_return_kv(self, x, mask=None, key_padding_mask=None, positions: torch.Tensor | None = None):
        """Full-sequence forward returning present K/V for caching.
        x: [B, L, D]
        Returns: x_out [B, L, D], k_present [B, H, L, head_dim], v_present [B, H, L, head_dim]
        """
        B, L, D = x.shape

        x_norm = self.norm1(x)

        q_proj = self.q_proj(x_norm)
        k_proj = self.k_proj(x_norm)
        v_proj = self.v_proj(x_norm)

        q = q_proj.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k[:, self.kv_map]
        v = v[:, self.kv_map]

        tok_feat_q = F.gelu(q_proj).reshape(B, L, self.n_heads, self.head_dim)
        tau_tok_q = torch.tanh((tok_feat_q * self.tau_wq).sum(dim=-1))
        grouped_tok_feat = torch.einsum('gh,blhd->blgd', self.group_agg, tok_feat_q)
        tau_tok_v_grouped = torch.tanh((grouped_tok_feat * self.tau_wv_kv).sum(dim=-1))
        tau_tok_v = tau_tok_v_grouped.gather(2, self.kv_map.view(1,1,-1).expand(B,L,-1))

        if positions is None:
            pos_log_1d = self._get_pos_log_cached(L, x.device)
            pos_log = pos_log_1d.view(1, L).expand(B, L)
        else:
            positions = positions.to(torch.float32)
            pos_log = torch.log1p(positions)
        alpha = torch.sigmoid(self.tau_alpha)
        tau_pos = 1.0 + alpha.view(1, self.n_heads, 1) * pos_log.view(B, 1, L) - 0.5

        tau_q = (tau_tok_q.transpose(1, 2) + tau_pos).unsqueeze(-1)
        tau_v = (tau_tok_v.transpose(1, 2) + tau_pos).unsqueeze(-1)

        q = q * tau_q
        v = v * tau_v

        q, k = self.rope(q, k, seq_len=L, positions=positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 2:
                mask_norm = mask[None, None, :, :]
            elif mask.dim() == 3:
                mask_norm = mask[:, None, :, :]
            elif mask.dim() == 4:
                mask_norm = mask if mask.size(1) == 1 else mask[:, :1, :, :]
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
            scores.masked_fill_(mask_norm, -float('inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)

        x_out = x + out
        x_out = x_out + self.ffn(self.norm2(x_out))

        # Present K/V are post-RoPE and post-mapping, per head
        k_present = k
        v_present = v

        return x_out, k_present, v_present

    def forward_incremental(self, x_last, positions_last: torch.Tensor, past_k: torch.Tensor | None, past_v: torch.Tensor | None):
        """Incremental step for the last token.
        x_last: [B, 1, D]; positions_last: [B, 1] float32
        past_k/v: [B, H, S, head_dim] or None
        Returns: x_last_out [B, 1, D], k_all [B, H, S+1, head_dim], v_all [B, H, S+1, head_dim]
        """
        B, T, D = x_last.shape
        assert T == 1, "forward_incremental expects a single-token sequence"

        x_norm = self.norm1(x_last)

        q_proj = self.q_proj(x_norm)
        k_proj = self.k_proj(x_norm)
        v_proj = self.v_proj(x_norm)

        q = q_proj.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_proj.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k[:, self.kv_map]
        v = v[:, self.kv_map]

        tok_feat_q = F.gelu(q_proj).reshape(B, T, self.n_heads, self.head_dim)
        tau_tok_q = torch.tanh((tok_feat_q * self.tau_wq).sum(dim=-1))
        grouped_tok_feat = torch.einsum('gh,blhd->blgd', self.group_agg, tok_feat_q)
        tau_tok_v_grouped = torch.tanh((grouped_tok_feat * self.tau_wv_kv).sum(dim=-1))
        tau_tok_v = tau_tok_v_grouped.gather(2, self.kv_map.view(1,1,-1).expand(B,T,-1))

        positions_last = positions_last.to(torch.float32)
        # For T==1 we don't need caching; compute directly
        pos_log = torch.log1p(positions_last)
        alpha = torch.sigmoid(self.tau_alpha)
        tau_pos = 1.0 + alpha.view(1, self.n_heads, 1) * pos_log.view(B, 1, T) - 0.5

        tau_q = (tau_tok_q.transpose(1, 2) + tau_pos).unsqueeze(-1)
        tau_v = (tau_tok_v.transpose(1, 2) + tau_pos).unsqueeze(-1)

        q = q * tau_q
        v = v * tau_v

        q, k = self.rope(q, k, seq_len=T, positions=positions_last)

        # Concatenate to cache
        if past_k is not None:
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all, v_all = k, v

        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_all)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.o_proj(out)

        x_out = x_last + out
        x_out = x_out + self.ffn(self.norm2(x_out))

        return x_out, k_all, v_all


class XOR8BitLM(nn.Module):
    """Fast XOR-based Language Model with optional MuToR."""

    def __init__(self, d_model=512, n_heads=8, n_layers=6,
                 max_len=2048, rope_base=10000,
                 mutor_dmax: int = 0, mutor_alpha: float = 0.3,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = GrayCodeEncoder()
        self.param_dtype = dtype
        # MuToR configuration
        self.mutor_dmax = int(mutor_dmax)
        self.mutor_alpha = float(mutor_alpha)

        # Precompute 8-bit lookup table [260, 8] for fast bit extraction
        lut_vals = torch.zeros(len(self.encoder), 8, dtype=torch.int64)
        byte_bits = (torch.arange(256, dtype=torch.int64).unsqueeze(-1) >> torch.arange(8, dtype=torch.int64)) & 1
        lut_vals[:256] = byte_bits
        # Special tokens bit patterns
        lut_vals[GrayCodeEncoder.START, 0] = 1
        lut_vals[GrayCodeEncoder.EOS, 1] = 1
        # PAD remains all zeros
        lut_vals[GrayCodeEncoder.REGISTER, 2] = 1  # REGISTER unique bit
        # Instruction message boundary tokens
        lut_vals[GrayCodeEncoder.IM_START, 3] = 1
        lut_vals[GrayCodeEncoder.IM_END, 4] = 1
        self.register_buffer('bit_lut', lut_vals.to(torch.float32))
        # Cache for causal masks by (device, seq_len)
        self._causal_masks: dict[tuple[str, int], torch.Tensor] = {}

        # Bit projection: 8 bits -> d_model
        self.bit_proj = nn.Linear(8, d_model, dtype=dtype)

        # Single learnable register embedding (additive bias)
        self.register_bias = nn.Parameter(torch.zeros(d_model))

        # Optional offset-aware register embeddings (index 0 reserved for non-register)
        if self.mutor_dmax > 0:
            self.offset_embeddings = nn.Embedding(self.mutor_dmax + 1, d_model, dtype=dtype)
            self._init_offset_embeddings()

        # RoPE for positional encoding
        if d_model % n_heads != 0 or ((d_model // n_heads) % 2 != 0):
            raise ValueError(
                f"Invalid head configuration: d_model={d_model}, n_heads={n_heads}. "
                f"Require d_model % n_heads == 0 and even head_dim for RoPE."
            )
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max_len, rope_base)

        groups_per_layer = golden_groups(n_layers, n_heads=n_heads)
        for i, g in enumerate(groups_per_layer):
            print(f"Layer {i}: {g} (KV groups: {len(g)})")

        self.layers = nn.ModuleList([
            AsymGQATransformerBlock(d_model, n_heads, d_model * 4, self.rope, groups, dtype=dtype)
            for groups in groups_per_layer
        ])

        self.norm = RMSNorm(d_model)

        # Output head (now 260 including REGISTER)
        self.out = nn.Linear(d_model, len(self.encoder), dtype=dtype)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_offset_embeddings(self):
        """Initialize offset embeddings with Gray code-inspired pattern; index 0 is zero."""
        with torch.no_grad():
            self.offset_embeddings.weight.zero_()
            # Indices 1..dmax get structured initialization
            for i in range(1, self.mutor_dmax + 1):
                gray = i ^ (i >> 1)
                bits = torch.tensor([(gray >> b) & 1 for b in range(8)], dtype=torch.float32, device=self.offset_embeddings.weight.device)
                pattern = bits.repeat(self.d_model // 8 + 1)[:self.d_model]
                self.offset_embeddings.weight[i].copy_(pattern * 0.02)

    @torch.jit.export
    def to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence to bit features (vectorized with LUT)."""
        # Direct LUT indexing for 0..259 (bytes + specials + REGISTER)
        x_clamped = x.clamp(min=0, max=self.bit_lut.shape[0] - 1)
        return self.bit_lut[x_clamped]

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return cached upper-triangular causal mask of shape [L, L] (bool) using small LRU."""
        # LRU implemented with OrderedDict semantics
        if not hasattr(self, '_causal_masks_lru'):
            from collections import OrderedDict
            self._causal_masks_lru = OrderedDict()
            self._causal_masks_max = 64

        key = (str(device), int(seq_len))
        cache = self._causal_masks_lru

        if key in cache:
            mask = cache.pop(key)
            # Refresh position to mark as most-recently-used
            cache[key] = mask
            # Move to device if needed (rare path)
            if mask.device != device:
                mask = mask.to(device)
                cache[key] = mask
            return mask

        # Miss → create
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
        cache[key] = mask
        # Evict least-recently-used if beyond capacity
        if len(cache) > self._causal_masks_max:
            cache.popitem(last=False)
        return mask

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                is_register: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional MuToR positions/registers support."""
        B, L = x.shape

        if (x == self.encoder.PAD).all():
            return torch.zeros(B, L, len(self.encoder), device=x.device, dtype=self.param_dtype)

        # Convert to bits and project
        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        # Add register bias where applicable
        if is_register is not None:
            h = h + self.register_bias.to(h.dtype) * is_register.unsqueeze(-1).to(h.dtype)

        # Build attention mask (normalize shapes in blocks)
        if is_register is None:
            mask = self._get_causal_mask(L, x.device)  # [L, L]
        else:
            base_mask = self._get_causal_mask(L, x.device)
            register_mask = is_register.unsqueeze(1).expand(-1, L, -1)  # [B, L, L]
            mask = base_mask.unsqueeze(0) | register_mask  # [B, L, L]

        # Key padding mask: True where token is PAD
        key_padding_mask = (x == self.encoder.PAD)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, mask, key_padding_mask, positions=positions)

        # Final norm and output
        h = self.norm(h)
        return self.out(h)

    def forward_mutor_sparse(self, x: torch.Tensor, register_offsets: torch.Tensor | None = None) -> torch.Tensor:
        """Single-stream sparse MuToR forward.
        - x: [B, L] token ids
        - register_offsets: [B, L] int64, 0 for non-register, 1..dmax for register positions
        """
        B, L = x.shape

        # Bit projection
        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        # Add offset embeddings where provided
        if register_offsets is not None and hasattr(self, 'offset_embeddings'):
            if register_offsets.dtype != torch.long:
                register_offsets = register_offsets.to(torch.long)
            h = h + self.offset_embeddings(register_offsets)

        # Standard causal mask
        mask = self._get_causal_mask(L, x.device)
        key_padding_mask = (x == self.encoder.PAD)

        # Transformer stack
        for layer in self.layers:
            h = layer(h, mask, key_padding_mask, positions=None)

        h = self.norm(h)
        return self.out(h)

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                           is_register: torch.Tensor | None = None,
                           past_kv: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None):
        """Forward that supports KV-cache for incremental generation.
        x: [B, T]
        positions: [B, T] float32 or None
        past_kv: list of (k, v) per layer or None; k/v shapes [B, H, S, head_dim]
        Returns: logits [B, T, V], new_past_kv
        """
        self.eval()
        B, T = x.shape

        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        if is_register is not None:
            h = h + self.register_bias.to(h.dtype) * is_register.unsqueeze(-1).to(h.dtype)

        # Mask: causal only for generation; MuToR masking not used in generation
        mask = None

        new_past: List[tuple[torch.Tensor, torch.Tensor]] = []
        use_incremental = past_kv is not None and T == 1

        if not use_incremental:
            # Full pass build and collect K/V
            for i, layer in enumerate(self.layers):
                h, k_present, v_present = layer.full_pass_return_kv(h, mask, None, positions=positions)
                new_past.append((k_present, v_present))
        else:
            # Incremental
            assert positions is not None, "positions must be provided for incremental generation"
            for i, layer in enumerate(self.layers):
                k_prev, v_prev = past_kv[i]
                h, k_all, v_all = layer.forward_incremental(h, positions_last=positions, past_k=k_prev, past_v=v_prev)
                new_past.append((k_all, v_all))

        h = self.norm(h)
        logits = self.out(h)
        return logits, new_past

    def _compute_entropy(self, probs):
        """Compute entropy of probability distribution."""
        valid = probs > 1e-10
        if not valid.any():
            return 0.0
        p = probs[valid]
        return -(p * torch.log(p)).sum().item()

    def _apply_top_h(self, sorted_logits, sorted_idx, alpha=0.4):
        """Apply Top-H filtering to already-sorted logits.
        Returns mask for tokens to keep."""

        # Get probabilities from sorted logits
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Compute full distribution entropy (for threshold)
        full_entropy = self._compute_entropy(sorted_probs)
        threshold = alpha * full_entropy

        # Build subset iteratively, tracking entropy
        keep_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
        keep_mask[..., 0] = True  # Always keep top token

        for i in range(1, min(100, sorted_probs.shape[-1])):  # Limit search to top-100 for speed
            if sorted_probs[..., i] < 1e-10:
                break
            keep_mask[..., i] = True
            # Compute entropy of current subset
            subset_probs = sorted_probs[keep_mask]
            subset_probs = subset_probs / subset_probs.sum()
            if self._compute_entropy(subset_probs) > threshold:
                keep_mask[..., i] = False  # Remove last token
                break

        return keep_mask

    @torch.no_grad()
    def generate_with_cache(self, prompt="", max_len=100, temp=1.0, sampling='top_p', top_p=0.95, alpha=0.4, debug=False):
        """Generation with KV-cache optimized for MPS/CPU. Supports top-p and top-h sampling."""
        self.eval()
        device = next(self.parameters()).device

        seq = self.encoder.encode(prompt)
        if len(seq) > 0 and seq[-1] == self.encoder.EOS:
            seq = seq[:-1]
        x = torch.from_numpy(seq).long().unsqueeze(0).to(device)

        past_kv = None
        # Running absolute positions for RoPE/tau
        cur_len = x.size(1)
        pos = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)

        # Prime cache with initial context
        logits, past_kv = self.forward_with_cache(x, positions=pos, is_register=None, past_kv=None)

        # Pre-allocate cache for maximum sequence length
        max_cache_len = min(cur_len + max_len, self.rope.max_seq_len)
        pre_allocated_kv = []
        for k, v in past_kv:
            # Allocate full-size tensors
            k_cache = torch.zeros(k.size(0), k.size(1), max_cache_len, k.size(3),
                                device=device, dtype=k.dtype)
            v_cache = torch.zeros(v.size(0), v.size(1), max_cache_len, v.size(3),
                                device=device, dtype=v.dtype)
            # Copy initial context
            k_cache[:, :, :cur_len] = k
            v_cache[:, :, :cur_len] = v
            pre_allocated_kv.append((k_cache, v_cache))

        for _ in range(max_len):
            # Stop if cache is full to avoid overflow writes
            if cur_len >= max_cache_len:
                if debug:
                    print(f"[gen-cache] reached max_cache_len={max_cache_len}; stopping to avoid cache overflow")
                break

            # Last token logits
            last_logits = logits[:, -1, :] / temp
            last_logits[..., self.encoder.START] = -float('inf')
            last_logits[..., self.encoder.PAD] = -float('inf')
            last_logits[..., self.encoder.REGISTER] = -float('inf')
            # Avoid sampling instruction boundary tokens during free generation
            if hasattr(self.encoder, 'IM_START'):
                last_logits[..., self.encoder.IM_START] = -float('inf')
            if hasattr(self.encoder, 'IM_END'):
                last_logits[..., self.encoder.IM_END] = -float('inf')

            if not torch.isfinite(last_logits).any():
                last_logits = torch.zeros_like(last_logits)
                last_logits[..., :256] = 1.0
                last_logits[..., self.encoder.EOS] = 1.0
                last_logits[..., self.encoder.START] = -float('inf')
                last_logits[..., self.encoder.PAD] = -float('inf')
                last_logits[..., self.encoder.REGISTER] = -float('inf')
                if debug:
                    print("[gen-cache] fallback logits")

            # Sort logits once
            sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
            sorted_logits = torch.where(
                torch.isfinite(sorted_logits), sorted_logits, torch.full_like(sorted_logits, -1e10)
            )

            # Apply sampling method
            if sampling == 'top_h':
                keep_mask = self._apply_top_h(sorted_logits, sorted_idx, alpha)
            else:  # top_p
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep_mask = cumsum <= top_p
                keep_mask[..., 0] = True

            # Apply mask and get final probabilities
            sorted_logits = torch.where(keep_mask, sorted_logits, torch.full_like(sorted_logits, -float('inf')))
            probs = F.softmax(sorted_logits, dim=-1)

            # Check for valid probabilities and sample
            if not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() == 0:
                probs = torch.zeros_like(last_logits)
                probs[..., :256] = 1.0 / 256
                probs[..., self.encoder.EOS] = 0.01
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
                if debug:
                    print("[gen-cache] fallback probs")
            else:
                next_token_sorted = torch.multinomial(probs, 1)
                next_token = sorted_idx.gather(-1, next_token_sorted)

            if next_token.item() == self.encoder.EOS:
                break

            # Update sequence and run incremental step
            x_next = next_token
            x = torch.cat([x, x_next], dim=1)
            cur_len += 1
            pos_next = torch.tensor([[cur_len - 1]], device=device, dtype=torch.float32)

            # Pass sliced cache and update in place
            current_kv = [(k[:, :, :cur_len-1], v[:, :, :cur_len-1]) for k, v in pre_allocated_kv]
            logits, new_kv = self.forward_with_cache(x_next, positions=pos_next, is_register=None, past_kv=current_kv)

            # Update the pre-allocated cache in place
            for i, (k_new, v_new) in enumerate(new_kv):
                pre_allocated_kv[i][0][:, :, :cur_len] = k_new
                pre_allocated_kv[i][1][:, :, :cur_len] = v_new

            if debug:
                last_tokens = x[0, -4:].tolist() if x.size(1) >= 4 else x[0].tolist()
                token_id = next_token.item()
                special = 'EOS' if token_id == self.encoder.EOS else (
                    'START' if token_id == self.encoder.START else (
                    'PAD' if token_id == self.encoder.PAD else ''))
                print(f"[gen-cache] last={last_tokens} -> next={token_id}{'('+special+')' if special else ''}")

        return self.encoder.decode(x[0].cpu().numpy())

    @torch.no_grad()
    def generate(self, prompt="", max_len=100, temp=1.0, top_p=0.9, debug=False):
        """Fast generation with top-p sampling - Fixed for numerical stability."""
        self.eval()
        device = next(self.parameters()).device

        # Encode prompt
        seq = self.encoder.encode(prompt)
        # Remove trailing EOS to allow continuation
        if len(seq) > 0 and seq[-1] == self.encoder.EOS:
            seq = seq[:-1]
        x = torch.from_numpy(seq).long().unsqueeze(0).to(device)

        for _ in range(max_len):
            # Crop context if it exceeds max length
            if x.size(1) > self.rope.max_seq_len:
                x = x[:, -self.rope.max_seq_len:]

            # Get logits
            logits = self(x)[:, -1, :] / temp

            # Prevent sampling of START, PAD, and REGISTER tokens
            logits[..., self.encoder.START] = -float('inf')
            logits[..., self.encoder.PAD] = -float('inf')
            logits[..., self.encoder.REGISTER] = -float('inf')
            # Avoid sampling instruction boundary tokens during free generation
            if hasattr(self.encoder, 'IM_START'):
                logits[..., self.encoder.IM_START] = -float('inf')
            if hasattr(self.encoder, 'IM_END'):
                logits[..., self.encoder.IM_END] = -float('inf')

            # Check if we have any valid logits
            if not torch.isfinite(logits).any():
                # Emergency fallback: allow all byte values
                logits = torch.zeros_like(logits)
                logits[..., :256] = 1.0  # Equal probability for all bytes
                logits[..., self.encoder.EOS] = 1.0  # Allow EOS
                logits[..., self.encoder.START] = -float('inf')
                logits[..., self.encoder.PAD] = -float('inf')
                logits[..., self.encoder.REGISTER] = -float('inf')
                if debug:
                    print("[gen] fallback logits -> uniform over bytes + EOS; masked START/PAD")

            # Top-p sampling with numerical stability
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)

            # Remove any remaining inf values
            sorted_logits = torch.where(
                torch.isfinite(sorted_logits),
                sorted_logits,
                torch.full_like(sorted_logits, -1e10)
            )

            sorted_probs = F.softmax(sorted_logits, dim=-1)

            # Top-p filtering (ensure at least 1 token kept)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumsum <= top_p
            # Always keep the highest-prob token
            keep_mask[..., 0] = True
            sorted_logits = torch.where(keep_mask, sorted_logits, torch.full_like(sorted_logits, -float('inf')))

            # Final probability distribution
            probs = F.softmax(sorted_logits, dim=-1)

            # Check for valid probabilities
            if not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() == 0:
                # Ultimate fallback: uniform distribution over bytes
                probs = torch.zeros_like(logits)
                probs[..., :256] = 1.0 / 256
                probs[..., self.encoder.EOS] = 0.01
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
                if debug:
                    print("[gen] fallback probs -> uniform bytes with small EOS mass")
            else:
                # Normal sampling
                next_token_sorted = torch.multinomial(probs, 1)
                next_token = sorted_idx.gather(-1, next_token_sorted)

            if debug:
                last_tokens = x[0, -4:].tolist() if x.size(1) >= 4 else x[0].tolist()
                token_id = next_token.item()
                special = 'EOS' if token_id == self.encoder.EOS else (
                    'START' if token_id == self.encoder.START else (
                    'PAD' if token_id == self.encoder.PAD else ''))
                print(f"[gen] last={last_tokens} -> next={token_id}{'('+special+')' if special else ''}")

            if next_token.item() == self.encoder.EOS:
                break

            x = torch.cat([x, next_token], dim=1)

        return self.encoder.decode(x[0].cpu().numpy())

# ============= TRAINING =============

class StreamingGrokMetrics:
    """Exponential moving average metrics for fast grokking signals."""

    def __init__(self, alpha=0.99):
        self.alpha = alpha  # EMA decay
        self.train_loss_ema = None
        self.eval_loss_ema = None
        self.perp_ema = None
        self.best_perp = float('inf')

    def update_train(self, loss: float):
        if self.train_loss_ema is None:
            self.train_loss_ema = loss
        else:
            self.train_loss_ema = self.alpha * self.train_loss_ema + (1 - self.alpha) * loss

    def update_eval(self, loss: float, perplexity: float):
        if self.eval_loss_ema is None:
            self.eval_loss_ema = loss
            self.perp_ema = perplexity
        else:
            self.eval_loss_ema = self.alpha * self.eval_loss_ema + (1 - self.alpha) * loss
            self.perp_ema = self.alpha * self.perp_ema + (1 - self.alpha) * perplexity

        self.best_perp = min(self.best_perp, perplexity)

    def get_signal(self) -> float:
        if self.train_loss_ema is None or self.eval_loss_ema is None:
            return 0.0

        # Smooth signals with EMA
        loss_gap = max(0, self.eval_loss_ema - self.train_loss_ema)
        loss_signal = loss_gap / max(self.eval_loss_ema, self.train_loss_ema, 1e-6)

        perp_signal = 0.0
        if self.best_perp < float('inf'):
            perp_signal = max(0, (self.perp_ema - self.best_perp) / self.best_perp)

        return 0.7 * loss_signal + 0.3 * perp_signal

class Trainer:
    """Efficient trainer with mixed precision and gradient accumulation."""

    def __init__(self, model: XOR8BitLM, lr=3e-4, warmup_steps=1000,
                 weight_decay=0.1, grad_accum_steps=1, device='cuda',
                 total_steps: int | None = None, ema_alpha=0.99,
                 mutor_mode: str = 'entropy',
                 mutor_density_ratio: float = 0.15,
                 mutor_min_spacing: int = 4,
                 mutor_update_every: int = 5,
                 mutor_buffer_momentum: float = 0.95,
                 mutor_bit_divergence_weight: float = 0.3):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.mutor_mode = mutor_mode

        print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

        print("\nModel architecture:")
        print(model)

        output = model.generate("Test", max_len=20)
        print(f"\nGenerated (untrained): {output}")

        self.metrics = StreamingGrokMetrics(alpha=ema_alpha)

        # Optimizer with weight decay on everything except biases and norm gains
        # Robustly exclude RMSNorm (scale/shift), LayerNorm weights, and any biases
        decay_names = set()
        no_decay_names = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
              continue  # frozen weights
            if (len(param.shape) == 1 or
                "bit_proj" in name or
                name.endswith('.bias') or
                '.norm' in name or
                name.endswith('.scale') or
                name.endswith('.shift') or
                'register_embedding' in name or  # Add this
                'offset_embeddings' in name):     # Add this
                no_decay_names.add(name)
            else:
                decay_names.add(name)

        # print(f"\ndecay_names: {decay_names}\n")
        # print(f"no_decay_names: {no_decay_names}\n")

        param_groups = [
            {'params': [p for n, p in model.named_parameters() if n in decay_names],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if n in no_decay_names],
             'weight_decay': 0.0}
        ]

        self.train_loss = None
        self.eval_loss = None
        self.perplexity = float('inf')
        self.best_perplexity = float('inf')

        self.opt = GrokAdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            grokking_signal_fns=[lambda: self.metrics.get_signal()]
        )

        # OneCycle schedule with proper total steps and warmup fraction
        if total_steps is None:
            total_steps = max(warmup_steps * 20, 1000)
        pct_start = min(max(warmup_steps / total_steps, 1e-6), 0.9)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy='cos'
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

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

        # Entropy MuToR controller (single-stream sparse)
        self.mutor_controller: EntropyMuToRController | None = None
        if self.mutor_mode == 'entropy' and getattr(self.model, 'mutor_dmax', 0) > 0:
            self.mutor_controller = EntropyMuToRController(
                max_seq_len=getattr(self.model.rope, 'max_seq_len', 2048),
                dmax=getattr(self.model, 'mutor_dmax', 3),
                density_ratio=mutor_density_ratio,
                min_spacing=mutor_min_spacing,
                update_every=mutor_update_every,
                momentum=mutor_buffer_momentum,
                bit_divergence_weight=mutor_bit_divergence_weight,
                device=self.device
            )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: int = None, use_mutor: bool = True) -> tuple[float, float]:
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
            batch = batch.to(self.device)
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

    def train_step(self, batch: torch.Tensor, use_mutor: bool = True) -> float:
        """Single training step with mixed precision and optional MuToR."""
        # Support optional (tokens, loss_mask) from packed dataset v2
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            tokens, loss_mask = batch
            batch = tokens.to(self.device)
            loss_mask = loss_mask.to(self.device)
        else:
            batch = batch.to(self.device)
            loss_mask = None

        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with self.autocast_ctx:
            if use_mutor and getattr(self.model, 'mutor_dmax', 0) > 0 and self.mutor_mode == 'dense':
                # Legacy dense MuToR path
                aug_batch, positions, is_register, aug_targets = mutor_augment_batch(inputs, targets, self.model, self.device)
                logits = self.model(aug_batch, positions=positions, is_register=is_register)
                loss = compute_mutor_loss(logits, aug_targets, is_register, self.encoder.PAD, alpha=self.model.mutor_alpha)
            elif use_mutor and getattr(self.model, 'mutor_dmax', 0) > 0 and self.mutor_mode == 'entropy' and self.mutor_controller is not None:
                # Entropy-guided single-stream sparse MuToR
                B, L = inputs.shape
                # Early features for difficulty
                bits = self.model.to_bits(inputs)  # [B, L, 8]
                h0 = self.model.bit_proj(bits)     # [B, L, D]

                # Compute difficulty and update EMA buffer
                layer0 = self.model.layers[0]
                difficulty = self.mutor_controller.compute_difficulty(h0, bits, layer0)  # [B, L]
                self.mutor_controller.update_buffer(difficulty)

                # Select registers using buffer for stability
                buffer_view = self.mutor_controller.difficulty_buffer[:L].unsqueeze(0).expand(B, -1)
                register_mask = self.mutor_controller.select_registers(buffer_view)  # [B, L] bool

                # Adaptive offsets from current difficulty
                register_offsets = self.mutor_controller.compute_offsets(difficulty, register_mask)  # [B, L] long

                # Valid lookahead positions: i + offset < batch.shape[1]
                idx = torch.arange(L, device=self.device).view(1, -1)
                target_pos = idx + register_offsets
                batch_len = batch.shape[1]
                reg_valid = register_mask & (target_pos < batch_len)

                # Forward once with offsets
                logits = self.model.forward_mutor_sparse(inputs, register_offsets=register_offsets)

                # Build combined targets and weights
                V = logits.size(-1)
                pad_id = self.encoder.PAD
                combined_targets = torch.full((B, L), pad_id, device=self.device, dtype=torch.long)

                # Non-register positions use next-token
                non_reg_mask = ~register_mask
                combined_targets[non_reg_mask] = targets[non_reg_mask]

                # Register positions use lookahead (where valid)
                clamped_pos = target_pos.clamp(max=batch_len - 1)
                lookahead_targets = batch.gather(dim=1, index=clamped_pos)
                combined_targets[reg_valid] = lookahead_targets[reg_valid]

                # Position weights: alpha at valid registers, (1-alpha) elsewhere
                weights = torch.full((B, L), 1.0 - self.model.mutor_alpha, device=self.device, dtype=logits.dtype)
                weights[reg_valid] = self.model.mutor_alpha

                # Compute weighted CE loss over valid positions, with optional dataset loss mask
                per_pos_loss = F.cross_entropy(
                    logits.reshape(B * L, V),
                    combined_targets.reshape(B * L),
                    ignore_index=pad_id,
                    reduction='none'
                ).view(B, L)

                base_valid = combined_targets != pad_id
                if loss_mask is not None:
                    # Build target-aligned mask per position
                    lm = (loss_mask > 0.5)
                    idx = torch.arange(L, device=self.device).view(1, -1)
                    idx_b = idx.expand(B, -1)
                    non_reg_target_idx = (idx_b + 1).clamp(max=L - 1)
                    gathered_nonreg = lm.gather(1, non_reg_target_idx)
                    gathered_reg = lm.gather(1, target_pos.clamp(max=batch_len - 1))
                    target_mask = torch.zeros(B, L, dtype=torch.bool, device=self.device)
                    target_mask[non_reg_mask] = gathered_nonreg[non_reg_mask]
                    target_mask[reg_valid] = gathered_reg[reg_valid]
                    valid_mask = base_valid & target_mask
                else:
                    valid_mask = base_valid

                if valid_mask.any():
                    loss = (per_pos_loss[valid_mask] * weights[valid_mask]).mean()
                else:
                    # Fallback: no valid positions, use standard next-token loss
                    loss = F.cross_entropy(
                        logits.reshape(-1, V),
                        targets.reshape(-1),
                        ignore_index=pad_id
                    )

                if os.environ.get('MUTOR_DEBUG', '0') == '1':
                    num_regs = register_mask.sum().item()
                    print(f"[mutor-sparse] regs={num_regs} ({num_regs/(B*L+1e-6):.3f}), offsets∈[1..{self.model.mutor_dmax}], valid={reg_valid.sum().item()}")
            else:
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
        if (self.step + 1) % self.grad_accum_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.opt)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

        self.step += 1
        return actual_loss

    @torch.no_grad()
    def eval_step(self, batch: torch.Tensor) -> tuple[float, int]:
        """Single evaluation step - returns loss and valid token count."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            tokens, loss_mask = batch
            batch = tokens.to(self.device)
            loss_mask = loss_mask.to(self.device)
        else:
            batch = batch.to(self.device)
            loss_mask = None
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
        """Fast single-batch evaluation for streaming metrics."""
        self.model.eval()
        loss, n_tokens = self.eval_step(val_batch)
        if n_tokens > 0:
            perplexity = math.exp(min(loss, 20))
            self.metrics.update_eval(loss, perplexity)
        self.model.train()

    @property
    def encoder(self):
        return self.model.encoder

# ============= UTILS =============

class EntropyMuToRController:
    """Controller for single-stream sparse MuToR selection and offsets.

    Maintains a per-position EMA difficulty buffer to amortize selection cost and stabilize choices.
    """

    def __init__(self,
                 max_seq_len: int,
                 dmax: int = 3,
                 density_ratio: float = 0.15,
                 min_spacing: int = 4,
                 update_every: int = 5,
                 momentum: float = 0.95,
                 bit_divergence_weight: float = 0.3,
                 device: torch.device | str = 'cpu'):
        self.dmax = int(dmax)
        self.density_ratio = float(density_ratio)
        self.min_spacing = int(min_spacing)
        self.update_every = int(update_every)
        self.momentum = float(momentum)
        self.bit_div_w = float(bit_divergence_weight)
        self.max_seq_len = int(max_seq_len)
        self.device = torch.device(device)

        self.difficulty_buffer = torch.zeros(self.max_seq_len, device=self.device, dtype=torch.float32)
        self.update_counts = torch.zeros(self.max_seq_len, device=self.device, dtype=torch.int32)
        self._step = 0

    @torch.no_grad()
    def compute_difficulty(self, h0: torch.Tensor, bits: torch.Tensor, layer0: nn.Module) -> torch.Tensor:
        """Compute multi-signal difficulty without logits.
        h0: [B, L, D] projected bit features
        bits: [B, L, 8] float32 0/1
        Returns difficulty in [0,1]: [B, L]
        """
        B, L, D = h0.shape

        # Normalize hidden states and compute entropy proxy (std/mean)
        h_norm = layer0.norm1(h0)
        mean = h_norm.mean(dim=-1)
        std = h_norm.std(dim=-1)
        h_entropy = std / (mean.abs() + 1e-6)
        h_entropy = torch.tanh(h_entropy)  # squash to ~[0,1]

        # Bit divergence via Hamming distance between neighbors
        flips = (bits[:, 1:, :] != bits[:, :-1, :]).float().mean(dim=-1)
        bit_div = F.pad(flips, (0, 1), value=0.0)

        # Attention uncertainty proxy from q/k projections (variance across heads)
        q = layer0.q_proj(h_norm).view(B, L, layer0.n_heads, layer0.head_dim)
        k = layer0.k_proj(h_norm).view(B, L, layer0.n_kv_heads, layer0.head_dim)
        # Map k to per-q head via kv_map to match head counts for the proxy (approximate)
        k_heads = k[:, :, layer0.kv_map, :]
        qk = (q * k_heads).sum(dim=-1)  # [B, L, H]
        qk_var = qk.var(dim=-1)
        attn_unc = torch.sigmoid(qk_var)

        # Weighted combination
        difficulty = (0.4 * h_entropy) + (self.bit_div_w * bit_div) + (0.3 * attn_unc)
        difficulty = difficulty.clamp(0.0, 1.0)
        return difficulty

    @torch.no_grad()
    def update_buffer(self, difficulty: torch.Tensor):
        """EMA update for per-position difficulty buffer using batch mean."""
        B, L = difficulty.shape
        if (self._step % self.update_every) == 0:
            batch_mean = difficulty.mean(dim=0)
            prev = self.difficulty_buffer[:L]
            self.difficulty_buffer[:L] = self.momentum * prev + (1.0 - self.momentum) * batch_mean
            self.update_counts[:L] += 1
        self._step += 1

    @torch.no_grad()
    def select_registers(self, difficulty: torch.Tensor) -> torch.Tensor:
        """Select register positions per batch with min spacing. Returns bool mask [B, L].

        Vectorized NMS-like selection using 1D max-pooling to enforce min_spacing.
        """
        B, L = difficulty.shape
        k = max(1, int(self.density_ratio * L))

        # Add tiny position-dependent bias to break ties deterministically
        idx = torch.arange(L, device=difficulty.device, dtype=difficulty.dtype)
        eps = (idx / max(L - 1, 1)).unsqueeze(0) * 1e-6
        biased = difficulty + eps

        # Max-pool to identify local maxima with separation >= min_spacing
        if self.min_spacing <= 1:
            pooled = biased
        else:
            kernel = 2 * self.min_spacing - 1
            pad = self.min_spacing - 1
            pooled = F.max_pool1d(biased.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad).squeeze(1)

        candidates = biased == pooled  # [B, L] bool, at most one peak per window after tie-break

        # Select top-k among candidates
        masked_scores = torch.where(candidates, difficulty, torch.full_like(difficulty, -float('inf')))
        k_eff = min(k, L)
        topk_vals, topk_idx = torch.topk(masked_scores, k=k_eff, dim=1, largest=True, sorted=False)
        valid = torch.isfinite(topk_vals)

        mask = torch.zeros(B, L, dtype=torch.bool, device=difficulty.device)
        if valid.any():
            rows = torch.arange(B, device=difficulty.device).unsqueeze(1).expand_as(topk_idx)
            mask[rows[valid], topk_idx[valid]] = True

        return mask

    @torch.no_grad()
    def compute_offsets(self, difficulty: torch.Tensor, register_mask: torch.Tensor) -> torch.Tensor:
        """Map difficulty to offsets in [1..dmax] at register positions, else 0."""
        B, L = difficulty.shape
        scaled = (difficulty * (self.dmax - 1)).floor().to(torch.long) + 1
        scaled = scaled.clamp_(1, self.dmax)
        offsets = torch.zeros(B, L, dtype=torch.long, device=difficulty.device)
        offsets[register_mask] = scaled[register_mask]
        return offsets

def mutor_augment_batch(inputs: torch.Tensor, targets: torch.Tensor, model: XOR8BitLM, device: torch.device):
    """Augment batch with interleaved REGISTER tokens and build positions/targets.
    inputs: [B, L], targets: [B, L] (next-token targets for inputs)
    Returns: aug_batch [B, L2], positions [B, L2], is_register [B, L2] (bool), aug_targets [B, L2]
    """
    B, L = inputs.shape
    L2 = 2 * L - 1

    # Sample offset d in [1, dmax]
    dmax = max(1, getattr(model, 'mutor_dmax', 1))
    d = int(torch.randint(1, dmax + 1, (1,), device=device).item())

    aug_batch = torch.full((B, L2), model.encoder.PAD, device=device, dtype=torch.long)
    aug_batch[:, 0::2] = inputs
    aug_batch[:, 1::2] = int(model.encoder.REGISTER)

    # Positions for RoPE
    # Keep integer math for indices, cast to float32 for model consumption
    positions = torch.zeros((B, L2), device=device, dtype=torch.float32)
    # Even (original tokens): 0..L-1
    base_pos = torch.arange(L, device=device, dtype=torch.long)
    positions[:, 0::2] = base_pos.to(torch.float32)

    # Odd indices (registers): L-1 positions
    # Register i (at position 2i+1) corresponds to token i (at position 2i)
    # So we need positions for tokens 0..L-2 (which have registers after them)
    reg_base_pos = base_pos[:-1]  # positions 0..L-2 (L-1 elements)
    reg_pos = torch.clamp(reg_base_pos + (d - 1), max=L - 1)
    positions[:, 1::2] = reg_pos.to(torch.float32)

    is_register = (aug_batch == int(model.encoder.REGISTER))

    # Build augmented targets
    aug_targets = torch.full_like(aug_batch, model.encoder.PAD)
    # Next-token targets at even indices
    aug_targets[:, 0::2] = targets
    # Register targets: for register i (after token i), predict token at i+d
    # Register i is at odd position 2i+1, corresponds to token i
    reg_target_idx = torch.clamp(reg_base_pos + d, max=L - 1)  # L-1 elements
    reg_targets = inputs.gather(1, reg_target_idx.unsqueeze(0).expand(B, -1))
    # Mask out PAD tokens - don't predict PAD as register output
    pad_mask = (inputs[:, :-1] == model.encoder.PAD)  # Which source positions are PAD
    reg_targets = torch.where(pad_mask, model.encoder.PAD, reg_targets)
    aug_targets[:, 1::2] = reg_targets

    if os.environ.get('MUTOR_DEBUG', '0') == '1':
        print(f"[mutor] d={d}, L={L}, L2={L2}")

    return aug_batch, positions, is_register, aug_targets


def compute_mutor_loss(logits, aug_targets, is_register, pad_id, alpha=0.3):
    """Corrected MuToR loss computation."""
    B, L2, V = logits.shape
    logits_f = logits.reshape(B * L2, V)
    targets_f = aug_targets.reshape(B * L2)
    is_reg_f = is_register.reshape(B * L2)

    valid = targets_f != pad_id

    if not valid.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Compute per-element losses
    losses = F.cross_entropy(logits_f[valid], targets_f[valid], reduction='none')

    # Apply weights
    weights = torch.where(is_reg_f[valid], alpha, 1.0 - alpha)
    weighted_loss = (losses * weights).mean()

    return weighted_loss

def save_model(model: XOR8BitLM, path: Union[str, Path], config: Dict[str, Any] = None, checkpoint: bool = False, epoch: int = 0):
    """Save model and config - updated for RoPE model."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # Save config
    allowed_keys = {'d_model', 'n_heads', 'n_layers', 'max_len', 'rope_base', 'mutor_dmax', 'mutor_alpha'}
    default_config = {
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': len(model.layers),
        'max_len': model.rope.max_seq_len,
        'rope_base': model.rope.base,
        'mutor_dmax': getattr(model, 'mutor_dmax', 0),
        'mutor_alpha': getattr(model, 'mutor_alpha', 0.3),
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

def load_model(path: Union[str, Path], device='cuda') -> XOR8BitLM:
    """Load model from checkpoint."""
    path = Path(path)

    # Load config
    with open(path / 'config.json', 'r') as f:
        loaded = json.load(f)
        # Be robust to extra metadata keys in older checkpoints
        allowed_keys = {'d_model', 'n_heads', 'n_layers', 'max_len', 'rope_base', 'mutor_dmax', 'mutor_alpha'}
        config = {k: v for k, v in loaded.items() if k in allowed_keys}

    # Create model
    model = XOR8BitLM(**config)

    # Load weights
    model.load_state_dict(torch.load(path / 'model.pt', map_location=device))

    return model.to(device)

# ============= MAIN TRAINING LOOP =============

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
    # MuToR new flags
    mutor_mode: str = 'entropy',  # 'entropy' | 'dense' | 'off'
    mutor_density_ratio: float = 0.15,
    mutor_min_spacing: int = 4,
    mutor_update_every: int = 5,
    mutor_buffer_momentum: float = 0.95,
    mutor_bit_divergence_weight: float = 0.3,
    # Curriculum settings (optional): list of stages [{'name': 'pretrain', 'packed_dirs': [...], 'epochs': 1, 'lr': 3e-4}, ...]
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
            "architecture": "XOR8BitLM",
            "epochs": epochs,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "rope_base": rope_base,
            "mutor_mode": mutor_mode,
            "mutor_density_ratio": mutor_density_ratio,
            "mutor_min_spacing": mutor_min_spacing,
            "mutor_update_every": mutor_update_every,
            "mutor_buffer_momentum": mutor_buffer_momentum,
            "mutor_bit_divergence_weight": mutor_bit_divergence_weight,
        },
    )

    print(f"Training on {device}")

    # Helper to build device-tuned DataLoader args
    def _loader_kwargs(for_eval: bool = False):
        if device == 'cuda':
            workers = min(8, (os.cpu_count() or 8))
            return dict(num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, shuffle=not for_eval)
        elif device == 'mps':
            return dict(num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None, shuffle=not for_eval)
        else:
            workers = min(4, (os.cpu_count() or 4))
            return dict(num_workers=workers, pin_memory=False, persistent_workers=True, prefetch_factor=2, shuffle=not for_eval)

    # Create datasets and loaders (packed preferred)
    def _build_packed_dataset(roots: List[Union[str, Path]]):
        # For each root, discover shards and concatenate them into one dataset
        ds_per_root = []
        for r in roots:
            shard_dirs = discover_shards(str(r))
            if not shard_dirs:
                continue
            shard_datasets = [
                PackedXORShardDataset(sd, seq_length=seq_length, stride=seq_length//2, pad_id=GrayCodeEncoder.PAD)
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

        return MixedPackedXORDataset(ds_per_root, policy=mixing_policy, weights=mixing_weights)

    using_curriculum = bool(curriculum and len(curriculum) > 0)

    if not using_curriculum:
        use_packed = False
        train_dataset = None
        val_dataset = None

        if packed_dirs is not None and len(packed_dirs) > 0:
            train_dataset = _build_packed_dataset([str(p) for p in packed_dirs])
            use_packed = True
            print(f"Using packed datasets for training. policy={mixing_policy}, weights={mixing_weights}")
        else:
            # Auto-detect packed shards under data_path directory
            dp = Path(data_path)
            if dp.is_dir():
                shard_dirs = discover_shards(str(dp))
                if shard_dirs:
                    train_dataset = _build_packed_dataset([dp])
                    use_packed = True
                    print(f"Auto-detected packed shards under {dp}")

        if train_dataset is None:
            # Fallback to simple text dataset
            train_dataset = XORDataset(data_path, seq_length)

        train_loader = DataLoader(train_dataset, batch_size, **_loader_kwargs(for_eval=False))

        # Validation dataset
        if val_packed_dirs is not None and len(val_packed_dirs) > 0:
            val_dataset = _build_packed_dataset([str(p) for p in val_packed_dirs])
        elif val_path is not None:
            vp = Path(val_path)
            if vp.is_dir() and discover_shards(str(vp)):
                val_dataset = _build_packed_dataset([vp])
            else:
                val_dataset = XORDataset(val_path, seq_length)
        else:
            if use_packed:
                # Heuristic: reuse train roots for eval with full-stride windows (no shuffle in loader)
                if packed_dirs is not None and len(packed_dirs) > 0:
                    val_dataset = _build_packed_dataset([str(p) for p in packed_dirs])
                else:
                    val_dataset = _build_packed_dataset([dp])
            else:
                # Use 10% of training data with different stride for pseudo-validation
                val_dataset = XORDataset(data_path, seq_length, stride=seq_length)

        val_loader = DataLoader(val_dataset, batch_size, **_loader_kwargs(for_eval=True))

    # Create model once and reuse across stages
    model = XOR8BitLM(d_model, n_heads, n_layers, seq_length, rope_base, mutor_dmax=3)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    def _run_stage(stage_name: str, stage_train_loader, stage_val_loader, stage_epochs: int, stage_lr: float, stage_steps: Optional[int] = None):
        steps_per_epoch = max(len(stage_train_loader), 1)
        planned_total = steps_per_epoch * stage_epochs
        effective_total = planned_total if stage_steps is None else min(planned_total, int(stage_steps))
        print(f"[stage] {stage_name}: steps_per_epoch={steps_per_epoch}, planned_total={planned_total}, effective_total={effective_total}")


        if stage_name == 'school':
            # Freeze lower layers and bit projection
            n_freeze = int(len(model.layers) * 0.6)  # Freeze 60% of layers

            # Freeze bit projection
            model.bit_proj.weight.requires_grad = False

            # Freeze RoPE (positional encoding)
            for param in model.rope.parameters():
                param.requires_grad = False

            # Freeze lower transformer layers
            for i in range(n_freeze):
                for param in model.layers[i].parameters():
                    param.requires_grad = False

        trainer = Trainer(
            model,
            lr=stage_lr,
            device=device,
            total_steps=effective_total,
            warmup_steps=min(1000, effective_total // 10),
            ema_alpha=ema_alpha,
            mutor_mode=mutor_mode,
            mutor_density_ratio=mutor_density_ratio,
            mutor_min_spacing=mutor_min_spacing,
            mutor_update_every=mutor_update_every,
            mutor_buffer_momentum=mutor_buffer_momentum,
            mutor_bit_divergence_weight=mutor_bit_divergence_weight,
        )

        val_iter = itertools.cycle(stage_val_loader)
        global_step = 0
        reached_cap = False
        for epoch in range(stage_epochs):
            model.train()
            pbar = tqdm(stage_train_loader, desc=f"{stage_name} Epoch {epoch+1}/{stage_epochs}")
            for batch_idx, train_batch in enumerate(pbar):
                loss = trainer.train_step(train_batch)
                global_step += 1
                if global_step % eval_interval == 0:
                    val_batch = next(val_iter)
                    trainer.quick_eval_update(val_batch)
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'train_ema': f"{trainer.metrics.train_loss_ema:.4f}" if trainer.metrics.train_loss_ema else "N/A",
                    'eval_ema': f"{trainer.metrics.eval_loss_ema:.4f}" if trainer.metrics.eval_loss_ema else "N/A",
                    'perp': f"{trainer.metrics.perp_ema:.2f}" if trainer.metrics.perp_ema else "N/A",
                    'grok': f"{trainer.metrics.get_signal():.3f}"
                })
                run.log({
                    'train/loss': loss,
                    'train/ema': trainer.metrics.train_loss_ema,
                    'validation/ema': trainer.metrics.eval_loss_ema,
                    'perplexity/ema': trainer.metrics.perp_ema,
                    'grok': trainer.metrics.get_signal()
                })
                if stage_steps is not None and global_step >= int(stage_steps):
                    print(f"[stage] {stage_name}: reached step cap ({global_step}/{int(stage_steps)}); ending stage after summary.")
                    reached_cap = True
                    break

            # Quick full eval snapshot
            model.eval()
            eval_losses = []
            for i, batch in enumerate(itertools.islice(stage_val_loader, 100)):
                loss, n_tokens = trainer.eval_step(batch)
                if n_tokens > 0:
                    eval_losses.append(loss)
            if eval_losses:
                avg_eval_loss = np.mean(eval_losses)
                epoch_perplexity = math.exp(min(avg_eval_loss, 20))
                print(f"\n{stage_name} Epoch {epoch+1} Summary:")
                print(f"  Streaming - Train EMA: {trainer.metrics.train_loss_ema:.4f}, "
                      f"Eval EMA: {trainer.metrics.eval_loss_ema:.4f}, "
                      f"Perp EMA: {trainer.metrics.perp_ema:.2f}")
                print(f"  Full Eval - Loss: {avg_eval_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
                print(f"  Grokking Signal: {trainer.metrics.get_signal():.3f}")
                run.log({
                    'validation/loss': avg_eval_loss,
                    'validation/perplexity': epoch_perplexity,
                    'validation/grok': trainer.metrics.get_signal()
                })

            # Save checkpoint per epoch
            save_model(model, model_path, {
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'n_layers': len(model.layers),
                'max_len': model.rope.max_seq_len,
                'rope_base': model.rope.base,
                'best_perplexity': trainer.metrics.best_perp,
                'stage': stage_name,
                'epoch': epoch + 1
            }, checkpoint=True, epoch=epoch + 1)

            model.eval()
            sample = model.generate(test_prompt, max_len=60)
            print(f"\nSample ({stage_name}): {sample}\n")
            if reached_cap:
                break

    if using_curriculum:
        # Run staged training with automatic split and mixing per dataset root
        for stage in curriculum:
            stage_name = stage.get('name', 'stage')
            stage_epochs = int(stage.get('epochs', 1))
            stage_lr = float(stage.get('lr', lr))
            stage_steps = stage.get('steps', None)
            if stage_steps is not None:
                stage_steps = int(stage_steps)
                if stage_steps <= 0:
                    raise ValueError(f"Curriculum stage '{stage_name}' has invalid steps={stage_steps}; must be > 0")
                print(f"[stage] {stage_name}: limiting to {stage_steps} steps")
            # Accept either 'packed_roots' (preferred) or legacy 'packed_dirs'
            roots = stage.get('packed_roots') or stage.get('packed_dirs') or []
            roots = [str(p) for p in roots]
            if not roots:
                raise ValueError(f"Curriculum stage '{stage_name}' requires 'packed_roots' or 'packed_dirs'")

            # Ensure train/val split exists (create if missing) for each dataset root
            val_ratio = float(stage.get('val_ratio', 0.05))
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
                    roots, split='train', seq_length=seq_length, pad_id=GrayCodeEncoder.PAD,
                    windows_needed=train_windows_needed, policy=mixing_policy, weights=mixing_weights
                )
                stage_val_ds = build_capped_mixed_dataset(
                    roots, split='val', seq_length=seq_length, pad_id=GrayCodeEncoder.PAD,
                    windows_needed=val_windows_needed, policy=mixing_policy, weights=mixing_weights
                )
            else:
                stage_train_ds = build_mixed_dataset(roots, split='train', seq_length=seq_length, pad_id=GrayCodeEncoder.PAD,
                                                     policy=mixing_policy, weights=mixing_weights)
                stage_val_ds = build_mixed_dataset(roots, split='val', seq_length=seq_length, pad_id=GrayCodeEncoder.PAD,
                                                   policy=mixing_policy, weights=mixing_weights)

            # Debug counts for visibility
            print(f"[stage] {stage_name}: roots={len(roots)}; building DataLoaders")

            stage_train_loader = DataLoader(stage_train_ds, batch_size, **_loader_kwargs(for_eval=False))
            stage_val_loader = DataLoader(stage_val_ds, batch_size, **_loader_kwargs(for_eval=True))
            _run_stage(stage_name, stage_train_loader, stage_val_loader, stage_epochs, stage_lr, stage_steps=stage_steps)
    else:
        # Single-stage legacy flow
        _run_stage('main', train_loader, val_loader, epochs, lr)

    run.finish()

    save_model(model, model_path)
    return model


def run_test_generations(model: XOR8BitLM, input_text: str, max_len: int = 512):
    model.eval()
    output = model.generate_with_cache(input_text, max_len=max_len)
    print(f"\nGenerated (top-k, temp=1.0):\n{output}")
    output = model.generate_with_cache(input_text, max_len=max_len, temp=0.5)
    print(f"\nGenerated (top-k, temp=0.5):\n{output}")

    output = model.generate_with_cache(input_text, max_len=max_len, sampling='top_h')
    print(f"\nGenerated (top-h, temp=1.0):\n{output}")
    output = model.generate_with_cache(input_text, max_len=max_len, sampling='top_h', temp=0.5)
    print(f"\nGenerated (top-h, temp=0.5):\n{output}")

# ============= EXAMPLE USAGE =============

if __name__ == "__main__":

    test_run = False
    load_and_test = False

    if load_and_test:
        model = load_model("xor_model", "mps")
        run_test_generations(model, "The world is a cold place.", max_len=512)
        exit()

    if test_run:
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
            epochs=5,
            d_model=512,
            n_heads=8,
            n_layers=8,
            rope_base=10000,
            test_prompt="The "
        )

        output = model.generate_with_cache(input_text, max_len=200)
        print(f"\nGenerated:\n{output}")
    else:

        # Train example (uncomment to run)
        model = train(
            "datasets/packed",
            seq_length=2048,
            batch_size=2,
            epochs=5,
            d_model=512,
            n_heads=8,
            n_layers=8,
            rope_base=10000,
            test_prompt="The ",
            curriculum=[
                {
                    'name': 'pretrain',
                    'packed_roots': [
                        'datasets/packed/orca-pre',
                        'datasets/packed/tiny-lessons',
                        'datasets/packed/tiny-textbooks',
                        'datasets/packed/tiny-superwiki'
                    ],
                    'epochs': 1,
                    'lr': 3e-4
                },
                {
                    'name': 'school',
                    'packed_roots': [
                        'datasets/packed/orca-inst',
                    ],

                    'steps': 100,
                    'lr': 2e-4
                }
            ]
        )

    output = model.generate_with_cache("Maailm on karm.", max_len=512)
    print(f"\nGenerated: {output}")

    output = model.generate_with_cache("Elu on ilus.", max_len=512)
    print(f"\nGenerated: {output}")

    run_test_generations(model, "The world is a cold place.", max_len=512)

    # Or train from HuggingFace
    # model = train_from_hf("wikitext", "wikitext-2-raw-v1", epochs=5)