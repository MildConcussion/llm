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
torch.manual_seed(42)
np.random.seed(42)

# ============= INITIALIZATION UTILITIES =============

def _init_weights_standard(module, std=0.02, mean=0.0):
    """Unified weight initialization for all module types."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif hasattr(module, 'scale'):  # RMSNorm
        nn.init.ones_(module.scale)
        if hasattr(module, 'shift') and module.shift is not None:
            nn.init.zeros_(module.shift)
    # Tau parameters are handled in their respective modules


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

    def _encode_qwen_message(self, role: str, content: str) -> np.ndarray:
        """Encode one Qwen-style message segment with IM_START/IM_END and role+content.
        Layout:
        [IM_START] + role + "\n" + content + [IM_END] + "\n"
        Roles are included as raw utf-8 bytes.
        """
        role_bytes = (role + "\n").encode('utf-8', errors='ignore')
        content_bytes = (content or "").encode('utf-8', errors='ignore')

        parts: List[np.ndarray] = []
        parts.append(np.array([self.IM_START], dtype=np.uint16))
        if role_bytes:
            rb = np.frombuffer(role_bytes, dtype=np.uint8)
            parts.append(self.gray_lut[rb].astype(np.uint16, copy=False))
        if content_bytes is not None:
            parts.append(self.gray_lut[np.frombuffer(content_bytes, dtype=np.uint8)].astype(np.uint16, copy=False))
        parts.append(np.array([self.IM_END], dtype=np.uint16))
        # trailing newline after IM_END
        parts.append(self.gray_lut[np.frombuffer(b"\n", dtype=np.uint8)].astype(np.uint16, copy=False))
        return np.concatenate(parts)

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

@torch.jit.script
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        return apply_rotary_pos_emb(q, k, cos, sin)

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
        # Match buffer dtype to parameter dtype to avoid casts in einsum
        group_agg = torch.zeros(self.n_kv_heads, self.n_heads, dtype=dtype)
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

        # Initialize tau gating parameters
        nn.init.normal_(self.tau_wq, std=0.02)
        nn.init.normal_(self.tau_wv_kv, std=0.02)
        nn.init.zeros_(self.tau_alpha)

        # LRU cache for position logs per (device, seq_len)
        # Implemented via shared LRUCache utility
        self.poslog_cache = LRUCache(capacity=64, move_tensors_to_device=True)

        if not hasattr(self, '_kv_map_expanded'):
            self.register_buffer('_kv_map_expanded', self.kv_map.view(1, 1, -1))

        # Fused computation for grouped tau
        if not hasattr(self, '_group_agg_wv'):
            # Precompute this product once during init
            with torch.no_grad():
                self._group_agg_wv = torch.einsum('gh,gd->ghd',
                                                self.group_agg,
                                                self.tau_wv_kv)
                self.register_buffer('_group_agg_wv_cached', self._group_agg_wv)

    def _get_pos_log_cached(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return cached log1p(arange(L)) [L] float32 on device (LRU)."""
        key = (str(device), int(seq_len))
        def _factory():
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            return torch.log1p(pos)
        return self.poslog_cache.get(key, factory=_factory, device=device)

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
        # Optimize tau computation with fewer intermediates
        # GELU features can be computed normally (no need for no_grad)
        # Compute GELU features once
        tok_feat_q = F.gelu(q_proj).reshape(B, L, self.n_heads, self.head_dim)

        # Single einsum for tau_tok_q
        tau_tok_q = torch.einsum('blhd,hd->blh', tok_feat_q, self.tau_wq)
        tau_tok_q = torch.tanh(tau_tok_q)

        # Now use the precomputed weights
        tau_tok_v_grouped = torch.tanh(
            torch.einsum('blhd,ghd->blg', tok_feat_q, self._group_agg_wv_cached)
        )

        # Gather using cached expansion
        tau_tok_v = tau_tok_v_grouped.gather(2, self._kv_map_expanded.expand(B, L, -1))

        # Position computation
        if positions is None:
            pos_log = self._get_pos_log_cached(L, x.device).unsqueeze(0).expand(B, -1)
        else:
            pos_log = torch.log1p(positions.float())

        # Alpha and tau_pos computation
        alpha = torch.sigmoid(self.tau_alpha).view(1, self.n_heads, 1)
        pos_log_expanded = pos_log.view(B, 1, L)
        tau_pos = alpha.mul(pos_log_expanded).add_(1.0 - 0.5)

        # Final taus - create new tensors, no in-place modification
        tau_q = tau_tok_q.transpose(1, 2).add(tau_pos).unsqueeze(-1)
        tau_v = tau_tok_v.transpose(1, 2).add(tau_pos).unsqueeze(-1)

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

@torch.jit.script
def gray_code_lut_apply(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Fast Gray code lookup - eliminates bounds checking overhead."""
    x_clamped = torch.clamp(x, min=0, max=lut.shape[0] - 1)
    return lut[x_clamped]

class MesicapLM(nn.Module):
    """Fast XOR-based Language Model."""

    def __init__(self, d_model=512, n_heads=8, n_layers=6,
                 max_len=2048, rope_base=10000,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = GrayCodeEncoder()
        self.param_dtype = dtype

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

        # Bit projection: 8 bits -> d_model
        self.bit_proj = nn.Linear(8, d_model, dtype=dtype)

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
            AsymGQATransformerBlock(d_model, n_heads, d_model * 12 if i == 0 else d_model * 4, self.rope, groups, dtype=dtype)
            for i, groups in enumerate(groups_per_layer)
        ])

        self.norm = RMSNorm(d_model)

        # Output head (now 260 including REGISTER)
        self.out = nn.Linear(d_model, len(self.encoder), dtype=dtype)

        # Initialize weights
        self.apply(_init_weights_standard)

        # Shared LRU cache for causal masks per (device, L)
        self._causal_mask_cache = LRUCache(capacity=64, move_tensors_to_device=True)

    @torch.jit.export
    def to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence to bit features (vectorized with LUT)."""
        # Direct LUT indexing for 0..259 (bytes + specials + REGISTER)
        return gray_code_lut_apply(x, self.bit_lut)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return cached upper-triangular causal mask of shape [L, L] (bool) using shared LRU."""
        key = (str(device), int(seq_len))
        def _factory():
            return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
        return self._causal_mask_cache.get(key, factory=_factory, device=device)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                return_cache: bool = False, past_kv: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None) -> Union[torch.Tensor, tuple[torch.Tensor, List]]:
        """Unified forward pass supporting all modes.

        Args:
            x: [B, L] token ids
            positions: [B, L] float32 or None for RoPE
            return_cache: if True, return (logits, new_past_kv) for KV-cache
            past_kv: list of (k, v) per layer for incremental generation

        Returns:
            logits [B, L, V] or (logits, new_past_kv) if return_cache=True
        """
        B, L = x.shape

        if (x == self.encoder.PAD).all():
            empty_logits = torch.zeros(B, L, len(self.encoder), device=x.device, dtype=self.param_dtype)
            return empty_logits if not return_cache else (empty_logits, [])

        # Convert to bits and project
        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        mask = self._get_causal_mask(L, x.device) if not return_cache else None

        key_padding_mask = (x == self.encoder.PAD)

        # Forward through layers
        new_past = []
        if return_cache:
            # KV-cache mode
            use_incremental = past_kv is not None and L == 1
            if not use_incremental:
                for layer in self.layers:
                    h, k_present, v_present = layer.full_pass_return_kv(h, mask, key_padding_mask, positions=positions)
                    new_past.append((k_present, v_present))
            else:
                assert positions is not None, "positions required for incremental generation"
                for i, layer in enumerate(self.layers):
                    k_prev, v_prev = past_kv[i]
                    h, k_all, v_all = layer.forward_incremental(h, positions, k_prev, v_prev)
                    new_past.append((k_all, v_all))
        else:
            # Standard forward
            for layer in self.layers:
                h = layer(h, mask, key_padding_mask, positions=positions)

        # Final norm and output
        h = self.norm(h)
        logits = self.out(h)

        return (logits, new_past) if return_cache else logits

    @torch.no_grad()
    def forward_with_cache(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                           past_kv: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None):
        """KV-cache aware forward - delegates to unified forward."""
        return self.forward(x, positions=positions, return_cache=True, past_kv=past_kv)


    @torch.no_grad()
    def generate(self, prompt="", max_len=100, temp=1.0, sampling='top_p', top_p=0.9, alpha=0.4,
                 debug=False, apply_chat_template=False, use_cache=True):
        """Unified generation with optional KV-cache. Supports top-p and top-h sampling."""
        self.eval()
        device = next(self.parameters()).device
        sampler = SamplingStrategy()

        # Prepare input sequence
        if apply_chat_template:
            system_preface = "You are Kulles, created by Rasmus. You are a helpful assistant."
            base_segs = [
                self.encoder._encode_qwen_message('system', system_preface),
                self.encoder._encode_qwen_message('user', prompt)
            ]
            parts = [np.array([self.encoder.START], dtype=np.uint16)]
            parts.extend(base_segs)
            parts.append(np.array([self.encoder.IM_START], dtype=np.uint16))
            role_bytes = ("assistant" + "\n").encode('utf-8', errors='ignore')
            rb = np.frombuffer(role_bytes, dtype=np.uint8)
            parts.append(self.encoder.gray_lut[rb].astype(np.uint16, copy=False))
            seq = np.concatenate(parts)
        else:
            seq = self.encoder.encode(prompt)

        if len(seq) > 0 and seq[-1] == self.encoder.EOS:
            seq = seq[:-1]
        x = torch.from_numpy(seq).long().unsqueeze(0).to(device)

        if not use_cache:
            # Simple generation without cache
            for _ in range(max_len):
                if x.size(1) > self.rope.max_seq_len:
                    x = x[:, -self.rope.max_seq_len:]

                logits = self(x)[:, -1, :] / max(temp, 1e-6)
                next_token = self._sample_next_token(logits, sampler, sampling, top_p, alpha, debug)

                if next_token.item() == self.encoder.EOS:
                    break
                x = torch.cat([x, next_token], dim=1)
        else:
            # Cached generation
            cur_len = x.size(1)
            pos = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
            logits, past_kv = self.forward_with_cache(x, positions=pos, past_kv=None)

            # Pre-allocate KV up to cap
            max_cache_len = min(cur_len + max_len, self.rope.max_seq_len)
            pre_allocated_kv = []
            for k, v in past_kv:
                k_cache = torch.zeros(k.size(0), k.size(1), max_cache_len, k.size(3), device=device, dtype=k.dtype)
                v_cache = torch.zeros(v.size(0), v.size(1), max_cache_len, v.size(3), device=device, dtype=v.dtype)
                k_cache[:, :, :cur_len] = k
                v_cache[:, :, :cur_len] = v
                pre_allocated_kv.append((k_cache, v_cache))

            for _ in range(max_len):
                if cur_len >= max_cache_len:
                    if debug:
                        print(f"[gen-cache] reached max_cache_len={max_cache_len}; stopping")
                    break

                last_logits = logits[:, -1, :] / max(temp, 1e-6)
                next_token = self._sample_next_token(last_logits, sampler, sampling, top_p, alpha, debug)

                if next_token.item() == self.encoder.EOS:
                    break

                # Update seq and cache step
                x_next = next_token
                x = torch.cat([x, x_next], dim=1)
                cur_len += 1
                pos_next = torch.tensor([[cur_len - 1]], device=device, dtype=torch.float32)

                current_kv = [(k[:, :, :cur_len-1], v[:, :, :cur_len-1]) for k, v in pre_allocated_kv]
                logits, new_kv = self.forward_with_cache(x_next, positions=pos_next, past_kv=current_kv)

                if cur_len < max_cache_len:
                    for i, (k_new, v_new) in enumerate(new_kv):
                        pre_allocated_kv[i][0][:, :, cur_len-1:cur_len] = k_new[:, :, -1:]
                        pre_allocated_kv[i][1][:, :, cur_len-1:cur_len] = v_new[:, :, -1:]
                else:
                    if debug:
                        print(f"[gen-cache] reached max_cache_len={max_cache_len}")
                    break

                if debug:
                    last_tokens = x[0, -4:].tolist() if x.size(1) >= 4 else x[0].tolist()
                    token_id = next_token.item()
                    special = 'EOS' if token_id == self.encoder.EOS else ('START' if token_id == self.encoder.START else ('PAD' if token_id == self.encoder.PAD else ''))
                    print(f"[gen-cache] last={last_tokens} -> next={token_id}{'('+special+')' if special else ''}")

        return self.encoder.decode(x[0].cpu().numpy())

    def _sample_next_token(self, logits, sampler, sampling, top_p, alpha, debug):
        """Helper to sample next token with special token masking and fallback."""
        # Mask specials
        logits[..., self.encoder.START] = -float('inf')
        logits[..., self.encoder.PAD] = -float('inf')
        logits[..., self.encoder.REGISTER] = -float('inf')
        if hasattr(self.encoder, 'IM_START'):
            logits[..., self.encoder.IM_START] = -float('inf')
        if hasattr(self.encoder, 'IM_END'):
            logits[..., self.encoder.IM_END] = -float('inf')

        if not torch.isfinite(logits).any():
            if debug:
                print("[gen] fallback logits")
            logits = torch.zeros_like(logits)
            logits[..., :256] = 1.0
            logits[..., self.encoder.EOS] = 1.0
            logits[..., self.encoder.START] = -float('inf')
            logits[..., self.encoder.PAD] = -float('inf')
            logits[..., self.encoder.REGISTER] = -float('inf')

        next_token = sampler.sample(logits, sampling=sampling, top_p=top_p, alpha=alpha, debug=debug)

        if debug:
            token_id = next_token.item()
            special = 'EOS' if token_id == self.encoder.EOS else ('START' if token_id == self.encoder.START else ('PAD' if token_id == self.encoder.PAD else ''))
            print(f"[gen] next={token_id}{'('+special+')' if special else ''}")

        return next_token


# ============= TRAINING =============

class StreamingGrokMetrics:
    """Minimal, robust grokking detector using scale-free metrics with smoothing."""

    def __init__(self, alpha_slow=0.99, alpha_fast=0.9,
                 k_loss: float = 3.0, k_perp: float = 2.0, k_improve: float = 4.0,
                 signal_smooth_alpha: float = 0.9, train_loss_boost_factor: float = 2.0,
                 eval_fast_penalty_threshold: float = 1.5, eval_fast_penalty_factor: float = 0.5):
        # EMA timescales
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)

        # Hyperparameters
        self.k_loss = float(k_loss)
        self.k_perp = float(k_perp)
        self.k_improve = float(k_improve)
        self.signal_smooth_alpha = float(signal_smooth_alpha)
        self.train_loss_boost_factor = float(train_loss_boost_factor)
        self.eval_fast_penalty_threshold = float(eval_fast_penalty_threshold)
        self.eval_fast_penalty_factor = float(eval_fast_penalty_factor)

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

        if self.signal_ema is None:
            self.signal_ema = signal_raw
        else:
            a = self.signal_smooth_alpha
            self.signal_ema = a * self.signal_ema + (1 - a) * signal_raw

        return float(self.signal_ema)


class MultiGrokOptimizer:
    """Single unified optimizer managing multiple component-specific optimizers."""

    COMPONENT_CONFIG = {
        'embeddings': {'lr_scale': 1.0, 'weight_decay': 0.01, 'optimizer': 'adamw', 'grok': False},
        'attention': {'lr_scale': 0.8, 'weight_decay': 0.01, 'optimizer': 'grokadamw', 'grok': True, 'gradient_clipping': 0.25},
        'ffn': {'lr_scale': 1.2, 'weight_decay': 0.01, 'optimizer': 'grokadamw', 'grok': True, 'grok_scale': 1.2, 'gradient_clipping': 1.0},
        'biases': {'lr_scale': 2.0, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'layer_norm': {'lr_scale': 1.5, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'tau': {'lr_scale': 0.1, 'weight_decay': 0.0, 'optimizer': 'adam', 'grok': False, 'gradient_clipping': 0.2},
        'bit_proj': {'lr_scale': 0.5, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False},
        'output': {'lr_scale': 0.3, 'weight_decay': 0.0, 'optimizer': 'adamw', 'grok': False, 'gradient_clipping': 0.1},
    }

    def __init__(self, model, base_lr=3e-4, weight_decay=0.01, gradient_clipping=1.0):
        self.model = model
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping

        # Extract number of layers from model
        self.n_layers = len(self.model.layers)

        # Regex patterns for parameter parsing
        import re
        self.patterns = {
            'layer_idx': re.compile(r'layers\.(\d+)\.')
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
            if 'layers.' in name:
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
            if 'tau' in name:
                component = 'tau'
            elif 'bit_proj' in name and not name.endswith('.bias'):
                component = 'bit_proj'
            elif name.endswith('.bias'):
                component = 'biases'
            elif 'norm' in name or 'scale' in name or 'shift' in name:
                component = 'layer_norm'
            elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj', 'out_proj']):
                component = 'attention'
            elif any(x in name for x in ['fc1', 'fc2', 'fc3', 'ffn']):
                component = 'ffn'
            elif 'out' in name:
                component = 'output'
            elif 'embedding' in name or 'offset_embeddings' in name:
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

    def _unpack_batch(self, batch):
        """Unpack batch tuple (tokens, loss_mask) or return tensor directly."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            tokens, loss_mask = batch
            return tokens.to(self.device), loss_mask.to(self.device)
        else:
            return batch.to(self.device), None

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
        'n_layers': len(model.layers),
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

    def create_datasets_and_loaders(self, data_path, val_path, packed_dirs, val_packed_dirs,
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
            train_dataset = XORDataset(data_path, seq_length)

        train_loader = DataLoader(train_dataset, batch_size, **self._get_dataloader_kwargs(for_eval=False))

        # Validation dataset
        if val_packed_dirs is not None and len(val_packed_dirs) > 0:
            val_dataset = self._build_packed_dataset([str(p) for p in val_packed_dirs], seq_length, mixing_policy, mixing_weights)
        elif val_path is not None:
            vp = Path(val_path)
            if vp.is_dir() and discover_shards(str(vp)):
                val_dataset = self._build_packed_dataset([vp], seq_length, mixing_policy, mixing_weights)
            else:
                val_dataset = XORDataset(val_path, seq_length)
        else:
            if use_packed:
                # Heuristic: reuse train roots for eval with full-stride windows (no shuffle in loader)
                if packed_dirs is not None and len(packed_dirs) > 0:
                    val_dataset = self._build_packed_dataset([str(p) for p in packed_dirs], seq_length, mixing_policy, mixing_weights)
                else:
                    val_dataset = self._build_packed_dataset([dp], seq_length, mixing_policy, mixing_weights)
            else:
                # Use 10% of training data with different stride for pseudo-validation
                val_dataset = XORDataset(data_path, seq_length, stride=seq_length)

        val_loader = DataLoader(val_dataset, batch_size, **self._get_dataloader_kwargs(for_eval=True))

        return train_loader, val_loader

    def create_curriculum_datasets_and_loaders(self, curriculum_stage: Dict[str, Any], seq_length: int, batch_size: int,
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

        stage_train_loader = DataLoader(stage_train_ds, batch_size, **self._get_dataloader_kwargs(for_eval=False))
        stage_val_loader = DataLoader(stage_val_ds, batch_size, **self._get_dataloader_kwargs(for_eval=True))

        return stage_train_loader, stage_val_loader


class CurriculumManager:
    """Manages curriculum learning stages and their execution."""

    def __init__(self, dataset_builder: DatasetBuilder, seq_length: int, batch_size: int,
                 mixing_policy: str, mixing_weights: Optional[List[float]]):
        self.dataset_builder = dataset_builder
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
                stage, self.seq_length, self.batch_size, self.mixing_policy, self.mixing_weights
            )

            # Run the stage
            stage_runner.run_stage(stage_name, stage_train_loader, stage_val_loader, stage_epochs, stage_lr, stage_steps)


class StageRunner:
    """Handles execution of individual training stages."""

    def __init__(self, model: MesicapLM, device: str, eval_interval: int, ema_alpha: float,
                 test_prompt: str, model_path: Union[str, Path], run):
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
            n_freeze = int(len(self.model.layers) * 0.8)  # Freeze 60% of layers

            # Freeze bit projection
            self.model.bit_proj.weight.requires_grad = False

            # Freeze RoPE (positional encoding)
            for param in self.model.rope.parameters():
                param.requires_grad = False

            # Freeze lower transformer layers
            for i in range(n_freeze):
                for param in self.model.layers[i].parameters():
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
            ema_alpha=self.ema_alpha
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
                'n_layers': len(self.model.layers),
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
            "architecture": "MesicapLM",
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

    # Create model once and reuse across stages
    model = MesicapLM(d_model, n_heads, n_layers, seq_length, rope_base)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Create helper classes
    dataset_builder = DatasetBuilder(device)
    curriculum_manager = CurriculumManager(dataset_builder, seq_length, batch_size, mixing_policy, mixing_weights)
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

    output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True)
    print(f"\nGenerated (chat template, temp=1.0):\n{output}\n")
    output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True, sampling='top_h')
    print(f"\nGenerated (chat template, top-h):\n{output}\n")

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
                epochs=5,
                d_model=512,
                n_heads=8,
                n_layers=8,
                rope_base=10000,
                test_prompt="The "
            )

            output = model.generate(input_text, max_len=200)
            print(f"\nGenerated:\n{output}")

        else:
            model = train(
            "datasets/packed",
            model_path="xor_test",
            seq_length=514,
            eval_interval=10,
            batch_size=8,
            epochs=5,
            d_model=48,
            n_heads=6,
            n_layers=6,
            rope_base=10000,
            test_prompt="The ",
            mixing_policy='round_robin_wrap',
            curriculum=[
                {
                    'name': 'pretrain',
                    'packed_roots': [
                        'datasets/packed/tiny-stories-512',
                    ],
                    'epochs': 1,
                    #'steps': 1500,
                    'lr': 3e-4
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

    output = model.generate("What is 2 + 2?", max_len=512, apply_chat_template=True)
    print(f"\nGenerated: {output}")

    run_test_generations(model, "The world is a cold place.", max_len=512)
