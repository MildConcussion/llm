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
import os

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
def golden_groups(n_layers, n_heads=8, min_groups=2, max_groups=4):
    """Endpoint-correct vectorized schedule.

    Guarantees first layer uses ``max_groups`` and last layer uses ``min_groups``
    (when ``n_layers > 1``). Also caps group counts by ``n_heads``.
    """
    φ = (1 + np.sqrt(5)) / 2

    # Clamp feasible bounds by n_heads
    max_g = int(min(max_groups, n_heads))
    min_g = int(max(1, min(min_groups, max_g)))

    if n_layers <= 0:
        return []

    if n_layers == 1:
        # Single layer: use the tighter of max_g and n_heads
        ng = max_g
        return [partition_heads_golden(n_heads, ng)]

    # Geometric decay over layers (0..L-1), normalized to hit endpoints exactly
    idx = np.arange(n_layers)
    geom = φ ** (-idx)
    # Normalize to [0,1] with f[0]=1, f[-1]=0
    f = (geom - geom[-1]) / (geom[0] - geom[-1])

    n_groups = np.round(min_g + (max_g - min_g) * f).astype(int)
    # Enforce endpoints in case rounding drifted
    n_groups[0] = max_g
    n_groups[-1] = min_g

    # Monotone non-increasing (defensive smoothing)
    for i in range(1, n_layers):
        if n_groups[i] > n_groups[i-1]:
            n_groups[i] = n_groups[i-1]

    # Asymmetric per-layer partitioning using φ-phase weighting and head rotation
    golden_conjugate = φ - 1.0  # ≈ 0.618...

    groups_per_layer = []
    for i, ng in enumerate(n_groups.tolist()):
        ng = int(ng)
        # Phase in [0,1) advances quasi-uniformly across layers
        phase = (i * golden_conjugate) % 1.0

        # φ-phase shifted weights: w_j ∝ φ^{-(j + phase)}
        j = np.arange(max(ng, 1), dtype=np.float64)
        weights = φ ** (-(j + phase))
        weights = weights / weights.sum()

        # Largest-remainder allocation for sizes
        raw = weights * n_heads
        floor_sizes = np.floor(raw).astype(int)
        remainder = raw - floor_sizes
        remaining = int(n_heads - floor_sizes.sum())
        if remaining > 0:
            idx = np.argsort(-remainder)
            floor_sizes[idx[:remaining]] += 1
        sizes = floor_sizes

        # Rotate head indices by a φ-based offset for asymmetric membership
        rotate_by = int(np.floor(phase * n_heads)) % max(1, n_heads)
        head_ring = np.roll(np.arange(n_heads, dtype=int), rotate_by)

        # Form contiguous groups on the rotated ring
        groups = []
        start = 0
        for sz in sizes:
            if sz > 0:
                groups.append(head_ring[start:start+sz].tolist())
                start += sz

        groups_per_layer.append(groups)

    return groups_per_layer


# ============= ENCODER =============

class GrayCodeEncoder:
    """Gray code preserves bit locality."""

    START, EOS, PAD = 256, 257, 258

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


class SelectiveAttentionModule(nn.Module):
    """Lightweight SSA temperature module with weight sharing."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        # Single learnable parameter for position-aware scaling
        self.pos_alpha = nn.Parameter(torch.zeros(1))
        # Output projection for token-aware temperature (weight sharing with attention)
        self.temp_out = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.temp_out.weight, std=0.02)

        # Precompute position scales
        positions = torch.arange(1, max_len + 1, dtype=torch.float32)
        self.register_buffer('log_positions', torch.log(positions))

    def forward(self, x, proj_weight):
        """
        x: [B, L, D]
        proj_weight: attention projection weights for weight sharing
        Returns: [B, L, 1] temperature values
        """
        B, L, D = x.shape

        # Token-aware temperature using shared attention weights
        # Use the transpose of projection weights as feature extractor
        # proj_weight shape: [out_dim, in_dim], we need [in_dim, hidden] for feature extraction
        # So we use first d_model columns of the transposed weight
        weight_t = proj_weight.t()[:D, :min(D, proj_weight.shape[0])]  # [d_model, hidden_dim]
        features = F.linear(x, weight_t.t())  # [B, L, hidden_dim]

        # Apply GeLU and project to scalar temperature
        features_gelu = F.gelu(features)
        # Average pool over hidden dimension then project to scalar
        features_pooled = features_gelu.mean(dim=-1, keepdim=True)  # [B, L, 1]
        token_temp = torch.tanh(features_pooled)  # [B, L, 1]

        # Position-aware temperature
        pos_scale = 1 + torch.sigmoid(self.pos_alpha) * self.log_positions[:L].unsqueeze(0).unsqueeze(-1)

        return token_temp * pos_scale


class AsymGQATransformerBlock(nn.Module):
    """Transformer with Asymmetric Grouped-Query Attention"""

    def __init__(self, d_model, n_heads, d_ff, rope,
                 groups=None, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope

        # Asymmetric grouping: list of lists [[0,1,2], [3], [4,5,6,7], ...]
        # If None, use standard MHA
        self.groups = groups or [[i] for i in range(n_heads)]
        self.n_kv_heads = len(self.groups)

        # Create mapping: which KV head does each Q head use?
        self.register_buffer('kv_map', self._create_kv_map())

        # Q always full size, K/V based on groups
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN and norms unchanged
        self.ffn = FeedForward(d_model, d_ff, dtype)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.q_temp_module = SelectiveAttentionModule(d_model)
        self.v_temp_module = SelectiveAttentionModule(d_model)

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

        # Project Q (full), K/V (grouped)
        q = self.q_proj(x_norm).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Get temperatures using weight sharing
        q_temps = self.q_temp_module(x_norm, self.q_proj.weight)  # [B, L, 1]
        v_temps = self.v_temp_module(x_norm, self.v_proj.weight)  # [B, L, 1]

        # Reshape for broadcasting across sequence dim (no transpose)
        q_temps = q_temps.to(dtype=q.dtype, device=q.device).unsqueeze(1)  # [B, 1, L, 1]
        v_temps = v_temps.to(dtype=v.dtype, device=v.device).unsqueeze(1)  # [B, 1, L, 1]

        # Scale queries (controls attention spikiness)
        q = q * q_temps

        # Scale values (suppresses noise)
        v = v * v_temps.expand(-1, self.n_kv_heads, -1, -1)

        # Expand K,V to match Q heads using pre-computed mapping
        k = k[:, self.kv_map]  # [B, n_heads, L, head_dim]
        v = v[:, self.kv_map]

        # Apply RoPE (optionally with custom positions)
        q, k = self.rope(q, k, seq_len=L, positions=positions)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Support [L, L] or [B, L, L]
            if mask.dim() == 2:
                scores.masked_fill_(mask[None, None, :, :], -float('inf'))
            else:
                scores.masked_fill_(mask[:, None, :, :], -float('inf'))
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask[:, None, None, :], -float('inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)

        # Residual + FFN
        x = x + out
        x = x + self.ffn(self.norm2(x))

        return x


class XOR8BitLM(nn.Module):
    """Fast XOR-based Language Model with optional MuToR."""

    def __init__(self, d_model=512, n_heads=8, n_layers=6,
                 max_len=2048, rope_base=10000,
                 mutor_dmax: int = 0, mutor_alpha: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = GrayCodeEncoder()
        # MuToR configuration
        self.mutor_dmax = int(mutor_dmax)
        self.mutor_alpha = float(mutor_alpha)
        self.register_id = 259  # New special token for MuToR register

        # Precompute 8-bit lookup table [260, 8] for fast bit extraction
        lut_vals = torch.zeros(260, 8, dtype=torch.int64)
        byte_bits = (torch.arange(256, dtype=torch.int64).unsqueeze(-1) >> torch.arange(8, dtype=torch.int64)) & 1
        lut_vals[:256] = byte_bits
        # Special tokens bit patterns
        lut_vals[GrayCodeEncoder.START, 0] = 1
        lut_vals[GrayCodeEncoder.EOS, 1] = 1
        # PAD remains all zeros
        lut_vals[self.register_id, 2] = 1  # REGISTER unique bit
        self.register_buffer('bit_lut', lut_vals.to(torch.float32))
        # Cache for causal masks by (device, seq_len)
        self._causal_masks: dict[tuple[str, int], torch.Tensor] = {}
        # Cache for base MuToR causal masks (per seq_len) to reuse across batch
        self._mutor_base_masks: dict[tuple[str, int], torch.Tensor] = {}

        # Bit projection: 8 bits -> d_model
        self.bit_proj = nn.Linear(8, d_model)

        # Single learnable register embedding (additive bias)
        self.register_bias = nn.Parameter(torch.zeros(d_model))

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
            AsymGQATransformerBlock(d_model, n_heads, d_model * 4, self.rope, groups)
            for groups in groups_per_layer
        ])

        self.norm = RMSNorm(d_model)

        # Output head (now 260 including REGISTER)
        self.out = nn.Linear(d_model, 260)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.jit.export
    def to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence to bit features (vectorized with LUT)."""
        # Direct LUT indexing for 0..259 (bytes + specials + REGISTER)
        x_clamped = x.clamp(min=0, max=self.bit_lut.shape[0] - 1)
        return self.bit_lut[x_clamped]

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return cached upper-triangular causal mask of shape [L, L] (bool)."""
        key = (str(device), seq_len)
        mask = self._causal_masks.get(key)
        if mask is None or mask.device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
            self._causal_masks[key] = mask
        return mask

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                is_register: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional MuToR positions/registers support."""
        B, L = x.shape

        # Convert to bits and project
        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        # Add register bias where applicable
        if is_register is not None:
            h = h + self.register_bias.to(h.dtype) * is_register.unsqueeze(-1).to(h.dtype)

        # Build attention mask
        if is_register is None:
            mask = self._get_causal_mask(L, x.device)
        else:
            base = self._get_causal_mask(L, x.device)  # [L, L]
            # Expand to [B, L, L]
            mask = base.unsqueeze(0).expand(B, L, L).clone()
            # Block attending to any register positions (for all queries)
            mask |= is_register[:, None, :].expand(B, L, L)
            # Note: this also blocks registers->registers as desired

        # Key padding mask: True where token is PAD
        key_padding_mask = (x == self.encoder.PAD)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, mask, key_padding_mask, positions=positions)

        # Final norm and output
        h = self.norm(h)
        return self.out(h)

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
            if hasattr(self, 'register_id'):
                logits[..., self.register_id] = -float('inf')

            # Check if we have any valid logits
            if not torch.isfinite(logits).any():
                # Emergency fallback: allow all byte values
                logits = torch.zeros_like(logits)
                logits[..., :256] = 1.0  # Equal probability for all bytes
                logits[..., self.encoder.EOS] = 1.0  # Allow EOS
                logits[..., self.encoder.START] = -float('inf')
                logits[..., self.encoder.PAD] = -float('inf')
                if hasattr(self, 'register_id'):
                    logits[..., self.register_id] = -float('inf')
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
                 total_steps: int | None = None, ema_alpha=0.99):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps

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
            if len(param.shape) == 1 or "bit_proj" in name or name.endswith('.bias') or '.norm' in name or name.endswith('.scale') or name.endswith('.shift'):
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

            # Use same autocast context as training
            if self.device == 'cuda':
                autocast_ctx = torch.amp.autocast(device_type='cuda')
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, 259),
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
        batch = batch.to(self.device)

        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with self.autocast_ctx:
            if use_mutor and getattr(self.model, 'mutor_dmax', 0) > 0:
                aug_batch, positions, is_register, aug_targets = mutor_augment_batch(inputs, targets, self.model, self.device)
                logits = self.model(aug_batch, positions=positions, is_register=is_register)
                loss = compute_mutor_loss(logits, aug_targets, is_register, self.encoder.PAD, alpha=self.model.mutor_alpha)
            else:
                logits = self.model(inputs)
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
        batch = batch.to(self.device)
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
    aug_batch[:, 1::2] = int(model.register_id)

    # Positions for RoPE
    positions = torch.zeros((B, L2), device=device, dtype=torch.long)
    # Even (original tokens): 0..L-1
    base_pos = torch.arange(L, device=device, dtype=torch.long)
    positions[:, 0::2] = base_pos
    # Odd (registers): pos = min(p + d - 1, L - 1)
    reg_pos = torch.clamp(base_pos + (d - 1), max=L - 1)
    positions[:, 1::2] = reg_pos

    is_register = (aug_batch == int(model.register_id))

    # Build augmented targets
    aug_targets = torch.full_like(aug_batch, model.encoder.PAD)
    # Next-token targets at even indices
    aug_targets[:, 0::2] = targets
    # Register targets: token at index min(p + d, L - 1) from inputs
    reg_target_idx = torch.clamp(base_pos + d, max=L - 1)
    reg_targets = inputs[:, reg_target_idx]
    aug_targets[:, 1::2] = reg_targets

    if os.environ.get('MUTOR_DEBUG', '0') == '1':
        print(f"[mutor] d={d}, L={L}, L2={L2}")

    return aug_batch, positions, is_register, aug_targets


def compute_mutor_loss(logits: torch.Tensor, aug_targets: torch.Tensor, is_register: torch.Tensor,
                       pad_id: int, alpha: float = 0.3) -> torch.Tensor:
    """Combine next-token and register CE losses with weight alpha.
    logits: [B, L2, V], aug_targets: [B, L2], is_register: [B, L2] (bool)
    """
    B, L2, V = logits.shape
    logits_f = logits.reshape(B * L2, V)
    targets_f = aug_targets.reshape(B * L2)
    is_reg_f = is_register.reshape(B * L2)

    # Masks
    valid = targets_f != pad_id
    mask_nt = (~is_reg_f) & valid
    mask_reg = is_reg_f & valid

    loss_nt = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    loss_reg = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    if mask_nt.any():
        loss_nt = F.cross_entropy(logits_f[mask_nt], targets_f[mask_nt])
    if mask_reg.any():
        loss_reg = F.cross_entropy(logits_f[mask_reg], targets_f[mask_reg])

    return (1.0 - alpha) * loss_nt + alpha * loss_reg

def save_model(model: XOR8BitLM, path: Union[str, Path], config: Dict[str, Any] = None):
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
):
    """Complete training pipeline."""

    print(f"Training on {device}")

    # Helper to build device-tuned DataLoader args
    def _loader_kwargs(for_eval: bool = False):
        if device == 'cuda':
            workers = min(8, (os.cpu_count() or 8))
            return dict(num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, shuffle=not for_eval)
        elif device == 'mps':
            return dict(num_workers=2, pin_memory=False, persistent_workers=True, prefetch_factor=2, shuffle=not for_eval)
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

    # Create model
    model = XOR8BitLM(d_model, n_heads, n_layers, seq_length, rope_base)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Create trainer with correct OneCycle total steps
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * epochs
    trainer = Trainer(model, lr=lr, device=device, total_steps=total_steps,
                      warmup_steps=min(1000, total_steps // 10), ema_alpha=ema_alpha)

    val_iter = itertools.cycle(val_loader)

    # Training loop with evaluation
    global_step = 0

    # Training loop
    for epoch in range(epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, train_batch in enumerate(pbar):
            # Training step (automatically updates streaming train loss)
            loss = trainer.train_step(train_batch)
            global_step += 1

            # Streaming evaluation update
            if global_step % eval_interval == 0:
                val_batch = next(val_iter)
                trainer.quick_eval_update(val_batch)

            # Update progress bar with streaming metrics
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'train_ema': f"{trainer.metrics.train_loss_ema:.4f}" if trainer.metrics.train_loss_ema else "N/A",
                'eval_ema': f"{trainer.metrics.eval_loss_ema:.4f}" if trainer.metrics.eval_loss_ema else "N/A",
                'perp': f"{trainer.metrics.perp_ema:.2f}" if trainer.metrics.perp_ema else "N/A",
                'grok': f"{trainer.metrics.get_signal():.3f}"
            })

        # Full evaluation at epoch end (optional, for logging)
        model.eval()
        eval_losses = []
        for i, batch in enumerate(itertools.islice(val_loader, 100)):  # Sample 100 batches
            loss, n_tokens = trainer.eval_step(batch)
            if n_tokens > 0:
                eval_losses.append(loss)

        if eval_losses:
            avg_eval_loss = np.mean(eval_losses)
            epoch_perplexity = math.exp(min(avg_eval_loss, 20))

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Streaming - Train EMA: {trainer.metrics.train_loss_ema:.4f}, "
                  f"Eval EMA: {trainer.metrics.eval_loss_ema:.4f}, "
                  f"Perp EMA: {trainer.metrics.perp_ema:.2f}")
            print(f"  Full Eval - Loss: {avg_eval_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            print(f"  Grokking Signal: {trainer.metrics.get_signal():.3f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_model(model, model_path, {
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'n_layers': len(model.layers),
                'max_len': model.rope.max_seq_len,
                'rope_base': model.rope.base,
                'best_perplexity': trainer.metrics.best_perp,
                'epoch': epoch + 1
            })

        # Generate sample
        model.eval()
        sample = model.generate("The ", max_len=60)
        print(f"\nSample: {sample}\n")

    # Final save
    save_model(model, model_path)
    return model

# ============= HF DATASET SUPPORT =============

def train_from_hf(dataset_name: str, **kwargs):
    """Train from HuggingFace dataset."""
    from datasets import load_dataset
    import tempfile

    # Load dataset
    ds = load_dataset(dataset_name, split='train')

    # Extract text and save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for item in tqdm(ds, desc="Processing dataset"):
            # Adjust field name as needed (text, content, etc.)
            text = item.get('text', item.get('content', str(item)))
            f.write(text + '\n')
        temp_path = f.name

    # Train
    model = train(temp_path, **kwargs)

    # Cleanup
    Path(temp_path).unlink()
    return model

# ============= EXAMPLE USAGE =============

if __name__ == "__main__":
    # Quick test
    print("Testing XOR8Bit Language Model")
    print("="*60)

    # Test encoder
    enc = GrayCodeEncoder()
    text = "Hello World!"
    encoded = enc.encode(text)
    decoded = enc.decode(encoded)
    print(f"Original: {text}")
    print(f"Encoded: {encoded[:10]}...")
    print(f"Decoded: {decoded}")
    print(f"Match: {'✓' if text == decoded else '✗'}")

    input_text = """
ALL:
Content, content.

MENENIUS:
O sir, you are not right: have you not known
The worthiest men have done't?

CORIOLANUS:
""".strip()

    encoded = enc.encode(input_text)
    decoded = enc.decode(encoded)
    print(f"Original: {input_text}")
    print(f"Encoded: {encoded}")
    print(f"Encoded length: {len(encoded)}")
    print(f"Decoded: {decoded}")
    print(f"Decoded length: {len(decoded)}")
    print(f"Match: {'✓' if input_text == decoded else '✗'}")

    # Train example (uncomment to run)
    model = train(
        "data/tiny_shakespeare.txt",
        seq_length=512,
        batch_size=8,
        epochs=5,
        d_model=512,
        n_heads=8,
        n_layers=8,
        rope_base=10000)

    output = model.generate(input_text, max_len=200)
    print(f"\nGenerated:\n{output}")

    output = model.generate("The world is a cold place.", max_len=200)
    print(f"\nGenerated (trained, unrepresented text): {output}")

    # Or train from HuggingFace
    # model = train_from_hf("wikitext", "wikitext-2-raw-v1", epochs=5)