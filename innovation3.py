"""
8-Bit XOR Language Model - Complete Implementation
Fast, elegant, vocabulary-free language modeling
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
import contextlib
from tqdm import tqdm

# ============= ENCODER =============

class XOR8BitEncoder:
    """Stateless XOR encoder - no vocabulary needed."""

    START, EOS, PAD = 256, 257, 258

    def encode(self, text: str) -> np.ndarray:
        """Encode text to XOR sequence."""
        if not text:
            return np.array([self.START, self.EOS], dtype=np.int64)

        text_bytes = text.encode('utf-8', errors='ignore')
        output = np.zeros(len(text_bytes) + 2, dtype=np.int64)

        output[0] = self.START
        output[1] = text_bytes[0]

        # XOR consecutive bytes
        for i in range(1, len(text_bytes)):
            output[i + 1] = text_bytes[i - 1] ^ text_bytes[i]

        output[-1] = self.EOS
        return output

    def decode(self, seq: np.ndarray) -> str:
        """Decode XOR sequence back to text."""
        if len(seq) < 2 or seq[0] != self.START:
            return ""

        result = []
        i = 1

        # First byte
        if i < len(seq) and seq[i] not in (self.EOS, self.PAD):
            if seq[i] < 256:
                result.append(int(seq[i]))
                i += 1

        # Reconstruct via XOR
        while i < len(seq) and seq[i] not in (self.EOS, self.PAD):
            if seq[i] < 256 and result:
                result.append(result[-1] ^ int(seq[i]))
            i += 1

        return bytes(result).decode('utf-8', errors='ignore')

# ============= DATASET =============

class XORDataset(Dataset):
    """Efficient dataset for XOR sequences."""

    def __init__(self, data_path: Union[str, Path], seq_length: int = 512,
                 stride: Optional[int] = None):
        self.encoder = XOR8BitEncoder()
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

    def forward(self, q, k, seq_len=None):
        """Apply rotary embeddings to queries and keys.
        Input shape: [B, H, L, head_dim]
        """
        if seq_len is None:
            seq_len = q.shape[2]  # Note: shape[2] because input is [B, H, L, head_dim]

        # Use precomputed values
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

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

class TransformerBlock(nn.Module):
    """Transformer layer with integrated RoPE - Fixed."""

    def __init__(self, d_model, n_heads, d_ff, rope, dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope = rope

        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dtype)

        # Layer norms (pre-norm architecture)
        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)


    def forward(self, x, mask=None, key_padding_mask=None):
        B, L, D = x.shape
        H = self.n_heads
        head_dim = D // H

        # Pre-norm and attention
        x_norm = self.norm1(x)

        # Project to Q, K, V and reshape to [B, H, L, head_dim]
        q = self.q_proj(x_norm).reshape(B, L, H, head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, L, H, head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, L, H, head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q, k = self.rope(q, k, seq_len=L)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            scores.masked_fill_(mask[None, None, :, :], -float('inf'))
        if key_padding_mask is not None:
            # Do not attend to PAD keys
            scores.masked_fill_(key_padding_mask[:, None, None, :], -float('inf'))

        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)

        # Residual connection
        x = x + out

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))

        return x

class XOR8BitLM(nn.Module):
    """Fast XOR-based Language Model."""

    def __init__(self, d_model=512, n_heads=8, n_layers=6,
                 max_len=2048, rope_base=10000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = XOR8BitEncoder()

        # Bit projection: 8 bits -> d_model
        self.bit_proj = nn.Linear(8, d_model)

        # RoPE for positional encoding
        if d_model % n_heads != 0 or ((d_model // n_heads) % 2 != 0):
            raise ValueError(
                f"Invalid head configuration: d_model={d_model}, n_heads={n_heads}. "
                f"Require d_model % n_heads == 0 and even head_dim for RoPE."
            )
        head_dim = d_model // n_heads
        self.rope = RotaryEmbedding(head_dim, max_len, rope_base)

        # Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4,
                                    self.rope)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

        # Output head
        self.out = nn.Linear(d_model, 259)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.jit.export
    def to_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence to bit features - FIXED VERSION."""
        B, L = x.shape
        device = x.device

        # Create bit representation
        bits = torch.zeros(B, L, 8, device=device, dtype=torch.float32)

        # Handle normal bytes (0-255)
        normal_mask = x < 256

        # Efficient bit extraction using broadcasting
        if normal_mask.any():
            # Get positions where we have normal bytes
            normal_vals = x.masked_fill(~normal_mask, 0)  # Zero out special tokens

            # Extract bits using bitwise operations
            for i in range(8):
                bit_vals = ((normal_vals >> i) & 1).float()
                bits[:, :, i] = bit_vals * normal_mask.float()

        # Handle special tokens
        # START token (256): set bit 0
        start_mask = (x == 256).unsqueeze(-1)
        bits[:, :, 0] = torch.where(start_mask[:, :, 0],
                                    torch.ones_like(bits[:, :, 0]),
                                    bits[:, :, 0])

        # EOS token (257): set bit 1
        eos_mask = (x == 257).unsqueeze(-1)
        bits[:, :, 1] = torch.where(eos_mask[:, :, 0],
                                   torch.ones_like(bits[:, :, 1]),
                                   bits[:, :, 1])

        # PAD token (258): stays all zeros

        return bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with RoPE."""
        B, L = x.shape

        # Convert to bits and project
        bits = self.to_bits(x)
        h = self.bit_proj(bits)

        # Create causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        # Key padding mask: True where token is PAD
        key_padding_mask = (x == self.encoder.PAD)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, mask, key_padding_mask)

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

            # Prevent sampling of START and PAD tokens
            logits[..., self.encoder.START] = -float('inf')
            logits[..., self.encoder.PAD] = -float('inf')

            # Check if we have any valid logits
            if not torch.isfinite(logits).any():
                # Emergency fallback: allow all byte values
                logits = torch.zeros_like(logits)
                logits[..., :256] = 1.0  # Equal probability for all bytes
                logits[..., self.encoder.EOS] = 1.0  # Allow EOS
                logits[..., self.encoder.START] = -float('inf')
                logits[..., self.encoder.PAD] = -float('inf')
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

            # Top-p filtering
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_idx_mask = cumsum - sorted_probs > top_p
            sorted_logits[sorted_idx_mask] = -float('inf')

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

class Trainer:
    """Efficient trainer with mixed precision and gradient accumulation."""

    def __init__(self, model: XOR8BitLM, lr=3e-4, warmup_steps=1000,
                 weight_decay=0.1, grad_accum_steps=1, device='cuda',
                 total_steps: int | None = None):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps

        # Optimizer with weight decay on everything except biases and norms
        decay = set()
        no_decay = set()
        for name, param in model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay.add(name)
            else:
                decay.add(name)

        self.opt = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if n in decay],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if n in no_decay],
             'weight_decay': 0.0}
        ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

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

    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step with mixed precision."""
        batch = batch.to(self.device)

        # Prepare inputs and targets
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Mixed precision forward (safe across devices)
        if self.device == 'cuda':
            autocast_ctx = torch.amp.autocast(device_type='cuda')
        elif self.device == 'cpu':
            autocast_ctx = torch.amp.autocast(device_type='cpu')
        else:
            # MPS or other devices: disable autocast by default for stability
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            logits = self.model(inputs)

            # Entropy-weighted loss
            loss = F.cross_entropy(
                logits.reshape(-1, 259),
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

        # Optimizer step
        if (self.step + 1) % self.grad_accum_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

        self.step += 1
        return loss.item() * self.grad_accum_steps

    @property
    def encoder(self):
        return self.model.encoder

# ============= UTILS =============

def save_model(model: XOR8BitLM, path: Union[str, Path], config: Dict[str, Any] = None):
    """Save model and config - updated for RoPE model."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    # Save config
    config = config or {
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': len(model.layers),
        'max_len': model.rope.max_seq_len,
        'rope_base': model.rope.base
    }

    with open(path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save weights
    torch.save(model.state_dict(), path / 'model.pt')

def load_model(path: Union[str, Path], device='cuda') -> XOR8BitLM:
    """Load model from checkpoint."""
    path = Path(path)

    # Load config
    with open(path / 'config.json', 'r') as f:
        config = json.load(f)

    # Create model
    model = XOR8BitLM(**config)

    # Load weights
    model.load_state_dict(torch.load(path / 'model.pt', map_location=device))

    return model.to(device)

# ============= MAIN TRAINING LOOP =============

def train(
    data_path: Union[str, Path],
    model_path: Union[str, Path] = "xor_model",
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    seq_length: int = 512,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    compile_model: bool = False
):
    """Complete training pipeline."""

    print(f"Training on {device}")

    # Create dataset and loader
    dataset = XORDataset(data_path, seq_length)
    loader = DataLoader(dataset, batch_size, shuffle=True,
                       num_workers=1, persistent_workers=True)

    # Create model
    model = XOR8BitLM(d_model, n_heads, n_layers, seq_length)

    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Create trainer with correct OneCycle total steps
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * epochs
    trainer = Trainer(model, lr=lr, device=device, total_steps=total_steps,
                      warmup_steps=min(1000, max(1, total_steps // 10)))

    # Training loop
    for epoch in range(epochs):
        model.train()
        losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            loss = trainer.train_step(batch)
            losses.append(loss)
            pbar.set_postfix({'loss': f"{np.mean(losses[-100:]):.4f}"})

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_model(model, model_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

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
    enc = XOR8BitEncoder()
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

    # Test model

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model("xor_model", device)
    model = model.to(device)

    #print("\nModel architecture:")
    #print(model)

    # Test generation (untrained)
    output = model.generate("The world is a cold place", max_len=200)
    print(f"\nGenerated (untrained): {output}")

    #print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    # Train example (uncomment to run)
    #model = train("data/tiny_shakespeare.txt", batch_size=16, epochs=20)



    #output = model.generate(input_text, max_len=200)
    #print(f"\nGenerated:\n{output}")

    # Or train from HuggingFace
    # model = train_from_hf("wikitext", "wikitext-2-raw-v1", epochs=5)