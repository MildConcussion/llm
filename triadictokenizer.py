from datasets import load_dataset
import torch
import numpy as np
from typing import List
import json
from collections import Counter
import time
from dotenv import load_dotenv


import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

try:
    # Fast Rust backend for BPE training
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as _TK_BPE
    from tokenizers.trainers import BpeTrainer as _TK_BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as _TK_ByteLevelPre
    from tokenizers.decoders import ByteLevel as _TK_ByteLevelDec
    _HAS_TOKENIZERS = True
except Exception:
    _HAS_TOKENIZERS = False

# Fast bit operations via lookup tables
_XOR_LUT = np.array([[i ^ j for j in range(256)] for i in range(256)], dtype=np.uint8)
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
_GRAY_LUT = np.array([i ^ (i >> 1) for i in range(256)], dtype=np.uint8)
_GRAY_INV = np.zeros(256, dtype=np.uint8)
for i in range(256):
    _GRAY_INV[_GRAY_LUT[i]] = i

class TriadicEncoder:
    """Unified encoder: Gray code + BPE + Semantic fusion."""

    # Special tokens
    PAD, START, END, NUM, SEP, IM_START, IM_END = 256, 257, 258, 259, 260, 261, 262

    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        # Instance-level special ids to allow remapping after load()
        self.IM_START = TriadicEncoder.IM_START
        self.IM_END = TriadicEncoder.IM_END
        # Start BPE tokens after all specials including IM_START/IM_END (may be raised after load)
        self.bpe_offset = 263
        self.bpe_merges = {}  # (token_id, token_id): new_token_id
        self.bpe_vocab = {}    # token_id: original_bytes (not Gray-coded)
        self.token_freqs = Counter()

    def train_bpe(self, texts: List[str], n_merges: int = None):
        """Train BPE merges.

        Uses HuggingFace tokenizers (Rust) when available for a 10-100x speedup,
        falling back to the Python implementation otherwise.
        """
        if n_merges is None:
            n_merges = self.vocab_size - self.bpe_offset

        # Fast path via HuggingFace tokenizers (Rust)
        if _HAS_TOKENIZERS:
            try:
                t0 = time.time()
                self._train_bpe_fast(texts, n_merges)
                t1 = time.time()
                print(f"[TriadicEncoder][FAST] Imported {n_merges} merges in {t1 - t0:.2f}s")
                return
            except Exception as e:
                print(f"[TriadicEncoder][FAST] Failed with error: {e}. Falling back to Python implementation.")

        # Collect sequences - keep both Gray-coded tokens and original bytes
        sequences = []
        for text in texts:
            if text:
                utf8_bytes = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
                # Create token sequence - initially just the gray-coded bytes
                tokens = _GRAY_LUT[utf8_bytes].tolist()
                sequences.append(tokens)
                self.token_freqs.update(tokens)

        # Initialize vocabulary: Gray code value -> original byte
        for i in range(256):
            gray_val = _GRAY_LUT[i]
            self.bpe_vocab[gray_val] = bytes([i])

        next_token = self.bpe_offset

        # BPE merging loop
        for merge_idx in range(n_merges):
            pair_counts = Counter()

            # Count pairs in all sequences
            for seq in sequences:
                if len(seq) >= 2:
                    for i in range(len(seq) - 1):
                        pair_counts[(seq[i], seq[i+1])] += 1

            if not pair_counts:
                break

            # Most frequent pair
            (a, b), count = pair_counts.most_common(1)[0]
            if count < 2:  # Stop if pairs are too rare
                break

            # Create new token for this pair
            self.bpe_merges[(a, b)] = next_token

            # Determine what bytes this new token represents
            bytes_a = self.bpe_vocab.get(a, bytes([_GRAY_INV[a] if a < 256 else 0]))
            bytes_b = self.bpe_vocab.get(b, bytes([_GRAY_INV[b] if b < 256 else 0]))
            self.bpe_vocab[next_token] = bytes_a + bytes_b

            # Replace pairs in sequences
            new_sequences = []
            for seq in sequences:
                if len(seq) < 2:
                    new_sequences.append(seq)
                    continue

                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i+1] == b:
                        new_seq.append(next_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)

            sequences = new_sequences
            next_token += 1

            if merge_idx % 100 == 0:
                print(f"Merge {merge_idx}: ({a}, {b}) -> {next_token-1}, count={count}")

    def _train_bpe_fast(self, texts: List[str], n_merges: int) -> None:
        """Train BPE merges using HuggingFace tokenizers and import into triadic space.

        - Trains a byte-level BPE on provided texts using the Rust backend
        - Converts learned merges to our Gray-coded token space
        - Populates self.bpe_merges and self.bpe_vocab
        """
        # Reset learned state
        self.bpe_merges = {}
        self.bpe_vocab = {}
        self.token_freqs = Counter()

        # Initialize base vocabulary: Gray-coded single bytes
        for i in range(256):
            gray_val = _GRAY_LUT[i]
            self.bpe_vocab[int(gray_val)] = bytes([i])

        # Configure a byte-level BPE tokenizer
        tokenizer = Tokenizer(_TK_BPE(unk_token=None, byte_fallback=True))
        tokenizer.pre_tokenizer = _TK_ByteLevelPre(add_prefix_space=False)

        # vocab_size = base 256 + desired merges
        target_vocab = 256 + int(n_merges)
        trainer = _TK_BpeTrainer(
            vocab_size=target_vocab,
            min_frequency=2,
            show_progress=True,
            special_tokens=[],
        )

        # Train from iterator of texts
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Extract merges and vocab from the trained model
        tk_cfg = json.loads(tokenizer.to_str())
        model_cfg = tk_cfg.get('model', {})
        merges_list = model_cfg.get('merges', [])
        vocab_dict = model_cfg.get('vocab', {})

        # Decoder to map token strings back to raw bytes
        bytelevel_dec = _TK_ByteLevelDec()

        # Map token string -> underlying bytes
        str_to_bytes = {}
        for s in vocab_dict.keys():
            try:
                decoded = bytelevel_dec.decode([s])
                b = decoded.encode('utf-8', 'ignore')
            except Exception:
                # Fallback: treat as plain string
                b = s.encode('utf-8', 'ignore')
            str_to_bytes[s] = b

        # Map token string -> our triadic token id (Gray or new id)
        tri_token_id = {}
        for s, b in str_to_bytes.items():
            if len(b) == 1:
                tri_token_id[s] = int(_GRAY_LUT[b[0]])

        next_token = self.bpe_offset
        imported = 0

        for merge_idx, merge in enumerate(merges_list):
            if imported >= n_merges:
                break
            if not merge:
                continue
            # Support both string merges ("a b") and list/tuple merges (["a","b"]) formats
            if isinstance(merge, str):
                parts = merge.split(' ')
            elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                parts = merge
            else:
                # Unexpected format; skip
                continue

            if len(parts) != 2:
                continue

            a_str, b_str = str(parts[0]), str(parts[1])

            if merge_idx == 0:
                print(f"[TriadicEncoder][FAST] Detected merges format: {'str' if isinstance(merge, str) else 'list'}")

            # Ensure constituents are known (base or previously merged)
            if a_str not in tri_token_id or b_str not in tri_token_id:
                # The merges list should be topologically ordered, but be defensive
                continue

            a_id = tri_token_id[a_str]
            b_id = tri_token_id[b_str]

            self.bpe_merges[(a_id, b_id)] = next_token

            bytes_a = str_to_bytes.get(a_str, b"")
            bytes_b = str_to_bytes.get(b_str, b"")
            bytes_new = bytes_a + bytes_b

            self.bpe_vocab[next_token] = bytes_new

            # New composed symbol string in tokenizer space is concatenation
            new_sym = a_str + b_str
            tri_token_id[new_sym] = next_token
            str_to_bytes[new_sym] = bytes_new

            imported += 1
            if imported % 200 == 0:
                print(f"[TriadicEncoder][FAST] Merge {imported}: ({a_id}, {b_id}) -> {next_token}")

            next_token += 1

    def encode(self, text: str, add_special: bool = True) -> torch.Tensor:
        """Encode text using triadic scheme."""
        if not text:
            return torch.tensor([self.START, self.END] if add_special else [], dtype=torch.long)

        # Convert to UTF-8 bytes and then to Gray codes
        utf8_bytes = np.frombuffer(text.encode('utf-8', 'ignore'), dtype=np.uint8)
        tokens = _GRAY_LUT[utf8_bytes].tolist()

        # Detect numbers and add NUM prefix
        result = []
        i = 0
        while i < len(tokens):
            # Check if this position starts a number
            orig_byte = utf8_bytes[i]
            if chr(orig_byte).isdigit():
                # Find extent of number
                j = i
                while j < len(utf8_bytes) and (chr(utf8_bytes[j]).isdigit() or chr(utf8_bytes[j]) == '.'):
                    j += 1
                if j > i:
                    result.append(self.NUM)  # Add number marker
                    # Add the number's tokens
                    for k in range(i, j):
                        result.append(tokens[k])
                    i = j
                    continue

            # Try to apply BPE merges
            if i < len(tokens) - 1:
                pair = (tokens[i], tokens[i+1])
                if pair in self.bpe_merges:
                    result.append(self.bpe_merges[pair])
                    i += 2
                    continue

            result.append(tokens[i])
            i += 1

        if add_special:
            result = [self.START] + result + [self.END]

        return torch.tensor(result, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()

        # Filter special tokens
        tokens = tokens[(tokens != self.PAD) & (tokens != self.START) &
                       (tokens != self.END) & (tokens != self.NUM) & (tokens != self.SEP) &
                       (tokens != self.IM_START) & (tokens != self.IM_END)]

        # Reconstruct original bytes
        utf8_bytes = []
        for tok in tokens:
            tok = int(tok)
            if tok < 256:
                # It's a Gray-coded byte
                utf8_bytes.append(_GRAY_INV[tok])
            elif tok in self.bpe_vocab:
                # It's a BPE token - add its bytes
                utf8_bytes.extend(self.bpe_vocab[tok])

        if not utf8_bytes:
            return ""

        return bytes(utf8_bytes).decode('utf-8', 'ignore')

    def encode_qwen_message(self, role: str, content: str) -> torch.Tensor:
        """Encode a Qwen-style message:
        Layout: [IM_START] + role + "\n" + content + [IM_END] + "\n"

        Returns a torch.LongTensor of token ids using this encoder's current specials.
        """
        role_tokens = self.encode(f"{role}\n", add_special=False)
        content_tokens = self.encode(content or "", add_special=False)
        newline_tokens = self.encode("\n", add_special=False)

        parts = [torch.tensor([int(self.IM_START)], dtype=torch.long)]
        if role_tokens.numel() > 0:
            parts.append(role_tokens)
        if content_tokens.numel() > 0:
            parts.append(content_tokens)
        parts.append(torch.tensor([int(self.IM_END)], dtype=torch.long))
        if newline_tokens.numel() > 0:
            parts.append(newline_tokens)
        return torch.cat(parts) if len(parts) > 0 else torch.empty(0, dtype=torch.long)

    def save(self, path: str):
        """Save encoder state."""
        state = {
            'vocab_size': self.vocab_size,
            'bpe_merges': {f"{k[0]}_{k[1]}": v for k, v in self.bpe_merges.items()},
            'bpe_vocab': {str(k): v.hex() for k, v in self.bpe_vocab.items()},
            'token_freqs': dict(self.token_freqs.most_common(10000))  # Keep top 10k
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        """Load encoder state."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.vocab_size = state['vocab_size']
        self.bpe_merges = {tuple(map(int, k.split('_'))): v
                          for k, v in state['bpe_merges'].items()}
        self.bpe_vocab = {int(k): bytes.fromhex(v)
                         for k, v in state['bpe_vocab'].items()}
        self.token_freqs = Counter(state.get('token_freqs', {}))
        # Detect highest used token id to avoid collisions with IM tokens
        used_ids = set()
        for (a, b), c in self.bpe_merges.items():
            used_ids.add(int(a))
            used_ids.add(int(b))
            used_ids.add(int(c))
        used_ids.update(int(k) for k in self.bpe_vocab.keys())
        max_used = max(used_ids) if used_ids else 260
        # If IM_START/IM_END collide with existing ids, remap them above max
        if max_used >= self.IM_START:
            self.IM_START = max_used + 1
            self.IM_END = max_used + 2
            self.bpe_offset = max(self.bpe_offset, self.IM_END + 1)

    def __len__(self):
        return self.vocab_size


# Example usage
if __name__ == "__main__":
    # Create encoder and train BPE
    encoder = TriadicEncoder(vocab_size=4096)
    ds = load_dataset("MildConcussion/smollm-corpus-sample-10000-lt-512", split="train")

    texts = [ex["text"] for ex in ds]

    encoder.train_bpe(texts, n_merges=10000)
    encoder.save("cosmo_encoder.json")

    # Test encoding/decoding
    test = "The value is 256 and 3.14€ öäõü. Testing is great! I love you! Tere tulemast Eestisse!"
    encoded = encoder.encode(test)
    decoded = encoder.decode(encoded)
    print(f"Original: {test}")
    print(f"Encoded: {encoded[:20]}...")  # First 20 tokens
    print(f"Character count: {len(test)}")
    print(f"Token count: {len(encoded)}")
    print(f"Decoded: {decoded}")