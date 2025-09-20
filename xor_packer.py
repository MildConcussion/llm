import os
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

# Local import; relies on repo layout
from innovation4 import GrayCodeEncoder


def _iter_text_files(paths: Iterable[str]) -> Iterable[Path]:
    exts = {'.txt', '.md'}
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for root, _, files in os.walk(pth):
                for fn in files:
                    if Path(fn).suffix.lower() in exts:
                        yield Path(root) / fn
        elif pth.is_file() and pth.suffix.lower() in exts:
            yield pth


def _encode_document_bytes(encoder: GrayCodeEncoder, text_bytes: bytes) -> np.ndarray:
    # START + gray(bytes) + EOS
    if not text_bytes:
        return np.array([encoder.START, encoder.EOS], dtype=np.uint16)

    byte_arr = np.frombuffer(text_bytes, dtype=np.uint8)
    gray = encoder.gray_lut[byte_arr]
    out = np.empty(gray.shape[0] + 2, dtype=np.uint16)
    out[0] = encoder.START
    out[1:-1] = gray.astype(np.uint16, copy=False)
    out[-1] = encoder.EOS
    return out


def pack_to_memmap(
    input_paths: List[str],
    output_dir: str,
    shard_tokens: int = 5_000_000,
) -> None:
    """
    Pack many documents into memmapped shards:
    - tokens.mmap (uint16, flat)
    - offsets.npy (int64, per-document start offset)
    - lengths.npy (int32, per-document length)

    Shards are created such that total tokens per shard <= shard_tokens.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    encoder = GrayCodeEncoder()

    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []

    def _flush_shard():
        nonlocal shard_idx, shard_tokens_accum, token_chunks, doc_offsets, doc_lengths
        if not token_chunks:
            return
        shard_dir = out_root / f"shard_{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        flat = np.concatenate(token_chunks).astype(np.uint16, copy=False)
        # Write memmap
        mmap_path = shard_dir / 'tokens.mmap'
        mm = np.memmap(mmap_path, dtype='uint16', mode='w+', shape=(flat.shape[0],))
        mm[:] = flat[:]
        mm.flush()
        del mm

        # Offsets/lengths
        np.save(shard_dir / 'offsets.npy', np.asarray(doc_offsets, dtype=np.int64))
        np.save(shard_dir / 'lengths.npy', np.asarray(doc_lengths, dtype=np.int32))

        meta = {
            'num_docs': int(len(doc_lengths)),
            'num_tokens': int(flat.shape[0]),
            'dtype': 'uint16',
            'format': 'xor_gray_docs',
        }
        with open(shard_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        shard_idx += 1
        shard_tokens_accum = 0
        token_chunks.clear()
        doc_offsets.clear()
        doc_lengths.clear()

    # Iterate documents
    total_docs = 0
    for fp in tqdm(list(_iter_text_files(input_paths)), desc='Packing docs'):
        try:
            # Read as bytes; tolerate encoding issues
            with open(fp, 'rb') as f:
                raw = f.read()
        except Exception:
            continue

        encoded = _encode_document_bytes(encoder, raw)

        # Start a new shard if needed
        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)

        token_chunks.append(encoded)
        doc_lengths.append(int(encoded.shape[0]))
        shard_tokens_accum += int(encoded.shape[0])
        total_docs += 1

        if shard_tokens_accum >= shard_tokens:
            _flush_shard()

    _flush_shard()

    # Root meta
    root_meta = {
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'schema': {
            'tokens': 'uint16 memmap flat',
            'offsets': 'int64 per-doc',
            'lengths': 'int32 per-doc',
        }
    }
    with open(out_root / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Pack XOR Gray-coded docs into memmap shards')
    ap.add_argument('--inputs', nargs='+', required=True, help='Files or directories with .txt/.md')
    ap.add_argument('--out', required=True, help='Output directory for shards')
    ap.add_argument('--shard_tokens', type=int, default=5_000_000)
    args = ap.parse_args()
    pack_to_memmap(args.inputs, args.out, shard_tokens=args.shard_tokens)


