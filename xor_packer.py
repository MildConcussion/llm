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
def _encode_qwen_message(encoder: GrayCodeEncoder, role: str, content: str) -> np.ndarray:
    """Encode one Qwen-style message segment with IM_START/IM_END and role+content.
    Layout:
      [IM_START] + role + "\n" + content + [IM_END] + "\n"
    Roles are included as raw utf-8 bytes.
    """
    role_bytes = (role + "\n").encode('utf-8', errors='ignore')
    content_bytes = (content or "").encode('utf-8', errors='ignore')

    parts: List[np.ndarray] = []
    parts.append(np.array([encoder.IM_START], dtype=np.uint16))
    if role_bytes:
        rb = np.frombuffer(role_bytes, dtype=np.uint8)
        parts.append(encoder.gray_lut[rb].astype(np.uint16, copy=False))
    if content_bytes is not None:
        parts.append(encoder.gray_lut[np.frombuffer(content_bytes, dtype=np.uint8)].astype(np.uint16, copy=False))
    parts.append(np.array([encoder.IM_END], dtype=np.uint16))
    # trailing newline after IM_END
    parts.append(encoder.gray_lut[np.frombuffer(b"\n", dtype=np.uint8)].astype(np.uint16, copy=False))
    return np.concatenate(parts)

def _concat_segments_with_bookends(encoder: GrayCodeEncoder, segs: List[np.ndarray]) -> np.ndarray:
    # [START] + segs + [EOS]
    parts = [np.array([encoder.START], dtype=np.uint16)]
    parts.extend(segs)
    parts.append(np.array([encoder.EOS], dtype=np.uint16))
    return np.concatenate(parts)

def _mask_span(total_len: int, start: int, end: int) -> np.ndarray:
    m = np.zeros(total_len, dtype=np.uint8)
    start = max(0, int(start))
    end = max(start, int(end))
    if start < total_len:
        m[start:min(end, total_len)] = 1
    return m



def pack_to_memmap(
    input_paths: List[str],
    output_dir: str,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
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
    skipped_too_long = 0

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

        if max_encoded_len is not None and int(encoded.shape[0]) > int(max_encoded_len):
            skipped_too_long += 1
            continue

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
        'max_encoded_len': max_encoded_len,
        'schema': {
            'tokens': 'uint16 memmap flat',
            'offsets': 'int64 per-doc',
            'lengths': 'int32 per-doc',
        },
        'skipped_due_to_max_len': int(skipped_too_long),
    }
    with open(out_root / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)
    if skipped_too_long:
        print(f"[pack] skipped {skipped_too_long} documents exceeding max_encoded_len={max_encoded_len}")


def _pack_split_from_iter(text_iter, output_dir: str, shard_tokens: int = 5_000_000, max_encoded_len: int | None = None) -> None:
    """Pack an iterable of raw text documents into memmap shard format."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    encoder = GrayCodeEncoder()
    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []
    skipped_too_long = 0

    def _flush_shard():
        nonlocal shard_idx, shard_tokens_accum, token_chunks, doc_offsets, doc_lengths
        if not token_chunks:
            return
        shard_dir = out_root / f"shard_{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        flat = np.concatenate(token_chunks).astype(np.uint16, copy=False)
        mmap_path = shard_dir / 'tokens.mmap'
        mm = np.memmap(mmap_path, dtype='uint16', mode='w+', shape=(flat.shape[0],))
        mm[:] = flat[:]
        mm.flush()
        del mm

        np.save(shard_dir / 'offsets.npy', np.asarray(doc_offsets, dtype=np.int64))
        np.save(shard_dir / 'lengths.npy', np.asarray(doc_lengths, dtype=np.int32))

        meta = {
            'num_docs': int(len(doc_lengths)),
            'num_tokens': int(flat.shape[0]),
            'dtype': 'uint16',
            'format': 'xor_gray_docs',
            'has_loss_mask': False,
        }
        with open(shard_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        shard_idx += 1
        shard_tokens_accum = 0
        token_chunks.clear()
        doc_offsets.clear()
        doc_lengths.clear()

    for text in tqdm(text_iter, desc=f"Packing split -> {output_dir}"):
        if text is None:
            continue
        try:
            if isinstance(text, (list, tuple)):
                text_bytes = "\n\n".join([str(t) for t in text]).encode('utf-8', errors='ignore')
            else:
                text_bytes = str(text).encode('utf-8', errors='ignore')
        except Exception:
            continue

        encoded = _encode_document_bytes(encoder, text_bytes)

        if max_encoded_len is not None and int(encoded.shape[0]) > int(max_encoded_len):
            skipped_too_long += 1
            continue

        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)

        token_chunks.append(encoded)
        doc_lengths.append(int(encoded.shape[0]))
        shard_tokens_accum += int(encoded.shape[0])

        if shard_tokens_accum >= shard_tokens:
            _flush_shard()

    _flush_shard()
    # Write split-level meta
    split_meta = {
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'skipped_due_to_max_len': int(skipped_too_long),
    }
    with open(Path(output_dir) / 'split_meta.json', 'w') as f:
        json.dump(split_meta, f, indent=2)
    if skipped_too_long:
        print(f"[pack] split {output_dir}: skipped {skipped_too_long} documents exceeding max_encoded_len={max_encoded_len}")


def pack_hf_dataset(
    dataset: str,
    subset: str | None,
    split: str,
    column: str,
    output_dir: str,
    val_ratio: float = 0.02,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
    filter_col: str | None = None,
    filter_vals: List[str] | None = None,
) -> None:
    """
    Pack a HuggingFace dataset's text column into memmapped shards.

    - dataset: e.g. 'nampdn-ai/tiny-lessons'
    - subset: optional subset name, or None
    - split: input split to read (e.g., 'train')
    - column: text column to use (e.g., 'textbook')
    - output_dir: root output directory; will create subdirs 'train' and 'val'
    - val_ratio: fraction to allocate to validation if no explicit val split exists
    - shard_tokens: target tokens per shard
    """
    print(f"[hf] loading dataset={dataset} subset={subset} split={split} column={column}")
    from datasets import load_dataset

    ds = load_dataset(dataset, name=subset, split=split)

    # Optional source filtering
    before_n = len(ds)
    if filter_col and filter_vals:
        if filter_col in ds.column_names:
            allowed = set(filter_vals)
            print(f"[hf] filtering on column '{filter_col}' in {dataset}: keeping values in {sorted(list(allowed))}")
            ds = ds.filter(lambda ex: ex.get(filter_col, None) in allowed)
            after_n = len(ds)
            print(f"[hf] filtered rows: kept {after_n} / {before_n}")
        else:
            print(f"[hf] warning: column '{filter_col}' not found; skipping filter.")

    # If there is only a train split, make a deterministic train/val split
    if val_ratio and val_ratio > 0:
        parts = ds.train_test_split(test_size=val_ratio, seed=42)
        train_ds = parts['train']
        val_ds = parts['test']
    else:
        train_ds = ds
        val_ds = None

    out_root = Path(output_dir)
    (out_root / 'train').mkdir(parents=True, exist_ok=True)
    if val_ds is not None:
        (out_root / 'val').mkdir(parents=True, exist_ok=True)

    print(f"[hf] packing train split: {len(train_ds)} rows")
    def _iter_col(dset):
        for ex in dset:
            yield ex.get(column, None)

    _pack_split_from_iter(_iter_col(train_ds), str(out_root / 'train'), shard_tokens=shard_tokens, max_encoded_len=max_encoded_len)

    if val_ds is not None:
        print(f"[hf] packing val split: {len(val_ds)} rows")
        _pack_split_from_iter(_iter_col(val_ds), str(out_root / 'val'), shard_tokens=shard_tokens, max_encoded_len=max_encoded_len)

    # Root meta
    root_meta = {
        'source': 'hf',
        'dataset': dataset,
        'subset': subset,
        'input_split': split,
        'column': column,
        'val_ratio': val_ratio,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'filter_col': filter_col,
        'filter_vals': filter_vals,
    }
    with open(Path(output_dir) / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)


def pack_hf_qwen_with_masks(
    dataset: str,
    subset: str | None,
    split: str,
    output_dir: str,
    task_type: str,
    # filters for codes
    programming_languages: List[str] | None = None,
    target_audiences: List[str] | None = None,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
    max_docs: int | None = None,
) -> None:
    """
    Pack HF datasets into shards with tokens.mmap and loss_mask.mmap using Qwen-style messages.

    task_type:
      - 'pretrain_text' for single text column named 'textbook' (mask all tokens)
      - 'single_turn_instruct' for columns {prompt?, question, response}
      - 'single_turn_codes' for columns {prompt?, response} with filters
      - 'multi_turn_chat' for column 'data' = list[str] alternating user/assistant; emits multiple documents per row
      - 'tiny_stories_instruct' for TinyStoriesInstruct dataset (groups rows into stories, creates instruction-response pairs)
    """
    print(f"[hf] loading dataset={dataset} subset={subset} split={split} task={task_type}")
    from datasets import load_dataset
    ds = load_dataset(dataset, name=subset, split=split)

    # Special handling for tiny_stories_instruct: group rows into complete stories
    if task_type == 'tiny_stories_instruct':
        stories = []
        current_story = []
        for i, ex in enumerate(ds):
            text = ex['text']
            current_story.append(text)
            if '<|endoftext|>' in text:
                stories.append(current_story)
                current_story = []
                if len(stories) >= 100000:
                    break
        print(f"[tiny_stories] grouped {len(stories)} complete stories from {len(ds)} rows")
        ds = stories  # Replace dataset with grouped stories

    # Apply filters for codes if requested
    elif task_type == 'single_turn_codes':
        if programming_languages:
            langs = set(programming_languages)
            ds = ds.filter(lambda ex: ex.get('programming_language', None) in langs)
        if target_audiences:
            tas = set(target_audiences)
            ds = ds.filter(lambda ex: ex.get('target_audience', None) in tas)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    encoder = GrayCodeEncoder()

    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []
    skipped_too_long = 0
    processed_docs = 0

    def _flush_shard():
        nonlocal shard_idx, shard_tokens_accum, token_chunks, mask_chunks, doc_offsets, doc_lengths
        if not token_chunks:
            return
        shard_dir = out_root / f"shard_{shard_idx:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        flat = np.concatenate(token_chunks).astype(np.uint16, copy=False)
        mflat = np.concatenate(mask_chunks).astype(np.uint8, copy=False)
        assert flat.shape[0] == mflat.shape[0]

        mmap_path = shard_dir / 'tokens.mmap'
        mm = np.memmap(mmap_path, dtype='uint16', mode='w+', shape=(flat.shape[0],))
        mm[:] = flat[:]
        mm.flush()
        del mm

        mpath = shard_dir / 'loss_mask.mmap'
        mm2 = np.memmap(mpath, dtype='uint8', mode='w+', shape=(mflat.shape[0],))
        mm2[:] = mflat[:]
        mm2.flush()
        del mm2

        np.save(shard_dir / 'offsets.npy', np.asarray(doc_offsets, dtype=np.int64))
        np.save(shard_dir / 'lengths.npy', np.asarray(doc_lengths, dtype=np.int32))

        meta = {
            'num_docs': int(len(doc_lengths)),
            'num_tokens': int(flat.shape[0]),
            'dtype': 'uint16',
            'format': 'xor_gray_docs_qwen',
            'has_loss_mask': True,
            'task_type': task_type,
            'max_encoded_len': max_encoded_len,
        }
        with open(shard_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        shard_idx += 1
        shard_tokens_accum = 0
        token_chunks.clear()
        mask_chunks.clear()
        doc_offsets.clear()
        doc_lengths.clear()

    def _emit_doc(tokens_arr: np.ndarray, mask_arr: np.ndarray):
        nonlocal shard_tokens_accum
        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)
        token_chunks.append(tokens_arr)
        mask_chunks.append(mask_arr)
        doc_lengths.append(int(tokens_arr.shape[0]))
        shard_tokens_accum += int(tokens_arr.shape[0])
        processed_docs += 1

    def _maybe_flush():
        if shard_tokens_accum >= shard_tokens or (max_docs is not None and processed_docs >= max_docs):
            _flush_shard()

    for ex in tqdm(ds, desc=f"Packing {task_type} -> {output_dir}"):
        try:
            if task_type == 'pretrain_text':
                text = ex.get('textbook', None)

                if text is None:
                    text = ex.get('text', None)

                if text is None:
                    text = ex.get('markdown', None)

                if text is None:
                    continue
                body = np.frombuffer(str(text).encode('utf-8', errors='ignore'), dtype=np.uint8)
                seg = encoder.gray_lut[body].astype(np.uint16, copy=False)
                toks = _concat_segments_with_bookends(encoder, [seg])
                if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                    skipped_too_long += 1
                    continue
                if max_docs is not None and processed_docs >= max_docs:
                    continue
                mask = np.ones_like(toks, dtype=np.uint8)
                # Keep START/EOS contributing to mask as well (or set to 0 if preferred)
                _emit_doc(toks, mask)
                _maybe_flush()

            elif task_type in ('single_turn_instruct', 'single_turn_codes'):
                system = ex.get('prompt', None)  # optional system
                user = ex.get('question', ex.get('prompt', None))
                assistant = ex.get('response', None)
                if assistant is None:
                    continue
                segs = []
                if system:
                    segs.append(_encode_qwen_message(encoder, 'system', str(system)))
                if user:
                    segs.append(_encode_qwen_message(encoder, 'user', str(user)))
                # assistant
                pre_len = sum(s.shape[0] for s in segs) + 1  # +1 for START
                asst_seg = _encode_qwen_message(encoder, 'assistant', str(assistant))
                # assistant content span within asst_seg: it includes role+"\n" then content bytes then IM_END and trailing \n
                toks = _concat_segments_with_bookends(encoder, segs + [asst_seg])
                if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                    skipped_too_long += 1
                    continue
                if max_docs is not None and processed_docs >= max_docs:
                    continue
                mask = np.zeros_like(toks, dtype=np.uint8)
                # Build mask over assistant content: find boundaries inside asst_seg
                # role prefix length: len('assistant\n') in gray-coded bytes = len(bytes)*1
                role_prefix_len = len(b'assistant\n')
                # Map raw bytes -> gray length 1:1
                asst_start = pre_len + 1 + role_prefix_len  # after IM_START + role\n
                # asst_seg layout: [IM_START] role+\n content_bytes [IM_END] \n
                asst_total = asst_seg.shape[0]
                # Subtract [IM_END] (1) and trailing \n (1) from tail
                content_len = max(0, asst_total - (1 + 1) - role_prefix_len)
                asst_end = asst_start + content_len
                mask |= _mask_span(toks.shape[0], asst_start, asst_end)
                _emit_doc(toks, mask)
                _maybe_flush()

            elif task_type == 'multi_turn_chat':
                data = ex.get('data', None)
                if not isinstance(data, list) or len(data) < 2:
                    continue
                # Optional system preface from requirement
                system_preface = "You are Kulles, created by Rasmus. You are a helpful assistant."
                base_segs = [_encode_qwen_message(encoder, 'system', system_preface)]
                # Build cumulative conversation
                turns = data
                # Generate a document per assistant turn
                for i in range(1, len(turns), 2):  # i is assistant turn index
                    segs = list(base_segs)
                    # add user/assistant pairs up to i
                    for j in range(0, i+1):
                        role = 'user' if (j % 2 == 0) else 'assistant'
                        segs.append(_encode_qwen_message(encoder, role, str(turns[j])))
                    # Compute assistant i mask span
                    pre_len = sum(s.shape[0] for s in segs[:-1]) + 1  # +1 for START
                    asst_seg = segs[-1]
                    role_prefix_len = len(b'assistant\n')
                    toks = _concat_segments_with_bookends(encoder, segs)
                    if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                        skipped_too_long += 1
                        continue
                    if max_docs is not None and processed_docs >= max_docs:
                        continue
                    mask = np.zeros_like(toks, dtype=np.uint8)
                    asst_total = asst_seg.shape[0]
                    asst_start = pre_len + 1 + role_prefix_len
                    content_len = max(0, asst_total - (1 + 1) - role_prefix_len)
                    asst_end = asst_start + content_len
                    mask |= _mask_span(toks.shape[0], asst_start, asst_end)
                    _emit_doc(toks, mask)
                    _maybe_flush()

            elif task_type == 'tiny_stories_instruct':
                # ex is now a list of strings representing one complete story
                story_lines = ex

                # Parse the story components
                features = None
                words = None
                summary = None
                story_content = []

                i = 0
                while i < len(story_lines):
                    line = story_lines[i]
                    if line.startswith('Features: '):
                        features = line[len('Features: '):]
                    elif line.startswith('Words: '):
                        words = line[len('Words: '):]
                    elif line.startswith('Summary: '):
                        summary = line[len('Summary: '):]
                    elif line == 'Story: ':
                        # Story content starts after this
                        i += 2  # Skip the empty line after "Story: "
                        while i < len(story_lines) and not story_lines[i].startswith('<|endoftext|>'):
                            if story_lines[i]:  # Skip empty lines
                                story_content.append(story_lines[i])
                            i += 1
                        break
                    i += 1

                if not all([features, words, summary]) or not story_content:
                    continue  # Skip malformed stories

                # Create instruction
                instruction = f"Write a story with {features} that includes the words {words} and has this summary: {summary}"

                # Join story content
                response = '\n'.join(story_content)

                # Create Qwen-style message
                segs = [
                    _encode_qwen_message(encoder, 'user', instruction)
                ]
                pre_len = sum(s.shape[0] for s in segs) + 1  # +1 for START
                asst_seg = _encode_qwen_message(encoder, 'assistant', response)
                toks = _concat_segments_with_bookends(encoder, segs + [asst_seg])

                if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                    skipped_too_long += 1
                    continue

                mask = np.zeros_like(toks, dtype=np.uint8)
                # Build mask over assistant content: find boundaries inside asst_seg
                role_prefix_len = len(b'assistant\n')
                asst_start = pre_len + 1 + role_prefix_len  # after IM_START + role\n
                asst_total = asst_seg.shape[0]
                content_len = max(0, asst_total - (1 + 1) - role_prefix_len)
                asst_end = asst_start + content_len
                mask |= _mask_span(toks.shape[0], asst_start, asst_end)
                _emit_doc(toks, mask)
                _maybe_flush()

            else:
                continue
        except Exception:
            continue

    _flush_shard()

    # Root meta for hf_qwen packing
    root_meta = {
        'source': 'hf_qwen',
        'dataset': dataset,
        'subset': subset,
        'input_split': split,
        'task_type': task_type,
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'skipped_due_to_max_len': int(skipped_too_long),
    }
    with open(Path(output_dir) / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)
    if skipped_too_long:
        print(f"[hf_qwen] skipped {skipped_too_long} documents exceeding max_encoded_len={max_encoded_len}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Pack XOR Gray-coded docs into memmap shards')
    sub = ap.add_subparsers(dest='cmd', required=True)

    # Local files
    ap_local = sub.add_parser('local', help='Pack local .txt/.md files')
    ap_local.add_argument('--inputs', nargs='+', required=True, help='Files or directories with .txt/.md')
    ap_local.add_argument('--out', required=True, help='Output directory for shards')
    ap_local.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_local.add_argument('--max_encoded_len', type=int, default=None, help='Skip docs with encoded length > this')

    # HF dataset
    ap_hf = sub.add_parser('hf', help='Pack a HuggingFace dataset column')
    ap_hf.add_argument('--dataset', required=True, help='e.g., nampdn-ai/tiny-lessons')
    ap_hf.add_argument('--subset', default=None, help='Optional subset name')
    ap_hf.add_argument('--split', default='train', help='Split name to read (default: train)')
    ap_hf.add_argument('--column', required=True, help='Text column name (e.g., textbook)')
    ap_hf.add_argument('--out', required=True, help='Output directory (train/ and val/ created)')
    ap_hf.add_argument('--val_ratio', type=float, default=0.02, help='Validation fraction if only one split exists')
    ap_hf.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_hf.add_argument('--max_encoded_len', type=int, default=None, help='Skip docs with encoded length > this')
    ap_hf.add_argument('--filter_col', default=None, help='Optional column name to filter on (e.g., source)')
    ap_hf.add_argument('--filter_vals', nargs='*', default=None, help='Allowed values for filter_col (space separated)')

    # HF Qwen-style with masks
    ap_hfq = sub.add_parser('hf_qwen', help='Pack HF datasets into Qwen-style shards with loss masks')
    ap_hfq.add_argument('--dataset', required=True, help='Dataset name, e.g., nampdn-ai/tiny-orca-textbooks')
    ap_hfq.add_argument('--subset', default=None, help='Optional subset name')
    ap_hfq.add_argument('--split', default='train', help='Split name to read (default: train)')
    ap_hfq.add_argument('--out', required=True, help='Output directory for shards')
    ap_hfq.add_argument('--task_type', required=True, choices=['pretrain_text','single_turn_instruct','single_turn_codes','multi_turn_chat','tiny_stories_instruct'])
    ap_hfq.add_argument('--programming_languages', nargs='*', default=None, help='Filter for codes: languages to include')
    ap_hfq.add_argument('--target_audiences', nargs='*', default=None, help='Filter for codes: audiences in ascending difficulty order')
    ap_hfq.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_hfq.add_argument('--max_encoded_len', type=int, default=None)
    ap_hfq.add_argument('--max_docs', type=int, default=None)
    args = ap.parse_args()
    if args.cmd == 'local':
        pack_to_memmap(args.inputs, args.out, shard_tokens=args.shard_tokens, max_encoded_len=args.max_encoded_len)
    elif args.cmd == 'hf':
        pack_hf_dataset(
            dataset=args.dataset,
            subset=args.subset,
            split=args.split,
            column=args.column,
            output_dir=args.out,
            val_ratio=args.val_ratio,
            shard_tokens=args.shard_tokens,
            max_encoded_len=args.max_encoded_len,
            filter_col=args.filter_col,
            filter_vals=args.filter_vals,
        )
    elif args.cmd == 'hf_qwen':
        pack_hf_qwen_with_masks(
            dataset=args.dataset,
            subset=args.subset,
            split=args.split,
            output_dir=args.out,
            task_type=args.task_type,
            programming_languages=args.programming_languages,
            target_audiences=args.target_audiences,
            shard_tokens=args.shard_tokens,
            max_encoded_len=args.max_encoded_len,
            max_docs=args.max_docs,
        )


