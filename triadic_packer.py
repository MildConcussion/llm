import os
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from triadictokenizer import TriadicEncoder


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


def _concat_segments_with_bookends(encoder: TriadicEncoder, segs: List[np.ndarray]) -> np.ndarray:
    # [START] + segs + [END]
    parts = [np.array([encoder.START], dtype=np.int64)]
    parts.extend(segs)
    parts.append(np.array([encoder.END], dtype=np.int64))
    return np.concatenate(parts) if len(parts) > 0 else np.array([], dtype=np.int64)


def _encode_text(encoder: TriadicEncoder, text: str) -> np.ndarray:
    toks = encoder.encode(text, add_special=True).cpu().numpy().astype(np.int64, copy=False)
    return toks


def _encode_plain_segment(encoder: TriadicEncoder, text: str) -> np.ndarray:
    # Segment without START/END so we can compose documents deterministically
    toks = encoder.encode(text, add_special=False).cpu().numpy().astype(np.int64, copy=False)
    return toks


def _encode_qwen_message(encoder: TriadicEncoder, role: str, content: str) -> np.ndarray:
    """Encode one Qwen-style message segment with IM_START/IM_END and role+content.
    Layout:
      [IM_START] + role + "\n" + content + [IM_END] + "\n"
    """
    role_seg = _encode_plain_segment(encoder, f"{role}\n")
    content_seg = _encode_plain_segment(encoder, content or "")
    newline_seg = _encode_plain_segment(encoder, "\n")
    parts: List[np.ndarray] = []
    parts.append(np.array([encoder.IM_START], dtype=np.int64))
    if role_seg.size > 0:
        parts.append(role_seg)
    if content_seg.size > 0:
        parts.append(content_seg)
    parts.append(np.array([encoder.IM_END], dtype=np.int64))
    if newline_seg.size > 0:
        parts.append(newline_seg)
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


def _mask_span(total_len: int, start: int, end: int) -> np.ndarray:
    m = np.zeros(total_len, dtype=np.uint8)
    start = max(0, int(start))
    end = max(start, int(end))
    if start < total_len:
        m[start:min(end, total_len)] = 1
    return m


def _choose_dtype_for_tokens(arrays: List[np.ndarray]) -> str:
    max_tok = 0
    for a in arrays:
        if a.size:
            max_tok = max(max_tok, int(np.max(a)))
    return 'uint16' if max_tok < 65536 else 'uint32'


def _flush_shard(
    out_root: Path,
    shard_idx: int,
    token_chunks: List[np.ndarray],
    doc_offsets: List[int],
    doc_lengths: List[int],
    mask_chunks: Optional[List[np.ndarray]] = None,
    fmt: str = 'triadic_docs',
    extra_meta: Optional[dict] = None,
) -> int:
    if not token_chunks:
        return shard_idx
    shard_dir = out_root / f"shard_{shard_idx:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    flat = np.concatenate(token_chunks).astype(np.int64, copy=False)
    has_mask = mask_chunks is not None and len(mask_chunks) > 0
    if has_mask:
        mflat = np.concatenate(mask_chunks).astype(np.uint8, copy=False)
        assert flat.shape[0] == mflat.shape[0]

    dtype = _choose_dtype_for_tokens([flat])
    flat_cast = flat.astype(dtype, copy=False)

    # Write memmaps
    mmap_path = shard_dir / 'tokens.mmap'
    mm = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(flat_cast.shape[0],))
    mm[:] = flat_cast[:]
    mm.flush()
    del mm

    if has_mask:
        mpath = shard_dir / 'loss_mask.mmap'
        mm2 = np.memmap(mpath, dtype='uint8', mode='w+', shape=(mflat.shape[0],))
        mm2[:] = mflat[:]
        mm2.flush()
        del mm2

    np.save(shard_dir / 'offsets.npy', np.asarray(doc_offsets, dtype=np.int64))
    np.save(shard_dir / 'lengths.npy', np.asarray(doc_lengths, dtype=np.int32))

    meta = {
        'num_docs': int(len(doc_lengths)),
        'num_tokens': int(flat_cast.shape[0]),
        'dtype': dtype,
        'format': fmt,
        'has_loss_mask': bool(has_mask),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(shard_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return shard_idx + 1


def pack_local(
    inputs: List[str],
    output_dir: str,
    encoder_path: str,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
) -> None:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    encoder = TriadicEncoder()
    encoder.load(encoder_path)

    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []
    skipped_too_long = 0

    for fp in tqdm(list(_iter_text_files(inputs)), desc='Packing triadic docs'):
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            continue

        toks = _encode_text(encoder, text)
        if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
            skipped_too_long += 1
            continue

        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)
        token_chunks.append(toks)
        doc_lengths.append(int(toks.shape[0]))
        shard_tokens_accum += int(toks.shape[0])

        if shard_tokens_accum >= shard_tokens:
            shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, fmt='triadic_docs')
            shard_tokens_accum = 0
            token_chunks.clear()
            doc_offsets.clear()
            doc_lengths.clear()

    shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, fmt='triadic_docs')

    root_meta = {
        'source': 'local',
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'encoder_path': str(encoder_path),
    }
    with open(out_root / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)
    if skipped_too_long:
        print(f"[triadic/local] skipped {skipped_too_long} docs exceeding max_encoded_len={max_encoded_len}")


def _pack_split_from_iter(
    text_iter,
    output_dir: str,
    encoder: TriadicEncoder,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
) -> dict:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []
    skipped_too_long = 0

    for text in tqdm(text_iter, desc=f"Packing split -> {output_dir}"):
        if text is None:
            continue
        try:
            text_str = str(text)
        except Exception:
            continue
        toks = _encode_text(encoder, text_str)
        if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
            skipped_too_long += 1
            continue

        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)
        token_chunks.append(toks)
        doc_lengths.append(int(toks.shape[0]))
        shard_tokens_accum += int(toks.shape[0])

        if shard_tokens_accum >= shard_tokens:
            shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, fmt='triadic_docs')
            shard_tokens_accum = 0
            token_chunks.clear()
            doc_offsets.clear()
            doc_lengths.clear()

    shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, fmt='triadic_docs')

    split_meta = {
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'skipped_due_to_max_len': int(skipped_too_long),
    }
    with open(Path(output_dir) / 'split_meta.json', 'w') as f:
        json.dump(split_meta, f, indent=2)
    return split_meta


def pack_hf_dataset(
    dataset: str,
    subset: str | None,
    split: str,
    column: str,
    output_dir: str,
    encoder_path: str,
    val_ratio: float = 0.02,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
) -> None:
    print(f"[triadic/hf] loading dataset={dataset} subset={subset} split={split} column={column}")
    from datasets import load_dataset

    ds = load_dataset(dataset, name=subset, split=split)

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

    encoder = TriadicEncoder()
    encoder.load(encoder_path)

    print(f"[triadic/hf] packing train split: {len(train_ds)} rows")

    def _iter_col(dset):
        for ex in dset:
            yield ex.get(column, None)

    train_meta = _pack_split_from_iter(_iter_col(train_ds), str(out_root / 'train'), encoder, shard_tokens=shard_tokens, max_encoded_len=max_encoded_len)

    if val_ds is not None:
        print(f"[triadic/hf] packing val split: {len(val_ds)} rows")
        _pack_split_from_iter(_iter_col(val_ds), str(out_root / 'val'), encoder, shard_tokens=shard_tokens, max_encoded_len=max_encoded_len)

    root_meta = {
        'source': 'hf',
        'dataset': dataset,
        'subset': subset,
        'input_split': split,
        'column': column,
        'val_ratio': val_ratio,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'encoder_path': str(encoder_path),
    }
    with open(Path(output_dir) / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)


def pack_hf_qwen_with_masks(
    dataset: str,
    subset: str | None,
    split: str,
    output_dir: str,
    encoder_path: str,
    task_type: str,
    programming_languages: List[str] | None = None,
    target_audiences: List[str] | None = None,
    shard_tokens: int = 5_000_000,
    max_encoded_len: int | None = None,
    max_docs: int | None = None,
) -> None:
    print(f"[triadic/hf_qwen] loading dataset={dataset} subset={subset} split={split} task={task_type}")
    from datasets import load_dataset
    ds = load_dataset(dataset, name=subset, split=split)

    # Filters for codes if requested
    if task_type == 'single_turn_codes':
        if programming_languages:
            langs = set(programming_languages)
            ds = ds.filter(lambda ex: ex.get('programming_language', None) in langs)
        if target_audiences:
            tas = set(target_audiences)
            ds = ds.filter(lambda ex: ex.get('target_audience', None) in tas)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    encoder = TriadicEncoder()
    encoder.load(encoder_path)

    shard_idx = 0
    shard_tokens_accum = 0
    token_chunks: List[np.ndarray] = []
    mask_chunks: List[np.ndarray] = []
    doc_offsets: List[int] = []
    doc_lengths: List[int] = []
    skipped_too_long = 0
    processed_docs = 0

    def _emit_doc(tokens_arr: np.ndarray, mask_arr: np.ndarray):
        nonlocal shard_tokens_accum, shard_idx, token_chunks, mask_chunks, doc_offsets, doc_lengths
        if shard_tokens_accum == 0:
            doc_offsets.append(0)
        else:
            doc_offsets.append(shard_tokens_accum)
        token_chunks.append(tokens_arr)
        mask_chunks.append(mask_arr)
        doc_lengths.append(int(tokens_arr.shape[0]))
        shard_tokens_accum += int(tokens_arr.shape[0])

    def _maybe_flush():
        nonlocal shard_idx, shard_tokens_accum, token_chunks, mask_chunks, doc_offsets, doc_lengths
        if shard_tokens_accum >= shard_tokens or (max_docs is not None and processed_docs >= max_docs):
            shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, mask_chunks=mask_chunks, fmt='triadic_docs_qwen')
            shard_tokens_accum = 0
            token_chunks.clear()
            mask_chunks.clear()
            doc_offsets.clear()
            doc_lengths.clear()

    for ex in tqdm(ds, desc=f"Packing {task_type} -> {output_dir}"):
        try:
            if task_type == 'pretrain_text':
                text = ex.get('textbook', ex.get('text', ex.get('markdown', None)))
                if text is None:
                    continue
                body = _encode_plain_segment(encoder, str(text))
                toks = _concat_segments_with_bookends(encoder, [body])
                if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                    skipped_too_long += 1
                    continue
                mask = np.ones_like(toks, dtype=np.uint8)
                _emit_doc(toks, mask)
                processed_docs += 1
                _maybe_flush()

            elif task_type in ('single_turn_instruct', 'single_turn_codes'):
                system = ex.get('prompt', None)
                user = ex.get('question', ex.get('prompt', None))
                assistant = ex.get('response', None)
                if assistant is None:
                    continue
                segs = []
                if system:
                    segs.append(_encode_qwen_message(encoder, 'system', str(system)))
                if user:
                    segs.append(_encode_qwen_message(encoder, 'user', str(user)))
                asst_seg = _encode_qwen_message(encoder, 'assistant', str(assistant))
                toks = _concat_segments_with_bookends(encoder, segs + [asst_seg])
                if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                    skipped_too_long += 1
                    continue
                if max_docs is not None and processed_docs >= max_docs:
                    continue
                # Compute mask span for assistant content using encoded segment lengths
                pre_len = sum(s.shape[0] for s in segs) + 1  # +1 for START
                role_len = _encode_plain_segment(encoder, 'assistant\n').shape[0]
                nl_len = _encode_plain_segment(encoder, '\n').shape[0]
                asst_total = asst_seg.shape[0]
                content_len = max(0, asst_total - (1 + role_len + 1 + nl_len))
                asst_start = pre_len + 1 + role_len  # 1 for IM_START
                asst_end = asst_start + content_len
                mask = np.zeros_like(toks, dtype=np.uint8)
                mask |= _mask_span(toks.shape[0], asst_start, asst_end)
                _emit_doc(toks, mask)
                processed_docs += 1
                _maybe_flush()

            elif task_type == 'multi_turn_chat':
                data = ex.get('data', None)
                if not isinstance(data, list) or len(data) < 2:
                    continue
                system_preface = "You are Kulles, created by Rasmus. You are a helpful assistant."
                base_segs = [_encode_qwen_message(encoder, 'system', system_preface)]
                turns = data
                for i in range(1, len(turns), 2):
                    segs = list(base_segs)
                    for j in range(0, i + 1):
                        role = 'user' if (j % 2 == 0) else 'assistant'
                        segs.append(_encode_qwen_message(encoder, role, str(turns[j])))
                    asst_seg = segs[-1]
                    toks = _concat_segments_with_bookends(encoder, segs)
                    if max_encoded_len is not None and int(toks.shape[0]) > int(max_encoded_len):
                        skipped_too_long += 1
                        continue
                    if max_docs is not None and processed_docs >= max_docs:
                        continue
                    pre_len = sum(s.shape[0] for s in segs[:-1]) + 1
                    role_len = _encode_plain_segment(encoder, 'assistant\n').shape[0]
                    nl_len = _encode_plain_segment(encoder, '\n').shape[0]
                    asst_total = asst_seg.shape[0]
                    content_len = max(0, asst_total - (1 + role_len + 1 + nl_len))
                    asst_start = pre_len + 1 + role_len
                    asst_end = asst_start + content_len
                    mask = np.zeros_like(toks, dtype=np.uint8)
                    mask |= _mask_span(toks.shape[0], asst_start, asst_end)
                    _emit_doc(toks, mask)
                    processed_docs += 1
                    _maybe_flush()
            else:
                continue
        except Exception:
            continue

    shard_idx = _flush_shard(out_root, shard_idx, token_chunks, doc_offsets, doc_lengths, mask_chunks=mask_chunks, fmt='triadic_docs_qwen')

    root_meta = {
        'source': 'hf_qwen',
        'dataset': dataset,
        'subset': subset,
        'input_split': split,
        'task_type': task_type,
        'shards': shard_idx,
        'shard_tokens_target': shard_tokens,
        'max_encoded_len': max_encoded_len,
        'encoder_path': str(encoder_path),
    }
    with open(Path(output_dir) / 'meta.json', 'w') as f:
        json.dump(root_meta, f, indent=2)
    if skipped_too_long:
        print(f"[triadic/hf_qwen] skipped {skipped_too_long} documents exceeding max_encoded_len={max_encoded_len}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Pack Triadic-encoded docs into memmap shards')
    sub = ap.add_subparsers(dest='cmd', required=True)

    # Local files
    ap_local = sub.add_parser('local', help='Pack local .txt/.md files')
    ap_local.add_argument('--inputs', nargs='+', required=True, help='Files or directories with .txt/.md')
    ap_local.add_argument('--out', required=True, help='Output directory for shards')
    ap_local.add_argument('--encoder', required=True, help='Path to TriadicEncoder JSON to load')
    ap_local.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_local.add_argument('--max_encoded_len', type=int, default=None, help='Skip docs with encoded length > this')

    # HF dataset
    ap_hf = sub.add_parser('hf', help='Pack a HuggingFace dataset column')
    ap_hf.add_argument('--dataset', required=True, help='e.g., nampdn-ai/tiny-lessons')
    ap_hf.add_argument('--subset', default=None, help='Optional subset name')
    ap_hf.add_argument('--split', default='train', help='Split name to read (default: train)')
    ap_hf.add_argument('--column', required=True, help='Text column name (e.g., textbook)')
    ap_hf.add_argument('--out', required=True, help='Output directory (train/ and val/ created)')
    ap_hf.add_argument('--encoder', required=True, help='Path to TriadicEncoder JSON to load')
    ap_hf.add_argument('--val_ratio', type=float, default=0.02, help='Validation fraction if only one split exists')
    ap_hf.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_hf.add_argument('--max_encoded_len', type=int, default=None, help='Skip docs with encoded length > this')

    # HF Qwen-style with masks
    ap_hfq = sub.add_parser('hf_qwen', help='Pack HF datasets into Qwen-style shards with loss masks')
    ap_hfq.add_argument('--dataset', required=True, help='Dataset name, e.g., nampdn-ai/tiny-orca-textbooks')
    ap_hfq.add_argument('--subset', default=None, help='Optional subset name')
    ap_hfq.add_argument('--split', default='train', help='Split name to read (default: train)')
    ap_hfq.add_argument('--out', required=True, help='Output directory for shards')
    ap_hfq.add_argument('--encoder', required=True, help='Path to TriadicEncoder JSON to load')
    ap_hfq.add_argument('--task_type', required=True, choices=['pretrain_text','single_turn_instruct','single_turn_codes','multi_turn_chat'])
    ap_hfq.add_argument('--programming_languages', nargs='*', default=None, help='Filter for codes: languages to include')
    ap_hfq.add_argument('--target_audiences', nargs='*', default=None, help='Filter for codes: audiences in ascending difficulty order')
    ap_hfq.add_argument('--shard_tokens', type=int, default=5_000_000)
    ap_hfq.add_argument('--max_encoded_len', type=int, default=None)
    ap_hfq.add_argument('--max_docs', type=int, default=None)

    args = ap.parse_args()
    if args.cmd == 'local':
        pack_local(args.inputs, args.out, encoder_path=args.encoder, shard_tokens=args.shard_tokens, max_encoded_len=args.max_encoded_len)
    elif args.cmd == 'hf':
        pack_hf_dataset(
            dataset=args.dataset,
            subset=args.subset,
            split=args.split,
            column=args.column,
            output_dir=args.out,
            encoder_path=args.encoder,
            val_ratio=args.val_ratio,
            shard_tokens=args.shard_tokens,
            max_encoded_len=args.max_encoded_len,
        )
    elif args.cmd == 'hf_qwen':
        pack_hf_qwen_with_masks(
            dataset=args.dataset,
            subset=args.subset,
            split=args.split,
            output_dir=args.out,
            encoder_path=args.encoder,
            task_type=args.task_type,
            programming_languages=args.programming_languages,
            target_audiences=args.target_audiences,
            shard_tokens=args.shard_tokens,
            max_encoded_len=args.max_encoded_len,
            max_docs=args.max_docs,
        )


