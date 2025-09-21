from pathlib import Path
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedXORShardDataset(Dataset):
    """
    Memory-mapped shard of Gray-coded XOR tokens with strict doc-boundary windowing.

    Exposes fixed-length windows with stride within each document. Windows never cross
    document boundaries. Short final windows are padded with PAD token id.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_length: int,
        stride: Optional[int] = None,
        pad_id: int = 258,
        device: Optional[torch.device] = None,
    ):
        self.shard_dir = Path(shard_dir)
        self.seq_length = int(seq_length)
        self.stride = int(stride) if stride is not None else int(seq_length // 2)
        self.pad_id = int(pad_id)
        self.device = device

        # Store paths and lazy-open memmaps to minimize open FDs
        self._tokens_path = self.shard_dir / 'tokens.mmap'
        self._loss_mask_path = self.shard_dir / 'loss_mask.mmap'
        self.tokens: Optional[np.memmap] = None
        self.loss_mask: Optional[np.memmap] = None

        # Load small index arrays fully into RAM to avoid mmap FDs
        self.offsets = np.asarray(np.load(self.shard_dir / 'offsets.npy'))
        self.lengths = np.asarray(np.load(self.shard_dir / 'lengths.npy'))

        # Build window index: global -> (doc_id, start)
        self._build_window_index()

    def _ensure_open(self):
        """Lazily open memmaps for tokens and optional loss mask."""
        if self.tokens is None:
            self.tokens = np.memmap(self._tokens_path, dtype='uint16', mode='r')
        if self.loss_mask is None and self._loss_mask_path.exists():
            self.loss_mask = np.memmap(self._loss_mask_path, dtype='uint8', mode='r')

    def _close_memmaps(self):
        """Best-effort close of underlying mmap objects to release FDs."""
        try:
            if isinstance(self.tokens, np.memmap) and getattr(self.tokens, "_mmap", None) is not None:
                self.tokens._mmap.close()
        except Exception:
            pass
        finally:
            self.tokens = None
        try:
            if isinstance(self.loss_mask, np.memmap) and getattr(self.loss_mask, "_mmap", None) is not None:
                self.loss_mask._mmap.close()
        except Exception:
            pass
        finally:
            self.loss_mask = None

    def __del__(self):
        # Ensure file descriptors are released promptly
        self._close_memmaps()

    def close(self):
        """Explicitly close memmaps."""
        self._close_memmaps()

    # Use as context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        # Avoid pickling open memmaps; workers will lazy-open after spawn
        state = self.__dict__.copy()
        state['tokens'] = None
        state['loss_mask'] = None
        return state

    def _build_window_index(self):
        # Current has loops - vectorize instead
        seq_len = self.seq_length
        stride = self.stride

        # Vectorized computation of windows per doc
        doc_lengths = self.lengths
        n_windows = np.where(
            doc_lengths <= seq_len,
            1,
            ((np.maximum(doc_lengths - seq_len, 0) + stride - 1) // stride) + 1
        )

        total_windows = n_windows.sum()
        self.window_doc_ids = np.repeat(np.arange(len(doc_lengths)), n_windows)

        # Vectorized start positions
        window_starts = []
        for doc_id, (L, nw) in enumerate(zip(doc_lengths, n_windows)):
            if nw == 1:
                window_starts.append([0])
            else:
                starts = np.arange(0, L - seq_len + 1, stride)
                if starts[-1] != L - seq_len:
                    starts = np.append(starts, L - seq_len)
                window_starts.append(starts)

        self.window_starts = np.concatenate(window_starts)

    def __len__(self) -> int:
        return int(self.window_doc_ids.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Lazy-open memmaps on first access
        self._ensure_open()
        doc_id = int(self.window_doc_ids[idx])
        start = int(self.window_starts[idx])
        offset = int(self.offsets[doc_id])
        length = int(self.lengths[doc_id])
        seq_len = self.seq_length

        # Slice within document
        end = min(start + seq_len, length)
        flat_start = offset + start
        flat_end = offset + end
        segment = self.tokens[flat_start:flat_end].astype(np.int64, copy=False)
        mask_seg = None
        if self.loss_mask is not None:
            mask_seg = self.loss_mask[flat_start:flat_end].astype(np.uint8, copy=False)

        if end - start < seq_len:
            out = np.full(seq_len, self.pad_id, dtype=np.int64)
            out[: end - start] = segment
            if mask_seg is not None:
                mask_out = np.zeros(seq_len, dtype=np.uint8)
                mask_out[: end - start] = mask_seg
        else:
            out = segment
            if mask_seg is not None:
                mask_out = mask_seg

        tensor = torch.from_numpy(out)
        if self.device is not None:
            tensor = tensor.to(self.device, non_blocking=True)
        if mask_seg is None:
            return tensor
        mask_tensor = torch.from_numpy(mask_out.astype(np.float32))
        if self.device is not None:
            mask_tensor = mask_tensor.to(self.device, non_blocking=True)
        return tensor, mask_tensor


class MixedPackedXORDataset(Dataset):
    """
    Combine multiple shard datasets with configurable mixing:
    - policy='round_robin': equalize across datasets in cyclic order until shortest exhausts.
    - policy='weighted': approximate proportion using weights; length equals sum of lengths
      of all datasets (with replacement sampling).
    """

    def __init__(
        self,
        datasets: List[PackedXORShardDataset],
        policy: str = 'round_robin',
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        assert len(datasets) > 0
        self.datasets = datasets
        self.policy = policy
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights is not None:
            self.weights = self.weights / self.weights.sum()
        self.rng = np.random.RandomState(seed)

        self.index: List[Tuple[int, int]] = []  # (ds_idx, local_idx)
        self._rebuild_index()

    def _rebuild_index(self):
        if self.policy == 'round_robin':
            # Cycle through datasets until the shortest exhausts
            lens = [len(ds) for ds in self.datasets]
            min_len = min(lens)
            order = []
            for i in range(min_len):
                for ds_idx, ds_len in enumerate(lens):
                    order.append((ds_idx, i))
            self.index = order
        elif self.policy == 'weighted':
            lens = [len(ds) for ds in self.datasets]
            total = int(sum(lens))
            # Sample dataset indices with replacement according to weights
            if self.weights is None:
                w = np.asarray(lens, dtype=np.float64)
                w = w / w.sum()
            else:
                w = self.weights
            ds_choices = self.rng.choice(len(self.datasets), size=total, p=w)
            counters = [0] * len(self.datasets)
            self.index = []
            for ds_idx in ds_choices:
                local_idx = counters[ds_idx] % len(self.datasets[ds_idx])
                self.index.append((int(ds_idx), int(local_idx)))
                counters[ds_idx] += 1
        else:
            raise ValueError(f"Unknown mixing policy: {self.policy}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds_idx, local_idx = self.index[idx]
        return self.datasets[ds_idx][local_idx]

    def set_epoch(self, epoch: int):
        # Optionally reshuffle for weighted policy
        if self.policy == 'weighted':
            self.rng.seed(epoch)
            self._rebuild_index()


def discover_shards(root: str) -> List[str]:
    """Return list of shard directories under root."""
    rootp = Path(root)
    if not rootp.exists():
        return []
    shards = []
    for p in sorted(rootp.iterdir()):
        if p.is_dir() and (p / 'tokens.mmap').exists():
            shards.append(str(p))
    return shards



# ========= SPLIT HELPERS =========

def _normalize_dataset_root(root: Path) -> Path:
    """If a path points to .../train or .../val, return its parent dataset root."""
    root = Path(root)
    name = root.name.lower()
    if name in ("train", "val"):
        return root.parent
    return root


def ensure_train_val_split(
    dataset_root: str | Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    materialize: str = 'move',
) -> tuple[Path, Path]:
    """Ensure dataset root has disjoint train/ and val/ shard folders.

    Behavior:
    - If root contains shard dirs directly -> create train/ and val/ and move a subset into val.
    - If root/train exists and root/val missing/empty -> move a subset of train shards into val.
    - If both exist with shards -> leave as-is.

    Returns: (train_dir, val_dir)
    """
    rng = np.random.RandomState(int(seed))
    root = _normalize_dataset_root(Path(dataset_root))
    train_dir = root / 'train'
    val_dir = root / 'val'

    # Case A: shards directly under root (no split yet)
    direct_shards = discover_shards(str(root))
    if direct_shards and not train_dir.exists() and not val_dir.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [Path(s) for s in direct_shards]
        shard_names = [p.name for p in shard_paths]
        order = np.argsort(shard_names)
        shard_paths = [shard_paths[i] for i in order]
        n = len(shard_paths)
        if n >= 2:
            n_val = max(1, int(round(n * float(val_ratio))))
            n_val = min(n - 1, n_val)  # keep at least 1 in train
        else:
            n_val = 0
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = set(idx[:n_val].tolist())
        # Move into split dirs
        moved_val = 0
        for i, p in enumerate(shard_paths):
            dest_parent = val_dir if i in val_idx else train_dir
            dest = dest_parent / p.name
            p.rename(dest)
            if i in val_idx:
                moved_val += 1
        # Fallback: if only one shard, perform doc-level split into train/val
        if moved_val == 0 and n == 1:
            original = train_dir / shard_paths[0].name
            _doclevel_split_single_shard(original, train_dir, val_dir, val_ratio=val_ratio, rng=rng)
            # original shard replaced by new split
            moved_val = 1
        print(f"[split] created {train_dir} and {val_dir}; shards train={n - moved_val}, val={moved_val}")
        return train_dir, val_dir

    # Case B: train/ exists (maybe val/ not yet)
    if train_dir.exists():
        train_shards = discover_shards(str(train_dir))
        val_dir.mkdir(parents=True, exist_ok=True)
        val_shards = discover_shards(str(val_dir))
        if len(val_shards) == 0 and len(train_shards) >= 2:
            # Derive val from train by moving a subset
            shard_paths = [Path(s) for s in train_shards]
            shard_names = [p.name for p in shard_paths]
            order = np.argsort(shard_names)
            shard_paths = [shard_paths[i] for i in order]
            n = len(shard_paths)
            n_val = max(1, int(round(n * float(val_ratio))))
            n_val = min(n - 1, n_val)
            idx = np.arange(n)
            rng.shuffle(idx)
            chosen = set(idx[:n_val].tolist())
            moved_val = 0
            for i, p in enumerate(shard_paths):
                if i in chosen:
                    dest = val_dir / p.name
                    p.rename(dest)
                    moved_val += 1
            print(f"[split] populated {val_dir} from {train_dir}; shards train={n - moved_val}, val={moved_val}")
        elif len(val_shards) == 0 and len(train_shards) == 1:
            # Fallback: single shard -> doc-level split into train/val
            original = Path(train_shards[0])
            _doclevel_split_single_shard(original, train_dir, val_dir, val_ratio=val_ratio, rng=rng)
        else:
            print(f"[split] using existing split under {root}; train_shards={len(train_shards)}, val_shards={len(val_shards)}")
        return train_dir, val_dir

    # Case C: both train/ and val/ missing but no shards directly -> nothing to do
    # Create empty structure to avoid surprises
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"[split] initialized empty split dirs at {train_dir} and {val_dir}")
    return train_dir, val_dir


def build_mixed_dataset(
    dataset_roots: List[str | Path],
    split: str,
    seq_length: int,
    pad_id: int,
    policy: str = 'round_robin',
    weights: Optional[List[float]] = None,
) -> Dataset:
    """Build a dataset by mixing shards across multiple dataset roots for a given split.

    Each root is expected to have subdir `split` containing shard dirs. Shards within a root
    are concatenated; roots are mixed with the specified policy.
    """
    assert split in ('train', 'val')
    per_root_datasets: List[Dataset] = []
    total_shards = 0
    for r in dataset_roots:
        root = _normalize_dataset_root(Path(r))
        split_dir = root / split
        shard_dirs = discover_shards(str(split_dir))
        if not shard_dirs:
            continue
        shard_datasets = [
            PackedXORShardDataset(sd, seq_length=seq_length, stride=seq_length // 2, pad_id=pad_id)
            for sd in shard_dirs
        ]
        total_shards += len(shard_datasets)
        if len(shard_datasets) == 1:
            per_root_datasets.append(shard_datasets[0])
        else:
            per_root_datasets.append(torch.utils.data.ConcatDataset(shard_datasets))

    if not per_root_datasets:
        raise ValueError(f"No shards found for split='{split}' under roots={dataset_roots}")

    print(f"[dataset] split={split} roots={len(dataset_roots)} total_shards={total_shards} policy={policy}")

    if len(per_root_datasets) == 1:
        return per_root_datasets[0]

    return MixedPackedXORDataset(per_root_datasets, policy=policy, weights=weights)


def _estimate_windows_for_lengths(lengths: np.ndarray, seq_length: int, stride: int) -> int:
    """Estimate number of fixed windows for given doc lengths without IO-heavy loads.

    Mirrors PackedXORShardDataset logic but counts only.
    """
    seq_len = int(seq_length)
    stride = int(stride)
    L = lengths.astype(np.int64)
    # diffs = max(L - seq_len, 0)
    diffs = np.maximum(L - seq_len, 0)
    # ceil(diffs / stride) = (diffs + stride - 1) // stride
    extra = (diffs + stride - 1) // max(stride, 1)
    per_doc = 1 + extra
    return int(per_doc.sum())


def _shard_window_count(shard_dir: str | Path, seq_length: int, stride: int) -> int:
    # Load into RAM to avoid holding a memmap FD during selection
    lengths = np.asarray(np.load(Path(shard_dir) / 'lengths.npy'))
    return _estimate_windows_for_lengths(lengths, seq_length, stride)


def build_capped_mixed_dataset(
    dataset_roots: List[str | Path],
    split: str,
    seq_length: int,
    pad_id: int,
    windows_needed: int,
    policy: str = 'round_robin',
    weights: Optional[List[float]] = None,
) -> Dataset:
    """Build a dataset mixing shards across roots but only selecting enough shards
    to cover approximately `windows_needed` training windows.

    Selection strategy: deterministic round-robin over roots picking next shard by
    lexicographic order until the cumulative window count meets/exceeds the target.
    Falls back to all shards if target cannot be met.
    """
    assert split in ('train', 'val')
    stride = max(1, int(seq_length // 2))

    roots: List[Path] = [Path(r) for r in dataset_roots]
    roots = [_normalize_dataset_root(r) for r in roots]

    per_root_all_shards: List[List[str]] = []
    for r in roots:
        split_dir = r / split
        shard_dirs = discover_shards(str(split_dir))
        shard_dirs = sorted(shard_dirs)
        per_root_all_shards.append(shard_dirs)

    if sum(len(s) for s in per_root_all_shards) == 0:
        raise ValueError(f"No shards found for split='{split}' under roots={dataset_roots}")

    selected_per_root: List[List[str]] = [[] for _ in roots]
    cursors = [0 for _ in roots]
    total_windows = 0

    # Greedy round-robin selection
    made_progress = True
    while total_windows < int(windows_needed) and made_progress:
        made_progress = False
        for ridx, shard_list in enumerate(per_root_all_shards):
            if total_windows >= int(windows_needed):
                break
            if cursors[ridx] >= len(shard_list):
                continue
            shard = shard_list[cursors[ridx]]
            cursors[ridx] += 1
            try:
                wc = _shard_window_count(shard, seq_length=seq_length, stride=stride)
            except Exception:
                wc = 0
            selected_per_root[ridx].append(shard)
            total_windows += wc
            made_progress = True

    # If we couldn't meet the target, proceed with what we have (all selected)
    print(f"[dataset-capped] split={split} roots={len(roots)} target_windows={windows_needed} selected_windows~{total_windows} policy={policy}")

    per_root_datasets: List[Dataset] = []
    total_selected_shards = 0
    for shard_dirs in selected_per_root:
        if not shard_dirs:
            continue
        shard_datasets = [
            PackedXORShardDataset(sd, seq_length=seq_length, stride=stride, pad_id=pad_id)
            for sd in shard_dirs
        ]
        total_selected_shards += len(shard_datasets)
        if len(shard_datasets) == 1:
            per_root_datasets.append(shard_datasets[0])
        else:
            per_root_datasets.append(torch.utils.data.ConcatDataset(shard_datasets))

    if not per_root_datasets:
        raise ValueError(f"No shards selected for split='{split}' under roots={dataset_roots}")

    if len(per_root_datasets) == 1:
        return per_root_datasets[0]

    return MixedPackedXORDataset(per_root_datasets, policy=policy, weights=weights)


def _doclevel_split_single_shard(shard_path: Path, train_dir: Path, val_dir: Path, val_ratio: float, rng: np.random.RandomState) -> None:
    """Split a single shard directory into train/val shards at document level without overlap."""
    # Load original arrays
    tokens = np.memmap(shard_path / 'tokens.mmap', dtype='uint16', mode='r')
    offsets = np.load(shard_path / 'offsets.npy', mmap_mode='r')
    lengths = np.load(shard_path / 'lengths.npy', mmap_mode='r')
    has_mask = (shard_path / 'loss_mask.mmap').exists()
    loss_mask = np.memmap(shard_path / 'loss_mask.mmap', dtype='uint8', mode='r') if has_mask else None

    n_docs = int(len(lengths))
    if n_docs <= 1:
        # Degenerate: move shard entirely to train; keep val empty
        dest = train_dir / shard_path.name
        if dest.exists():
            # Clean stale
            for p in dest.iterdir():
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                except Exception:
                    pass
        shard_path.rename(dest)
        print(f"[split-doc] only {n_docs} doc(s); kept in train: {dest}")
        return

    # Choose docs for val
    idx = np.arange(n_docs)
    rng.shuffle(idx)
    n_val = max(1, int(round(n_docs * float(val_ratio))))
    n_val = min(n_docs - 1, n_val)
    val_set = set(idx[:n_val].tolist())

    # Build gather lists
    train_tokens = []
    val_tokens = []
    train_offsets = []
    val_offsets = []
    train_lengths = []
    val_lengths = []
    train_mask = [] if has_mask else None
    val_mask = [] if has_mask else None

    cur_off_train = 0
    cur_off_val = 0

    for doc_id, L in enumerate(lengths.tolist()):
        start = int(offsets[doc_id])
        end = start + int(L)
        seg = tokens[start:end].astype(np.uint16, copy=False)
        if doc_id in val_set:
            val_tokens.append(seg)
            val_offsets.append(cur_off_val)
            val_lengths.append(int(L))
            if has_mask:
                val_mask.append(loss_mask[start:end].astype(np.uint8, copy=False))
            cur_off_val += int(L)
        else:
            train_tokens.append(seg)
            train_offsets.append(cur_off_train)
            train_lengths.append(int(L))
            if has_mask:
                train_mask.append(loss_mask[start:end].astype(np.uint8, copy=False))
            cur_off_train += int(L)

    # Remove original shard
    backup = shard_path.parent / (shard_path.name + "_orig_backup")
    try:
        shard_path.rename(backup)
    except Exception:
        backup = None

    # Write train shard
    train_sub = train_dir / shard_path.name
    train_sub.mkdir(parents=True, exist_ok=True)
    flat_train = np.concatenate(train_tokens).astype(np.uint16, copy=False) if train_tokens else np.array([], dtype=np.uint16)
    mm = np.memmap(train_sub / 'tokens.mmap', dtype='uint16', mode='w+', shape=(flat_train.shape[0],))
    mm[:] = flat_train[:]
    mm.flush()
    del mm
    np.save(train_sub / 'offsets.npy', np.asarray(train_offsets, dtype=np.int64))
    np.save(train_sub / 'lengths.npy', np.asarray(train_lengths, dtype=np.int32))
    if has_mask and train_mask:
        mflat = np.concatenate(train_mask).astype(np.uint8, copy=False)
        mm2 = np.memmap(train_sub / 'loss_mask.mmap', dtype='uint8', mode='w+', shape=(mflat.shape[0],))
        mm2[:] = mflat[:]
        mm2.flush()
        del mm2
    meta = {
        'num_docs': int(len(train_lengths)),
        'num_tokens': int(flat_train.shape[0]),
        'dtype': 'uint16',
        'format': 'xor_gray_docs' + ("_qwen" if has_mask else ""),
        'has_loss_mask': bool(has_mask),
    }
    with open(train_sub / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Write val shard
    val_sub = val_dir / shard_path.name
    val_sub.mkdir(parents=True, exist_ok=True)
    flat_val = np.concatenate(val_tokens).astype(np.uint16, copy=False) if val_tokens else np.array([], dtype=np.uint16)
    mm = np.memmap(val_sub / 'tokens.mmap', dtype='uint16', mode='w+', shape=(flat_val.shape[0],))
    mm[:] = flat_val[:]
    mm.flush()
    del mm
    np.save(val_sub / 'offsets.npy', np.asarray(val_offsets, dtype=np.int64))
    np.save(val_sub / 'lengths.npy', np.asarray(val_lengths, dtype=np.int32))
    if has_mask and val_mask:
        mflat = np.concatenate(val_mask).astype(np.uint8, copy=False)
        mm2 = np.memmap(val_sub / 'loss_mask.mmap', dtype='uint8', mode='w+', shape=(mflat.shape[0],))
        mm2[:] = mflat[:]
        mm2.flush()
        del mm2
    meta['num_docs'] = int(len(val_lengths))
    meta['num_tokens'] = int(flat_val.shape[0])
    with open(val_sub / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    if backup is not None:
        # Clean backup to save space
        try:
            for p in backup.iterdir():
                if p.is_file() or p.is_symlink():
                    p.unlink()
            backup.rmdir()
        except Exception:
            pass

