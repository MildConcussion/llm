from pathlib import Path
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

        # Load metadata and maps
        self.tokens = np.memmap(self.shard_dir / 'tokens.mmap', dtype='uint16', mode='r')
        self.offsets = np.load(self.shard_dir / 'offsets.npy', mmap_mode='r')
        self.lengths = np.load(self.shard_dir / 'lengths.npy', mmap_mode='r')

        # Build window index: global -> (doc_id, start)
        self._build_window_index()

    def _build_window_index(self):
        seq_len = self.seq_length
        stride = self.stride
        doc_windows = []
        doc_starts = []

        for doc_id, L in enumerate(self.lengths.tolist()):
            if L <= seq_len:
                doc_windows.append(1)
                doc_starts.append([0])
                continue

            starts = list(range(0, max(L - seq_len + 1, 1), stride))
            # Ensure last window covers tail (may violate exact stride at the end)
            last_start = max(L - seq_len, 0)
            if starts[-1] != last_start:
                starts.append(last_start)

            doc_windows.append(len(starts))
            doc_starts.append(starts)

        total_windows = int(sum(doc_windows))
        self.window_doc_ids = np.empty(total_windows, dtype=np.int32)
        self.window_starts = np.empty(total_windows, dtype=np.int64)

        cursor = 0
        for doc_id, starts in enumerate(doc_starts):
            n = len(starts)
            self.window_doc_ids[cursor:cursor+n] = doc_id
            self.window_starts[cursor:cursor+n] = np.asarray(starts, dtype=np.int64)
            cursor += n

    def __len__(self) -> int:
        return int(self.window_doc_ids.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
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

        if end - start < seq_len:
            out = np.full(seq_len, self.pad_id, dtype=np.int64)
            out[: end - start] = segment
        else:
            out = segment

        tensor = torch.from_numpy(out)
        if self.device is not None:
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor


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


