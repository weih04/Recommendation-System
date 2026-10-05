
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

UNK_IDX = 0           # rare/unseen move; masked from the output and never a target


class SessionDataset(Dataset):
    """One game becomes one or more fixed-length windows of (context, next move).

    Games run from 3 to 349 plies, so they are cut into overlapping windows of `max_len` and
    right-padded to a uniform length. `stride` controls the overlap: a window starting part-way
    through a game would otherwise begin with no context at all.

    `eval_mask` marks the positions this window is responsible for scoring. Without it the
    overlapping region would be counted twice, and the metrics would be computed over a different
    set of transitions than the Markov baseline sees. Positions whose target is `<unk>` are excluded,
    since no model can be asked to predict an item it was never given.
    """

    def __init__(self, seqs: dict[str, list[int]], max_len: int, stride: int, pad_idx: int):
        self.samples: list[tuple[list[int], list[int], list[bool]]] = []
        self.max_len = max_len
        self.pad_idx = pad_idx

        for seq in seqs.values():
            if len(seq) < 2:
                continue
            for start in range(0, len(seq) - 1, stride):
                end = min(start + max_len + 1, len(seq))
                window = seq[start:end]
                x_seq, y_seq = window[:-1], window[1:]
                length = len(x_seq)
                pad = max_len - length

                x_pad = x_seq + [pad_idx] * pad
                # <unk> targets become padding so the loss ignores them
                y_pad = [pad_idx if tok == UNK_IDX else tok for tok in y_seq] + [pad_idx] * pad

                # score only the positions this window adds, so overlaps are not double-counted
                new_from = 0 if start == 0 else (max_len - stride)
                eval_mask = [(i >= new_from and i < length and y_seq[i] != UNK_IDX)
                             for i in range(max_len)]

                self.samples.append((x_pad, y_pad, eval_mask))
                if end == len(seq):
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_pad, y_pad, mask = self.samples[idx]
        return (torch.tensor(x_pad, dtype=torch.long),
                torch.tensor(y_pad, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool))


class LoaderFactory:
    """Builds and caches the windowed datasets and their loaders.

    Both `max_len` and `batch_size` are tuned, and they cost very different amounts to change:
    a new batch size is a new DataLoader over the same data, while a new window length means
    re-cutting every game. Datasets are therefore cached per `max_len` and reused across the trials
    that share one.

    Memory: a cached dataset at max_len=200 is roughly four times the size of one at 50, so caching
    several window lengths at once is not free. Call `clear(max_len)` to drop one if it bites.
    """

    SPLITS = ("train", "val", "test", "trainval")
    SHUFFLED = ("train", "trainval")

    def __init__(self, seqs_by_split: dict[str, dict[str, list[int]]], pad_idx: int):
        missing = set(self.SPLITS) - set(seqs_by_split)
        if missing:
            raise ValueError(f"missing splits: {sorted(missing)}")
        self.seqs = seqs_by_split
        self.pad_idx = pad_idx
        self._datasets: dict[int, dict[str, SessionDataset]] = {}
        self._loaders: dict[tuple[int, int], dict[str, DataLoader]] = {}

    def datasets(self, max_len: int, stride: int | None = None) -> dict[str, SessionDataset]:
        """Windowed datasets for one window length. Default stride is half the window."""
        if max_len not in self._datasets:
            stride = stride or max_len // 2
            self._datasets[max_len] = {
                name: SessionDataset(self.seqs[name], max_len=max_len, stride=stride,
                                     pad_idx=self.pad_idx)
                for name in self.SPLITS
            }
        return self._datasets[max_len]

    def __call__(self, batch_size: int, max_len: int) -> dict[str, DataLoader]:
        key = (max_len, batch_size)
        if key not in self._loaders:
            self._loaders[key] = {
                name: DataLoader(ds, batch_size=batch_size, shuffle=(name in self.SHUFFLED))
                for name, ds in self.datasets(max_len).items()
            }
        return self._loaders[key]

    def clear(self, max_len: int | None = None) -> None:
        """Drop cached datasets and loaders, for one window length or all of them."""
        if max_len is None:
            self._datasets.clear(); self._loaders.clear()
        else:
            self._datasets.pop(max_len, None)
            for key in [k for k in self._loaders if k[0] == max_len]:
                del self._loaders[key]

    def n_eval_points(self, split: str, max_len: int) -> int:
        """How many transitions this split is scored on — must match what the Markov model sees."""
        return sum(sum(mask) for _, _, mask in self.datasets(max_len)[split].samples)
