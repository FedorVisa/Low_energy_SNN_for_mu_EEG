"""Dataset wrappers for loading preprocessed EEG numpy arrays into PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from torch.utils.data import Dataset


class EEGset(Dataset):
    """Loads subject-level EEG arrays saved by the preprocessing scripts."""

    def __init__(
        self,
        root_path,
        pick_id: Sequence[int] = (1,),
        settup: str = "train",
        T: int = 125,
        EA: bool = False,
        loo: bool = False,
        all_id: Iterable[int] = range(1, 10),
        zscore: bool = False,
        zscore_eps: float = 1e-6,
    ):
        self.T = int(T)
        self.zscore = bool(zscore)
        self.zscore_eps = float(zscore_eps)

        split_dir = Path(root_path) / settup
        if loo:
            if EA:
                split_dir = split_dir / "EA"
            subject_ids = pick_id if settup == "test" else sorted(set(all_id) - set(pick_id))
        else:
            subject_ids = pick_id

        self.data_info = self._read_subjects(split_dir, subject_ids)

    def __getitem__(self, index):
        eeg, label = self.data_info[index]
        if self.zscore:
            eeg = self._zscore(eeg)
        return eeg, label

    def __len__(self):
        return len(self.data_info)

    def _trim_time(self, eeg):
        if eeg.shape[1] == self.T:
            return eeg.astype(np.float32)
        return eeg[:, : self.T].astype(np.float32)

    def _zscore(self, eeg):
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True)
        return (eeg - mean) / (std + self.zscore_eps)

    def _read_subjects(self, split_dir: Path, subject_ids: Iterable[int]):
        samples = []
        for subject_id in subject_ids:
            data_file = split_dir / f"data_id{subject_id}.npy"
            label_file = split_dir / f"label_id{subject_id}.npy"
            eeg_trials = np.load(data_file, allow_pickle=True)
            labels = np.load(label_file, allow_pickle=True)

            for eeg, label in zip(eeg_trials, labels):
                samples.append((self._trim_time(eeg), label))
        return samples
