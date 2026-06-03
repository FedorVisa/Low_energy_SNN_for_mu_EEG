"""Training-time EEG augmentation utilities for temporal and amplitude perturbations."""

import numpy as np
from torch.utils.data import Dataset


class EEGAugmentedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        copies_per_sample=2,
        reverse_prob=0.5,
        gaussian_std_min=0.01,
        gaussian_std_max=0.05,
        scale_min=0.9,
        scale_max=1.1,
        shift_max=20,
        seed=2023,
    ):
        self.samples = []
        self.copies_per_sample = int(max(0, copies_per_sample))
        self.reverse_prob = float(reverse_prob)
        self.gaussian_std_min = float(gaussian_std_min)
        self.gaussian_std_max = float(gaussian_std_max)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.shift_max = int(max(0, shift_max))
        self.rng = np.random.default_rng(seed)

        for x, y in base_dataset:
            x = np.asarray(x, dtype=np.float32)
            self.samples.append((x, int(y), False))
            for _ in range(self.copies_per_sample):
                self.samples.append((self._augment(x), int(y), True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, _ = self.samples[index]
        return x.astype(np.float32), y

    def _augment(self, x):
        augmented = x.copy()

        if self.rng.random() < self.reverse_prob:
            augmented = augmented[:, ::-1]

        scale = self.rng.uniform(self.scale_min, self.scale_max)
        augmented = augmented * scale

        std = self.rng.uniform(self.gaussian_std_min, self.gaussian_std_max)
        noise = self.rng.normal(0.0, std, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise

        if self.shift_max > 0:
            shift = int(self.rng.integers(-self.shift_max, self.shift_max + 1))
            augmented = np.roll(augmented, shift=shift, axis=-1)

        return augmented.astype(np.float32)