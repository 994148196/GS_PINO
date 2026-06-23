from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import PARAM_NAMES


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def build_input(sample: dict[str, np.ndarray], param_mean: np.ndarray | None = None, param_std: np.ndarray | None = None) -> np.ndarray:
    R = sample["R"]
    Z = sample["Z"]
    params = sample["params"].astype(np.float32)
    p = params if param_mean is None else (params - param_mean) / param_std
    R0, a, kappa = params[0], params[1], params[2]
    channels = [
        ((R - R0) / a).astype(np.float32),
        (Z / a).astype(np.float32),
        (Z / (kappa * a)).astype(np.float32),
        sample["mask"].astype(np.float32),
        sample["sdf"].astype(np.float32),
        sample["rho"].astype(np.float32),
        np.sin(sample["theta"]).astype(np.float32),
        np.cos(sample["theta"]).astype(np.float32),
    ]
    channels.extend([np.full_like(R, v, dtype=np.float32) for v in p])
    return np.stack(channels, axis=0)


class GSDataset(Dataset):
    def __init__(self, path: str, indices: np.ndarray | None = None, param_norm: Normalization | None = None):
        raw = np.load(path)
        self.raw = raw
        n = raw["params"].shape[0]
        self.indices = np.arange(n) if indices is None else indices
        params = raw["params"]
        if param_norm is None:
            self.param_norm = Normalization(params.mean(axis=0), params.std(axis=0) + 1e-6)
        else:
            self.param_norm = param_norm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        i = int(self.indices[item])
        sample = {k: self.raw[k][i] for k in ["R", "Z", "mask", "sdf", "rho", "theta", "params"]}
        x = build_input(sample, self.param_norm.mean, self.param_norm.std)
        y = self.raw["psi_bar"][i][None, ...].astype(np.float32)
        mask = self.raw["mask"][i][None, ...].astype(np.float32)
        sdf = self.raw["sdf"][i][None, ...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(sdf), torch.from_numpy(self.raw["params"][i].astype(np.float32))


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
