"""Dataset and input-channel construction for the masked U-FNO surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import PARAM_NAMES


@dataclass
class Normalization:
    """Mean/std pair used to normalize the eight scalar input parameters."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Normalize an array with broadcasting-compatible mean and std."""
        return (x - self.mean) / self.std


def build_input(sample: dict[str, np.ndarray], param_mean: np.ndarray | None = None, param_std: np.ndarray | None = None) -> np.ndarray:
    """Convert one raw `.npz` sample into U-FNO input channels.

    The output is channel-first `[C, nr, nz]`.  Geometry channels teach the model
    where the non-rectangular LCFS sits inside the rectangular grid, and scalar
    parameters are normalized then broadcast so every pixel knows the equilibrium
    setting it belongs to.
    """
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

    channels.extend([np.full_like(R, value, dtype=np.float32) for value in p])
    return np.stack(channels, axis=0)


class GSDataset(Dataset):
    """PyTorch dataset backed by a generated GS `.npz` archive."""

    def __init__(self, path: str, indices: np.ndarray | None = None, param_norm: Normalization | None = None):
        raw = np.load(path)
        self.R = raw["R"]
        self.Z = raw["Z"]
        self.mask = raw["mask"]
        self.sdf = raw["sdf"]
        self.rho = raw["rho"]
        self.theta = raw["theta"]
        self.params = raw["params"]
        self.psi_bar = raw["psi_bar"]
        self.profile_params = raw.get("profile_params", np.zeros((raw["params"].shape[0], 2), dtype=np.float32))
        self.axes = raw.get("axes", np.zeros((raw["params"].shape[0], 4), dtype=np.float32))

        n = self.params.shape[0]
        self.indices = np.arange(n) if indices is None else indices

        if param_norm is None:
            self.param_norm = Normalization(self.params.mean(axis=0), self.params.std(axis=0) + 1e-6)
        else:
            self.param_norm = param_norm

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return len(self.indices)

    def __getitem__(self, item: int):
        """Return `(x, y, mask, sdf, params, metadata)` tensors for one sample.

        metadata is a dict containing per-sample PDE information:
          - R, Z : 2-D coordinate grids
          - profile_params : [L, Beta0]
          - psi_axis, psi_lcfs : boundary values for psiN conversion
          - alpha_m, alpha_n : from scalar parameters
        """
        i = int(self.indices[item])

        sample = {key: getattr(self, key)[i] for key in ["R", "Z", "mask", "sdf", "rho", "theta", "params"]}

        x = build_input(sample, self.param_norm.mean, self.param_norm.std)
        y = self.psi_bar[i][None, ...].astype(np.float32)

        mask = self.mask[i][None, ...].astype(np.float32)
        sdf = self.sdf[i][None, ...].astype(np.float32)
        params = self.params[i].astype(np.float32)

        # Per-sample metadata for PDE loss and integral constraints
        metadata = {
            "R": torch.from_numpy(self.R[i].astype(np.float32)),
            "Z": torch.from_numpy(self.Z[i].astype(np.float32)),
            "profile_params": torch.from_numpy(self.profile_params[i].astype(np.float32)),
            "R0": float(params[0]),
            "alpha_m": float(params[6]),
            "alpha_n": float(params[7]),
            "psi_axis": float(self.axes[i, 3]),
            "psi_lcfs": float(self.axes[i, 2]),
            "R_axis": float(self.axes[i, 0]),
            "Z_axis": float(self.axes[i, 1]),
        }

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(sdf), torch.from_numpy(params), metadata


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int = 0):
    """Deterministically split sample indices into train/validation/test groups."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
