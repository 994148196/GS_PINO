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
    # Coordinate grids and raw scalar parameters for this equilibrium case.
    R = sample["R"]
    Z = sample["Z"]
    params = sample["params"].astype(np.float32)

    # Normalize scalar parameters with training-set statistics when provided.
    p = params if param_mean is None else (params - param_mean) / param_std

    # Unpack the shape parameters needed for coordinate normalization.
    R0, a, kappa = params[0], params[1], params[2]

    # Spatial channels: normalized coordinates plus LCFS-aware geometry features.
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

    # Broadcast each normalized scalar parameter into a constant image channel.
    channels.extend([np.full_like(R, value, dtype=np.float32) for value in p])
    return np.stack(channels, axis=0)


class GSDataset(Dataset):
    """PyTorch dataset backed by a generated GS `.npz` archive."""

    def __init__(self, path: str, indices: np.ndarray | None = None, param_norm: Normalization | None = None):
        # Keep the npz handle open; numpy lazily reads compressed arrays on access.
        raw = np.load(path)
        self.raw = raw

        # Select a subset for train/validation/test, or all samples by default.
        n = raw["params"].shape[0]
        self.indices = np.arange(n) if indices is None else indices

        # Training dataset computes normalization; val/test reuse the same stats.
        params = raw["params"]
        if param_norm is None:
            self.param_norm = Normalization(params.mean(axis=0), params.std(axis=0) + 1e-6)
        else:
            self.param_norm = param_norm

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return len(self.indices)

    def __getitem__(self, item: int):
        """Return `(x, y, mask, sdf, params)` tensors for one sample."""
        # Map local dataset position to the raw archive index.
        i = int(self.indices[item])

        # Gather only fields needed by `build_input` to keep the contract explicit.
        sample = {key: self.raw[key][i] for key in ["R", "Z", "mask", "sdf", "rho", "theta", "params"]}

        # Build model input and target.  The target has a singleton channel axis.
        x = build_input(sample, self.param_norm.mean, self.param_norm.std)
        y = self.raw["psi_bar"][i][None, ...].astype(np.float32)

        # Mask/SDF are returned separately for loss functions and plotting.
        mask = self.raw["mask"][i][None, ...].astype(np.float32)
        sdf = self.raw["sdf"][i][None, ...].astype(np.float32)
        params = self.raw["params"][i].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(sdf), torch.from_numpy(params)


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int = 0):
    """Deterministically split sample indices into train/validation/test groups."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]
