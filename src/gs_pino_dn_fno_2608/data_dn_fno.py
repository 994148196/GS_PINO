"""Dataset for the arXiv:2608.05555 double-null FNO surrogate.

Builds the paper's 9-channel input field (Eq. 5 mapping):
    G: (R, Z, Paxis, Ip, fvac, R_lo^X, Z_lo^X, R_up^X, Z_up^X) -> psi(R,Z)
from the .npz files produced by generate_dn_dataset.py.

Normalization (paper Sec. II.C):
  - R, Z: linear map to [-1, 1]
  - Paxis, Ip, fvac: z-scored with TRAIN-set mean/std, broadcast to the grid
  - the 4 X-point coordinates: likewise z-scored with train stats, broadcast
  - target psi: z-scored with TRAIN-set mean/std
Training loss and relative L2 error are computed in this normalized
representation; physical quantities use the inverse transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# paper domain
RMIN, RMAX = 0.1, 2.0
ZMIN, ZMAX = -2.0, 2.0

CHANNEL_NAMES = ["R", "Z", "Paxis", "Ip", "fvac", "R_lo^X", "Z_lo^X", "R_up^X", "Z_up^X"]


def compute_stats(npz: dict, n: int | None = None) -> dict[str, np.ndarray]:
    """Train-set mean/std of the 3 run params, 4 X-point coords and target psi.

    n optionally limits to the first n samples (scaling study: paper computes
    stats from the full train pool; keep n=None for that).
    """
    params = npz["params"][:n]
    xc = npz["x_coords"][:n]
    psi = npz["psi_total"][:n]

    scalar_mean = np.concatenate([params.mean(axis=0), xc.mean(axis=0)]).astype(np.float32)
    scalar_std = np.concatenate([params.std(axis=0), xc.std(axis=0)]).astype(np.float32)
    psi_mean = float(psi.mean())
    psi_std = float(psi.std())
    return {
        "scalar_mean": scalar_mean, "scalar_std": scalar_std,
        "psi_mean": np.float32(psi_mean), "psi_std": np.float32(psi_std),
        "n_train": int(n if n is not None else len(psi)),
    }


def _normalize_rz(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear map R in [0.1, 2.0] and Z in [-2, 2] to [-1, 1]."""
    Rn = 2.0 * (R - RMIN) / (RMAX - RMIN) - 1.0
    Zn = 2.0 * (Z - ZMIN) / (ZMAX - ZMIN) - 1.0
    return Rn.astype(np.float32), Zn.astype(np.float32)


class DNFnoDataset(Dataset):
    """9-channel field dataset; inputs are assembled on the fly per sample."""

    def __init__(self, npz_path: str | Path, stats: dict | None = None,
                 indices: np.ndarray | None = None):
        self.npz_path = str(npz_path)
        with np.load(npz_path) as d:
            self.R, self.Z = _normalize_rz(d["R"], d["Z"])
            self.psi_total = d["psi_total"].astype(np.float32)      # (N, 65, 65)
            self.params = d["params"].astype(np.float32)            # (N, 3): Ip, paxis, fvac
            self.x_coords = d["x_coords"].astype(np.float32)        # (N, 4)
            self.mask = d["mask"].astype(np.float32)
            self.dpdpsi = d["dpdpsi"].astype(np.float32)
            self.FdFdpsi = d["FdFdpsi"].astype(np.float32)
            self.axes = d["axes"].astype(np.float32)                # R_axis, Z_axis, psi_bndry, psi_axis
            self.n_full = len(self.psi_total)

        if stats is None:  # compute stats from the full dataset (train use)
            stats = compute_stats({
                "params": self.params, "x_coords": self.x_coords, "psi_total": self.psi_total})
        self.stats = stats

        if indices is None:
            indices = np.arange(self.n_full)
        self.indices = np.asarray(indices, dtype=np.int64)

        self.scalar_mean = self.stats["scalar_mean"]
        self.scalar_std = self.stats["scalar_std"]
        self.psi_mean = float(self.stats["psi_mean"])
        self.psi_std = float(self.stats["psi_std"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        scalars = np.concatenate([self.params[i], self.x_coords[i]])  # (7,)
        scalars = (scalars - self.scalar_mean) / self.scalar_std
        # broadcast the 7 scalar channels to the grid
        scalar_fields = np.broadcast_to(scalars[:, None, None], (7,) + self.R.shape)

        x = np.concatenate([self.R[None], self.Z[None], scalar_fields], axis=0)  # (9, 65, 65)
        y = (self.psi_total[i] - self.psi_mean) / self.psi_std                   # (65, 65)
        return torch.from_numpy(x.copy()), torch.from_numpy(y[None])


def rel_l2_normalized(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error in the normalized representation (paper Eq. 7)."""
    num = torch.norm(pred - target, p=2, dim=(1, 2, 3))
    den = torch.norm(target, p=2, dim=(1, 2, 3))
    return (num / (den + 1e-12)).mean()


def nested_train_indices(n_full: int, n_train: int, perm_seed: int = 12345) -> np.ndarray:
    """Fixed permutation (paper: seed 12345); nested subsets are the first n indices."""
    rng = np.random.default_rng(perm_seed)
    return rng.permutation(n_full)[:n_train]
