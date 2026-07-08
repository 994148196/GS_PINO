"""Dataset and input-channel construction for the free-boundary GS PINO surrogate.

Following PlaNet-equil architecture:
- Input: measures (scalar parameters), R (grid coordinates), Z (grid coordinates)
- Output: psi_total (total poloidal flux) over entire computational domain

The network predicts psi_total directly (not decomposed into plasma + coil contributions).

Note: freegs requires grid size to be 2^n + 1 (e.g., 65 = 2^6 + 1). We interpolate to 64x64
for efficient FFT operations and mixed precision training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-7)


def interp_fun(f: np.ndarray, RR: np.ndarray, ZZ: np.ndarray, rr: np.ndarray, zz: np.ndarray) -> np.ndarray:
    x_pts = RR[:, 0].ravel()
    y_pts = ZZ[0, :].ravel()
    interp_func = RegularGridInterpolator((x_pts, y_pts), f.T)
    f_int = interp_func(
        np.column_stack((rr.reshape(-1, 1), zz.reshape(-1, 1))),
        method="quintic",
    ).reshape(rr.shape)
    return f_int


class FreeBndDataset(Dataset):
    """PyTorch dataset backed by a free-boundary GS .npz archive."""

    def __init__(self, path: str, indices: np.ndarray | None = None, param_norm: Normalization | None = None, nr: int = 64, nz: int = 64):
        raw = np.load(path)
        self.base_R = raw["R"]
        self.base_Z = raw["Z"]
        self.base_psi_total = raw["psi_total"]
        self.base_psi_plasma = raw["psi_plasma"]
        self.base_psi_coils = raw["psi_coils"]
        self.base_mask = raw["mask"]
        self.base_rhs = raw.get("rhs", None)
        self.coil_currents = raw["coil_currents"]
        self.params = raw["params"]
        self.axes = raw.get("axes", np.zeros((raw["params"].shape[0], 4), dtype=np.float32))
        self.L = raw.get("L", np.zeros(raw["params"].shape[0], dtype=np.float32))
        self.Beta0 = raw.get("Beta0", np.zeros(raw["params"].shape[0], dtype=np.float32))

        self.nr = nr
        self.nz = nz

        n = self.params.shape[0]
        self.indices = np.arange(n) if indices is None else indices

        all_params = np.concatenate([self.coil_currents, self.params], axis=1)
        if param_norm is None:
            self.param_norm = Normalization(all_params.mean(axis=0), all_params.std(axis=0))
        else:
            self.param_norm = param_norm

        self.n_measures = all_params.shape[1]

        self._preload_data()

    def _preload_data(self):
        n = len(self.indices)
        self.R_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.Z_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.psi_total_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.psi_plasma_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.psi_coils_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.mask_data = np.zeros((n, self.nr, self.nz), dtype=np.float32)
        self.rhs_data = np.zeros((n, self.nr - 2, self.nz - 2), dtype=np.float32)

        for idx in range(n):
            i = int(self.indices[idx])
            base_R = self.base_R[i]
            base_Z = self.base_Z[i]

            if base_R.shape[0] != self.nr or base_R.shape[1] != self.nz:
                rr = np.linspace(base_R[:, 0].min(), base_R[:, 0].max(), self.nr)
                zz = np.linspace(base_Z[0, :].min(), base_Z[0, :].max(), self.nz)
                R, Z = np.meshgrid(rr, zz, indexing='ij')

                self.psi_total_data[idx] = interp_fun(f=self.base_psi_total[i], RR=base_R, ZZ=base_Z, rr=R, zz=Z)
                self.psi_plasma_data[idx] = interp_fun(f=self.base_psi_plasma[i], RR=base_R, ZZ=base_Z, rr=R, zz=Z)
                self.psi_coils_data[idx] = interp_fun(f=self.base_psi_coils[i], RR=base_R, ZZ=base_Z, rr=R, zz=Z)
                self.mask_data[idx] = interp_fun(f=self.base_mask[i], RR=base_R, ZZ=base_Z, rr=R, zz=Z)
                
                if self.base_rhs is not None:
                    self.rhs_data[idx] = interp_fun(f=self.base_rhs[i], RR=base_R[1:-1, 1:-1], ZZ=base_Z[1:-1, 1:-1], rr=R[1:-1, 1:-1], zz=Z[1:-1, 1:-1])
                else:
                    self.rhs_data[idx] = np.zeros((self.nr - 2, self.nz - 2), dtype=np.float32)
            else:
                R = base_R
                Z = base_Z
                self.psi_total_data[idx] = self.base_psi_total[i]
                self.psi_plasma_data[idx] = self.base_psi_plasma[i]
                self.psi_coils_data[idx] = self.base_psi_coils[i]
                self.mask_data[idx] = self.base_mask[i]
                self.rhs_data[idx] = self.base_rhs[i] if self.base_rhs is not None else np.zeros((self.nr - 2, self.nz - 2), dtype=np.float32)

            self.R_data[idx] = R.astype(np.float32)
            self.Z_data[idx] = Z.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        i = int(self.indices[item])

        coil_currents = self.coil_currents[i].astype(np.float32)
        params = self.params[i].astype(np.float32)

        measures = np.concatenate([coil_currents, params], axis=0)
        measures = self.param_norm.apply(measures).astype(np.float32)

        R = self.R_data[item]
        Z = self.Z_data[item]
        psi_total = self.psi_total_data[item]
        psi_plasma = self.psi_plasma_data[item]
        psi_coils = self.psi_coils_data[item]
        mask = self.mask_data[item]
        rhs = self.rhs_data[item]

        metadata = {
            "R": torch.from_numpy(R),
            "Z": torch.from_numpy(Z),
            "psi_axis": float(self.axes[i, 3]),
            "psi_bndry": float(self.axes[i, 2]),
            "R_axis": float(self.axes[i, 0]),
            "Z_axis": float(self.axes[i, 1]),
            "L": float(self.L[i]),
            "Beta0": float(self.Beta0[i]),
            "Ip": float(params[0]),
            "paxis": float(params[1]),
            "alpha_m": float(params[2]),
            "alpha_n": float(params[3]),
            "fvac": float(params[4]),
            "coil_currents": torch.from_numpy(coil_currents),
        }

        return (
            torch.from_numpy(measures),
            torch.from_numpy(R),
            torch.from_numpy(Z),
            torch.from_numpy(psi_total),
            torch.from_numpy(mask),
            torch.from_numpy(psi_plasma),
            torch.from_numpy(psi_coils),
            torch.from_numpy(rhs),
            metadata,
        )


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]