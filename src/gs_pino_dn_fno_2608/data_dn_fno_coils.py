"""Experiment: DN FNO with control-coil currents instead of X-point coordinates.

Baseline (arXiv:2608.05555) feeds the 4 X-point coordinates as input scalars;
this variant replaces them with the 4 freegs control-coil currents
(P1L/P1U/P2L/P2U, saved as `coil_currents` in the dataset). Channel count is
inferred from the data: 9 on the baseline data/ (3 params + 4 coils, exp001)
and 11 on data_v2/ (5 params + 4 coils, exp003). The model architecture
(build_model(in_channels=...)) is unchanged.

The baseline modules (data_dn_fno.py / train_dn_fno.py / evaluate_dn_fno.py)
are intentionally NOT modified; this file mirrors their structure for the
experiment. Shared helpers (R/Z mapping, rel L2, nested subsetting) are reused
from data_dn_fno to keep the input pipeline bit-identical otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gs_pino_dn_fno_2608.data_dn_fno import (
    CHANNEL_NAMES,
    _normalize_rz,
    nested_train_indices,
    rel_l2_normalized,
)

CHANNEL_NAMES_COILS = ["R", "Z", "Paxis", "Ip", "fvac",
                       "I_P1L", "I_P1U", "I_P2L", "I_P2U"]


def compute_stats_coils(npz: dict, n: int | None = None) -> dict[str, np.ndarray]:
    """Train-set mean/std of the run params (3 or 5), 4 coil currents and target psi.

    Mirrors data_dn_fno.compute_stats with coil_currents replacing x_coords.
    """
    params = npz["params"][:n]
    coils = npz["coil_currents"][:n]
    psi = npz["psi_total"][:n]

    scalar_mean = np.concatenate([params.mean(axis=0), coils.mean(axis=0)]).astype(np.float32)
    scalar_std = np.concatenate([params.std(axis=0), coils.std(axis=0)]).astype(np.float32)
    psi_mean = float(psi.mean())
    psi_std = float(psi.std())
    return {
        "scalar_mean": scalar_mean, "scalar_std": scalar_std,
        "psi_mean": np.float32(psi_mean), "psi_std": np.float32(psi_std),
        "n_train": int(n if n is not None else len(psi)),
        "input_mode": "coils",
    }


class DNFnoDatasetCoils(Dataset):
    """Field dataset with coil currents as the 4 extra scalars (9 or 11 channels).

    Field attributes (psi_total / mask / dpdpsi / FdFdpsi / axes) match
    DNFnoDataset so downstream evaluation code can reuse the same physics
    diagnostics unchanged.
    """

    def __init__(self, npz_path: str | Path, stats: dict | None = None,
                 indices: np.ndarray | None = None):
        self.npz_path = str(npz_path)
        with np.load(npz_path) as d:
            self.R, self.Z = _normalize_rz(d["R"], d["Z"])
            self.psi_total = d["psi_total"].astype(np.float32)      # (N, 65, 65)
            self.params = d["params"].astype(np.float32)            # (N, 3): Ip, paxis, fvac
            self.coil_currents = d["coil_currents"].astype(np.float32)  # (N, 4)
            self.mask = d["mask"].astype(np.float32)
            self.dpdpsi = d["dpdpsi"].astype(np.float32)
            self.FdFdpsi = d["FdFdpsi"].astype(np.float32)
            self.axes = d["axes"].astype(np.float32)                # R_axis, Z_axis, psi_bndry, psi_axis
            self.n_full = len(self.psi_total)

        if stats is None:  # compute stats from the full dataset (train use)
            stats = compute_stats_coils({
                "params": self.params, "coil_currents": self.coil_currents,
                "psi_total": self.psi_total})
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
        scalars = np.concatenate([self.params[i], self.coil_currents[i]])  # (7,): data | (9,): data_v2
        scalars = (scalars - self.scalar_mean) / self.scalar_std
        # broadcast the scalar channels to the grid (7 exp001 / 9 exp003)
        scalar_fields = np.broadcast_to(scalars[:, None, None], (len(scalars),) + self.R.shape)

        x = np.concatenate([self.R[None], self.Z[None], scalar_fields], axis=0)  # (2+n_scalars, 65, 65)
        y = (self.psi_total[i] - self.psi_mean) / self.psi_std                   # (65, 65)
        return torch.from_numpy(x.copy()), torch.from_numpy(y[None])
