"""Experiment: DN FNO with control-coil currents instead of X-point coordinates.

Baseline (arXiv:2608.05555) feeds the 4 X-point coordinates as input scalars;
this variant replaces them with the freegs control-coil currents (saved as
`coil_currents` in the dataset). Channel count is inferred from the data:
9 on the baseline data/ (3 params + 4 coils, exp001), 11 on data_v2/ (5 params
+ 4 coils, exp003) and 18 on data_v5/ (5 params + 11 MAST coils, exp011).
The model architecture (build_model(in_channels=...)) is unchanged.

Multi-file support (comma-separated npz paths, data_v5 dn+sn mixed training)
and ground-truth geometry fields (xpts_actual / o_point / anchor, data_v4+)
mirror data_dn_fno.DNFnoDataset so the main-script evaluate/visualize pipelines
(the v2 geometry: truth-anchored X-point pairing + ray-cast separatrix) run
unchanged on this dataset.

Historical mirror note: exp001/003/005/007 use the standalone
train/evaluate_dn_fno_coils.py scripts, which still route to this dataset;
exp011 uses the main scripts via --input-mode coils.
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

# data_v5 MAST 18 channels in true construction order: [R, Z] grid channels +
# params(5) + coil_currents(11). Data order (verified against data_v5 npz):
# params = [Ip, paxis, fvac, alpha_m, alpha_n]; coil order follows the freegs
# MAST() machine definition (P2U..P6L + P1 solenoid, machine.py). The old 9/11
# channel lists (TestTokamak P1L/P1U/P2L/P2U) remain valid for data/ & data_v2
# subsets; this constant is documentation-only (no code consumers).
CHANNEL_NAMES_COILS = [
    "R", "Z",
    "Ip", "paxis", "fvac", "alpha_m", "alpha_n",
    "I_P2U", "I_P2L", "I_P3U", "I_P3L",
    "I_P4U", "I_P4L", "I_P5U", "I_P5L",
    "I_P6U", "I_P6L", "I_P1_sol",
]


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
    """Field dataset with coil currents as the extra scalars (9/11/18 channels).

    Field attributes (psi_total / mask / dpdpsi / FdFdpsi / axes) match
    DNFnoDataset so downstream evaluation code can reuse the same physics
    diagnostics unchanged.

    npz_path may be a comma-separated list of files (mixed-config training):
    rows are concatenated per field; R/Z are taken from the first file (all
    files must share the grid — data_v5 dn/sn are both MAST 65x65).

    Ground-truth geometry (data_v4+): xpts_actual / o_point / anchor are
    loaded when present (None otherwise) so the main-script v2 geometry
    (truth-anchored X-point pairing) works on this dataset too.
    """

    def __init__(self, npz_path: str | Path | list, stats: dict | None = None,
                 indices: np.ndarray | None = None):
        self.npz_path = str(npz_path)
        paths = [p.strip() for p in npz_path.split(",") if p.strip()] \
            if isinstance(npz_path, str) else [str(p) for p in npz_path]

        def _load_many(key: str) -> np.ndarray:
            first = True
            parts = []
            for p in paths:
                with np.load(p) as d:
                    if key in ("R", "Z"):
                        if first:
                            parts.append(d[key])
                    else:
                        parts.append(d[key])
                first = False
            return parts[0] if key in ("R", "Z") else np.concatenate(parts, axis=0)

        self.R, self.Z = _normalize_rz(_load_many("R"), _load_many("Z"))
        self.psi_total = _load_many("psi_total").astype(np.float32)      # (N, 65, 65)
        self.params = _load_many("params").astype(np.float32)            # (N, 3 or 5): Ip, paxis, fvac[, alpha_m, alpha_n]
        self.coil_currents = _load_many("coil_currents").astype(np.float32)  # (N, 4 or 11)
        self.mask = _load_many("mask").astype(np.float32)
        self.dpdpsi = _load_many("dpdpsi").astype(np.float32)
        self.FdFdpsi = _load_many("FdFdpsi").astype(np.float32)
        self.axes = _load_many("axes").astype(np.float32)                # R_axis, Z_axis, psi_bndry, psi_axis
        # ground-truth geometry (data_v4+): xpts_actual = separatrix X-points
        # (N, n_xpt, 3) with psi, o_point = magnetic axis (N, 3), anchor =
        # isoflux anchor (N, 2); None for the paper data/ & data_v2 sets.
        # xpts_actual row count varies by config (DN 2 vs SN 1) — mixed
        # concatenation NaN-pads to the max row count (missing rows -> None in
        # the v2 pairing -> NaN geometry, honest).
        def _load_many_pad(key: str) -> np.ndarray:
            parts = []
            for p in paths:
                with np.load(p) as d:
                    parts.append(d[key])
            max_rows = max(a.shape[1] for a in parts)
            out = []
            for a in parts:
                if a.shape[1] < max_rows:
                    pad = np.full((a.shape[0], max_rows, a.shape[2]), np.nan,
                                  dtype=a.dtype)
                    pad[:, :a.shape[1]] = a
                    out.append(pad)
                else:
                    out.append(a)
            return np.concatenate(out, axis=0)

        self.xpts_actual = None
        self.o_point = None
        self.anchor = None
        with np.load(paths[0]) as d:
            if "xpts_actual" in d.files and "o_point" in d.files:
                self.xpts_actual = _load_many_pad("xpts_actual").astype(np.float32)
                self.o_point = _load_many("o_point").astype(np.float32)
            if "anchor" in d.files:
                self.anchor = _load_many("anchor").astype(np.float32)
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
        # constant channels -> 0 instead of NaN (same guard as data_dn_fno.py)
        scalars = (scalars - self.scalar_mean) / np.maximum(self.scalar_std, 1e-8)
        # broadcast the scalar channels to the grid (7 exp001 / 9 exp003)
        scalar_fields = np.broadcast_to(scalars[:, None, None], (len(scalars),) + self.R.shape)

        x = np.concatenate([self.R[None], self.Z[None], scalar_fields], axis=0)  # (2+n_scalars, 65, 65)
        y = (self.psi_total[i] - self.psi_mean) / self.psi_std                   # (65, 65)
        return torch.from_numpy(x.copy()), torch.from_numpy(y[None])
