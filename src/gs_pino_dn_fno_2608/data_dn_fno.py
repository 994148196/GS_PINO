"""Dataset for the arXiv:2608.05555 double-null FNO surrogate.

Builds the paper's 9-channel input field (Eq. 5 mapping):
    G: (R, Z, Paxis, Ip, fvac, R_lo^X, Z_lo^X, R_up^X, Z_up^X) -> psi(R,Z)
from the .npz files produced by generate_dn_dataset.py.

Channel count follows the dataset:
  - data/ (paper):       3 params + 4 X-points =  7 scalars =  9 channels
  - data_v2/ (alphas):   5 params + 4 X-points =  9 scalars = 11 channels
  - data_v3/ (alphas + sampled isoflux anchor, use_anchor=True):
                         5 params + 4 X-points + 2 anchor = 11 scalars = 13 channels

Normalization (paper Sec. II.C):
  - R, Z: linear map to [-1, 1]
  - Paxis, Ip, fvac: z-scored with TRAIN-set mean/std, broadcast to the grid
  - the 4 X-point coordinates (and, with use_anchor, the 2 anchor coords):
    likewise z-scored with train stats, broadcast
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
CHANNEL_NAMES_XA = CHANNEL_NAMES + ["R_anc", "Z_anc"]


def compute_stats(npz: dict, n: int | None = None,
                  use_anchor: bool = False,
                  use_config: bool = False) -> dict[str, np.ndarray]:
    """Train-set mean/std of the run params, 4 X-point coords and target psi.

    use_anchor=True (data_v3) additionally z-scores the 2 isoflux-anchor
    coordinates (npz must contain "anchor").
    use_config=True (data_v5) appends the 1-channel config code (0=DN, 1=SN,
    npz must contain "config").
    Scalar order: params | x_coords | [anchor] | [config] — must match
    DNFnoDataset.__getitem__.
    n optionally limits to the first n samples (scaling study: paper computes
    stats from the full train pool; keep n=None for that).
    """
    params = npz["params"][:n]
    xc = npz["x_coords"][:n]
    psi = npz["psi_total"][:n]

    extra_mean, extra_std = [xc.mean(axis=0)], [xc.std(axis=0)]
    if use_anchor:
        anc = npz["anchor"][:n]
        extra_mean.append(anc.mean(axis=0))
        extra_std.append(anc.std(axis=0))
    if use_config:
        cfg_arr = npz["config"][:n]
        extra_mean.append(cfg_arr.mean(axis=0))
        extra_std.append(cfg_arr.std(axis=0))
    scalar_mean = np.concatenate([params.mean(axis=0), *extra_mean]).astype(np.float32)
    scalar_std = np.concatenate([params.std(axis=0), *extra_std]).astype(np.float32)
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
    """Field dataset (9/11/13/14 channels); inputs assembled on the fly per sample.

    use_anchor=True (data_v3, xpoints+anchor mode) appends the 2 sampled
    isoflux-anchor coordinates to the scalars -> 13 channels.
    use_config=True (data_v5, mixed-config mode) appends the 1-channel config
    code (0=DN, 1=SN) -> 14 channels (with anchor); SN samples keep a (0,0)
    upper-X-point placeholder (constant channel -> z-scored 0, disambiguated
    by the config channel).

    npz_path may be a comma-separated list of files (mixed-config training):
    rows are concatenated per field; R/Z are taken from the first file (all
    files must share the grid — data_v5 dn/sn are both MAST 65x65).
    """

    def __init__(self, npz_path: str | Path | list, stats: dict | None = None,
                 indices: np.ndarray | None = None, use_anchor: bool = False,
                 use_config: bool = False):
        self.npz_path = str(npz_path)
        self.use_anchor = use_anchor
        self.use_config = use_config
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
        self.psi_total = _load_many("psi_total").astype(np.float32)   # (N, 65, 65)
        self.params = _load_many("params").astype(np.float32)         # (N, 3 or 5)
        self.x_coords = _load_many("x_coords").astype(np.float32)     # (N, 4)
        self.mask = _load_many("mask").astype(np.float32)
        self.dpdpsi = _load_many("dpdpsi").astype(np.float32)
        self.FdFdpsi = _load_many("FdFdpsi").astype(np.float32)
        self.axes = _load_many("axes").astype(np.float32)             # R_axis, Z_axis, psi_bndry, psi_axis
        self.anchor = None
        if use_anchor:
            with np.load(paths[0]) as d:
                if "anchor" not in d.files:
                    raise RuntimeError(
                        "use_anchor=True but dataset has no 'anchor' field — "
                        "regenerate with generate_dn_dataset.py --isoflux-sampling")
            self.anchor = _load_many("anchor").astype(np.float32)
        self.config = None
        if use_config:
            with np.load(paths[0]) as d:
                if "config" not in d.files:
                    raise RuntimeError(
                        "use_config=True but dataset has no 'config' field — "
                        "regenerate with generate_dn_dataset.py (data_v5)")
            self.config = _load_many("config").astype(np.float32)
        self.n_full = len(self.psi_total)

        if stats is None:  # compute stats from the full dataset (train use)
            stats = compute_stats({
                "params": self.params, "x_coords": self.x_coords,
                "psi_total": self.psi_total,
                **({"anchor": self.anchor} if use_anchor else {}),
                **({"config": self.config} if use_config else {})},
                use_anchor=use_anchor, use_config=use_config)
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
        extra = []
        if self.anchor is not None:
            extra.append(self.anchor[i])
        if self.config is not None:
            extra.append(self.config[i])
        scalars = np.concatenate([self.params[i], self.x_coords[i], *extra])  # (n_scalars,)
        # constant channels (e.g. data_v4 anchor Z == 0 -> std == 0) map to 0
        # instead of NaN; non-constant channels are unchanged (std >> 1e-8)
        scalars = (scalars - self.scalar_mean) / np.maximum(self.scalar_std, 1e-8)
        # broadcast the scalar channels to the grid (7 paper / 9 data_v2 / 11 data_v3-xa)
        scalar_fields = np.broadcast_to(scalars[:, None, None], (len(scalars),) + self.R.shape)

        x = np.concatenate([self.R[None], self.Z[None], scalar_fields], axis=0)  # (2+n_scalars, 65, 65)
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
