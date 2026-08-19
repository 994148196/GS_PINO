"""Dataset for physics-informed FNO training on data_v5/dn (MAST DN, 65x65).

DNPinoDataset subclasses DNFnoDatasetCoils (18-channel input: R, Z + 5 params
+ 11 coil currents) and adds the physics targets the pure-MSE pipeline lacks:

  - y_psi : z-scored psi_plasma   (network predicts the plasma field only)
  - y_j   : z-scored J_phi        (J = R*p' + F*F'/(mu0*R), reconstructed from
            the stored dpdpsi/FdFdpsi fields; zero outside the core mask)
  - j_phys: physical J_phi (A/m^2), the data RHS used by mode `rhs`
  - mask / ip: per-sample core mask and target plasma current

Stats extensions (computed on the full train pool, repository convention):
  psi_plasma_mean/std, j_mean/j_std (plasma cells), pde_scale =
  mean|mu0*R*J|_plasma (Wb/m^2), ip_scale = mean|Ip| (A).

Measured on data_v5/dn/train.npz (2000 samples):
  j_mean 3.47e5, j_std 4.20e5 A/m^2 | psi_plasma mean/std 0.03795/0.03470 Wb
  pde_scale 0.362 Wb/m^2 | ip_scale 5.51e5 A
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils

MU0 = 4.0 * np.pi * 1e-7


def _load_many_fields(paths: list[str], key: str) -> np.ndarray:
    """Concatenate a field across comma-separated npz files (mirrors the base
    class's multi-file support; data_v5 dn/sn share the 65x65 grid)."""
    parts = []
    for p in paths:
        with np.load(p) as d:
            parts.append(d[key])
    return np.concatenate(parts, axis=0)


class DNPinoDataset(DNFnoDatasetCoils):
    """18ch FNO dataset with psi_plasma / J / physics meta for GS residuals."""

    def __init__(self, npz_path: str | Path | list, stats: dict | None = None,
                 indices: np.ndarray | None = None):
        # use_config=False: 18ch input (v5 npz carries an all-zero config
        # field; exp011 trained 18ch, we keep the same convention)
        super().__init__(npz_path, stats=stats, indices=indices,
                         use_config=False)
        paths = [p.strip() for p in str(npz_path).split(",") if p.strip()]

        # physical R/Z grids (shared across samples; base class stores the
        # normalized [-1,1] versions only)
        with np.load(paths[0]) as d:
            self.R_phys = d["R"].astype(np.float32)          # (65, 65) m
            self.Z_phys = d["Z"].astype(np.float32)
        self.dR = float(self.R_phys[1, 0] - self.R_phys[0, 0])
        self.dZ = float(self.Z_phys[0, 1] - self.Z_phys[0, 0])

        self.psi_plasma = _load_many_fields(paths, "psi_plasma").astype(np.float32)
        self.greens = _load_many_fields(paths, "greens").astype(np.float32)  # (N, 11, 65, 65)

        # J_phi = R*p' + F*F'/(mu0*R), zero outside the core mask
        # (float64 internally; verified vs Ip: rel err ~3e-4)
        r = self.R_phys.astype(np.float64)
        j = r * self.dpdpsi.astype(np.float64) \
            + self.FdFdpsi.astype(np.float64) / (MU0 * r)
        self.j_phys = np.where(self.mask > 0.5, j, 0.0).astype(np.float32)

        # extended stats (full pool when stats is None; otherwise assume they
        # were carried over from the train pool)
        if "psi_plasma_mean" not in self.stats:
            self.stats = {**self.stats, **self._compute_extended_stats()}
        self.psi_plasma_mean = float(self.stats["psi_plasma_mean"])
        self.psi_plasma_std = float(self.stats["psi_plasma_std"])
        self.j_mean = float(self.stats["j_mean"])
        self.j_std = float(self.stats["j_std"])
        self.pde_scale = float(self.stats["pde_scale"])
        self.ip_scale = float(self.stats["ip_scale"])

    def _compute_extended_stats(self) -> dict[str, np.ndarray]:
        plasma = self.mask > 0.5
        j_plasma = self.j_phys[plasma]
        mu0rj = MU0 * self.R_phys[None] * self.j_phys          # (N, 65, 65)
        return {
            "psi_plasma_mean": np.float32(self.psi_plasma.mean()),
            "psi_plasma_std": np.float32(self.psi_plasma.std()),
            "j_mean": np.float32(j_plasma.mean()),
            "j_std": np.float32(j_plasma.std()),
            "pde_scale": np.float32(np.abs(mu0rj[plasma]).mean()),
            "ip_scale": np.float32(np.abs(self.params[:, 0]).mean()),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self.indices[idx]
        x, _ = super().__getitem__(idx)                        # (18, 65, 65)
        return {
            "x": x,
            "y_psi": torch.from_numpy(
                ((self.psi_plasma[i] - self.psi_plasma_mean) / self.psi_plasma_std)[None]
                .astype(np.float32)),
            "y_j": torch.from_numpy(
                ((self.j_phys[i] - self.j_mean) / self.j_std)[None].astype(np.float32)),
            "mask": torch.from_numpy(self.mask[i][None]),
            "j_phys": torch.from_numpy(self.j_phys[i][None]),
            "ip": torch.from_numpy(self.params[i, :1]),
        }


if __name__ == "__main__":
    # smoke: print the extended stats and check the reconstruction identities
    ds = DNPinoDataset("dn_fno_2608/data_v5/dn/train.npz")
    s = ds.stats
    item = ds[0]
    print(f"n={len(ds)}  18ch input: {item['x'].shape[0]} channels")
    print(f"psi_total  mean/std : {float(s['psi_mean']):.5f} / {float(s['psi_std']):.5f} Wb")
    print(f"psi_plasma mean/std : {ds.psi_plasma_mean:.5f} / {ds.psi_plasma_std:.5f} Wb")
    print(f"j_mean/j_std        : {ds.j_mean:.4g} / {ds.j_std:.4g} A/m^2")
    print(f"pde_scale / ip_scale: {ds.pde_scale:.4f} Wb/m^2 / {ds.ip_scale:.4g} A")

    # Ip reconstruction check: sum(mask*J)*dA vs params[:, 0]
    ip_est = (ds.j_phys * (ds.mask > 0.5)).sum(axis=(1, 2)) * ds.dR * ds.dZ
    rel = np.abs(ip_est - ds.params[:, 0]) / ds.params[:, 0]
    print(f"Ip reconstruction rel err: mean {rel.mean():.2e}, max {rel.max():.2e}")
    print(f"__getitem__ keys: {sorted(item)} | x {tuple(item['x'].shape)} "
          f"y_psi {tuple(item['y_psi'].shape)} ip {tuple(item['ip'].shape)}")
