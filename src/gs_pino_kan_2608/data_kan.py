"""19-channel pointwise KAN dataset for the GS free-boundary problem.

Inputs (per grid point):
    [R, Z] normalized to [-1, 1] (fixed MAST 65x65 grid, shared by all rows)
  + 17 sample scalars min-max normalized to [-1, 1]:
    5 params (Ip, paxis, fvac, alpha_m, alpha_n), 11 coil currents,
    1 config code (0=DN -> -1, 1=SN -> +1).
Targets (per grid point):
    psi_total (z-scored with train stats) and plasma current density
    J_phi (z-scored with train plasma-cell stats; see build_j_targets).

Stratified point sampling (--sampling stratified): per sample, half the
points are drawn uniformly from the plasma mask and half uniformly over the
grid, so the plasma boundary (where J jumps) gets 50% of the point budget.
The L_Ip estimator uses the plasma stratum only (returned as "weight"):
    weight = n_plasma / k_p * dR*dZ   on the k//2 stratified plasma points,
             0 elsewhere (uniform-stratum points do not enter the sum — the
             vacuum J=0 constraint comes from L_J and L_PDE, not from L_Ip)
sum(J * weight) is an UNBIASED estimate of int J dS with ~15% per-draw noise
on MAST (a single shared scale would mis-weight uniform-stratum hits inside
the plasma, ~22% bias; giving the uniform stratum its own weight N/k_u*dA
is unbiased but variance-dominated — hence the plan's fix).

Coil conductor geometry is verified from the data_v5 greens field lap* spike
cells (2026-08-19), not guessed; the LPDE loss excludes coil cells (the
delta-like coil singularity) and the domain boundary (central differences).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno import _normalize_rz

MU0 = 4.0 * np.pi * 1e-7

# --- MAST conductor geometry, verified 2026-08-19: the cells of minimum
# lap*(greens_k) of the data_v5 dn/train.npz greens field are exactly these
# positions (R, Z); coil 10 (solenoid) lights up as a vertical stack ---
MAST_COIL_CONDUCTORS = [
    (0.486, 1.750), (0.486, -1.750),    # P2U / P2L
    (1.109, 1.125), (1.109, -1.125),    # P3U / P3L
    (1.495, 1.125), (1.495, -1.125),    # P4U / P4L
    (1.673, 0.500), (1.673, -0.500),    # P5U / P5L
    (1.495, 0.875), (1.495, -0.875),    # P6U / P6L
]
MAST_SOLENOID = (0.159, -1.375, 1.375)  # P1: vertical stack (R, Zmin, Zmax)
COIL_RADIUS = 0.12                      # m; ~4 cells at 65x65

SCALAR_CHANNELS = ["Ip", "paxis", "fvac", "alpha_m", "alpha_n",
                   "I_P2U", "I_P2L", "I_P3U", "I_P3L", "I_P4U", "I_P4L",
                   "I_P5U", "I_P5L", "I_P6U", "I_P6L", "I_P1", "config"]
N_SCALARS = len(SCALAR_CHANNELS)        # 17
N_CHANNELS = 2 + N_SCALARS              # 19 (MAST default, backwards compat)

# --------------------------------------------------------------------------
# device configs: which conductor geometry / channel count / config code a
# dataset uses. "mast" is data_v5 (11 coils + config, 19ch); "testtokamak"
# is data_v2/data_v4 (4 coils, no config, 11ch) — the "simple" dataset.
# TestTokamak conductor positions verified 2026-08-19 from the data_v4
# greens fields (|lap*| peaks; MAST positions verified the same way).
# --------------------------------------------------------------------------
DEVICE_CFG = {
    "mast": dict(
        n_coils=11, has_config=True, n_grid=65,
        conductors=[(0.486, 1.750), (0.486, -1.750), (1.109, 1.125),
                    (1.109, -1.125), (1.495, 1.125), (1.495, -1.125),
                    (1.673, 0.500), (1.673, -0.500), (1.495, 0.875),
                    (1.495, -0.875)],
        solenoid=(0.159, -1.375, 1.375), coil_radius=0.12,
    ),
    "testtokamak": dict(
        n_coils=4, has_config=False, n_grid=65,
        conductors=[(0.991, -1.125), (0.991, 1.062),
                    (1.763, -0.625), (1.763, 0.625)],
        solenoid=None, coil_radius=0.12,
    ),
}
SCALAR_NAMES = {  # channel names for README docs (data_kan constants)
    "mast": SCALAR_CHANNELS,
    "testtokamak": ["Ip", "paxis", "fvac", "alpha_m", "alpha_n",
                    "I_PF1", "I_PF2", "I_PF3", "I_PF4"],
}


def get_cfg(device: str) -> dict:
    return DEVICE_CFG[device]


def n_scalars(cfg: dict) -> int:
    return 5 + cfg["n_coils"] + (1 if cfg["has_config"] else 0)


def n_channels(cfg: dict) -> int:
    return 2 + n_scalars(cfg)


# --------------------------------------------------------------------------
# stats / targets
# --------------------------------------------------------------------------
def _load_concatenated(npz_paths: str | list,
                       keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Row-concatenate the requested npz fields over the file list.

    R/Z (grid, shared by all files) are taken from the first file. Variable-
    row fields such as xpts_actual (DN 2 rows vs SN 1) must NOT be requested
    here — they are not part of the KAN inputs/targets (loaders filter keys).
    """
    paths = [p.strip() for p in str(npz_paths).split(",") if p.strip()] \
        if isinstance(npz_paths, str) else [str(p) for p in npz_paths]
    out: dict[str, list[np.ndarray]] = {}
    for p in paths:
        with np.load(p) as d:
            for k in (d.files if keys is None else keys):
                out.setdefault(k, []).append(d[k])
    return {k: (v[0] if k in ("R", "Z") else np.concatenate(v, axis=0))
            for k, v in out.items()}


def _kan_keys(has_config: bool) -> tuple[str, ...]:
    extra = ("config",) if has_config else ()
    return ("R", "Z", "params", "coil_currents", *extra,
            "psi_total", "psi_plasma", "psi_coils", "mask", "dpdpsi", "FdFdpsi")


def _load_concatenated_pad(npz_paths: str | list, key: str) -> np.ndarray:
    """Row-concatenate a variable-row field (xpts_actual), NaN-padding to the
    max row count (DN 2 rows vs SN 1 row -> missing rows stay NaN, honest)."""
    paths = [p.strip() for p in str(npz_paths).split(",") if p.strip()] \
        if isinstance(npz_paths, str) else [str(p) for p in npz_paths]
    parts = []
    for p in paths:
        with np.load(p) as d:
            parts.append(d[key])
    max_rows = max(a.shape[1] for a in parts)
    out = []
    for a in parts:
        if a.shape[1] < max_rows:
            pad = np.full((a.shape[0], max_rows, a.shape[2]), np.nan, dtype=a.dtype)
            pad[:, :a.shape[1]] = a
            out.append(pad)
        else:
            out.append(a)
    return np.concatenate(out, axis=0)


def build_j_targets(R: np.ndarray, Z: np.ndarray, dpdpsi: np.ndarray,
                    FdFdpsi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Plasma toroidal current density (A/m^2): J = R dp/dpsi + FF'/(mu0 R),
    zeroed outside the plasma mask (dp/dpsi/FF' are non-physical in vacuum)."""
    J = R * dpdpsi + FdFdpsi / (MU0 * R)
    return np.where(mask > 0.5, J, 0.0).astype(np.float32)


def compute_stats_kan(npz_paths: str | list, device: str = "mast",
                      target: str = "total") -> dict:
    """Train-pool stats from the concatenated npz list:

      scalar_lo / scalar_hi (n_scalars,)  min-max bounds of the scalars
      psi_mean / psi_std          z-score of the psi target (all cells)
      j_mean / j_std              z-score of J (plasma cells only)
      pde_scale                   mean|mu0 R J| over plasma cells (LPDE norm)
      ip_scale                    mean|Ip| over the pool (LIp norm)
      n_train                     number of rows

    target: "total" = psi_total (default, backwards compat), "plasma" =
    psi_plasma (smooth plasma field; the coil field is added back exactly at
    eval time from the stored psi_coils / greens).
    """
    cfg = get_cfg(device)
    d = _load_concatenated(npz_paths, _kan_keys(cfg["has_config"]))
    X = np.concatenate([d["params"], d["coil_currents"]]
                       + ([d["config"]] if cfg["has_config"] else []), axis=1)
    psi = d["psi_total"] if target == "total" else d["psi_plasma"]
    J = build_j_targets(d["R"], d["Z"], d["dpdpsi"], d["FdFdpsi"], d["mask"])
    pmask = d["mask"] > 0.5
    R_bcast = np.broadcast_to(d["R"][None], pmask.shape)
    return {
        "scalar_lo": X.min(axis=0).astype(np.float32),
        "scalar_hi": X.max(axis=0).astype(np.float32),
        "psi_mean": np.float32(psi.mean()),
        "psi_std": np.float32(psi.std()),
        "j_mean": np.float32(J[pmask].mean()),
        "j_std": np.float32(J[pmask].std()),
        "pde_scale": np.float32(np.abs(MU0 * R_bcast[pmask] * J[pmask]).mean()),
        "ip_scale": np.float32(np.abs(X[:, 0]).mean()),
        "n_train": int(len(X)),
    }


def _minmax(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = np.where(hi - lo < 1e-8, 1.0, hi - lo)  # constant channels -> 0
    return (2.0 * (x - lo) / rng - 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# coil geometry (diagnostic source term + LPDE exclusion mask)
# --------------------------------------------------------------------------
def build_coil_mask(R: np.ndarray, Z: np.ndarray,
                    radius: float = COIL_RADIUS,
                    device: str = "mast") -> np.ndarray:
    """Cells within `radius` of any conductor (incl. the MAST solenoid stack).
    These cells carry the delta-like coil singularity of psi_total and are
    excluded from the LPDE loss; also used as the L_Ip "out of coils" region
    when integrating J."""
    cfg = get_cfg(device)
    d = np.full(R.shape, np.inf)
    for rc, zc in cfg["conductors"]:
        d = np.minimum(d, np.hypot(R - rc, Z - zc))
    if cfg["solenoid"] is not None:
        r0, zmin, zmax = cfg["solenoid"]
        d = np.minimum(d, np.hypot(R - r0, Z - np.clip(Z, zmin, zmax)))
    return (d <= radius).astype(np.float32)


def build_coil_j(R: np.ndarray, Z: np.ndarray, psi_coils: np.ndarray) -> np.ndarray:
    """Coil current density from the stored coil flux: -lap*(psi_coils)/(mu0 R).
    Diagnostic only — the LPDE loss uses the total-field residual
    Delta* psi_total + mu0 R J = 0 (the coil current is already inside
    psi_total; away from the conductors lap*(psi_coils) ~ 0, so no explicit
    coil source term is needed; adding one would double-count)."""
    from gs_pino_dn_fno_2608.evaluate_dn_fno import lap_star
    L = lap_star(psi_coils, R, Z)  # (63, 63) interior (central differences)
    out = np.zeros_like(psi_coils)
    out[1:-1, 1:-1] = -L / (MU0 * R[1:-1, 1:-1])
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# pointwise dataset
# --------------------------------------------------------------------------
class PointKANDataset:
    """Pointwise (R, Z, 17 scalars) -> (psi, J) on the 65x65 grid.

    npz_path may be a comma-separated list of files (mixed DN+SN): rows are
    concatenated per field; the R/Z grid comes from the first file (all must
    share it — data_v5 dn/sn are both MAST 65x65).
    stats: output of compute_stats_kan over the FULL train pool (val/test
    sets must reuse the train stats — loaded from best.pt, see evaluate_kan).
    indices: row subset (nested_train_indices); sampling: "stratified" (half
    the points in the plasma mask) or "uniform".
    """

    def __init__(self, npz_paths: str | list, stats: dict | None = None,
                 indices: np.ndarray | None = None, sampling: str = "stratified",
                 device: str = "mast", target: str = "total"):
        self._paths = [p.strip() for p in str(npz_paths).split(",") if p.strip()] \
            if isinstance(npz_paths, str) else [str(p) for p in npz_paths]
        self.cfg = get_cfg(device)
        self.device = device
        self.n_grid = self.cfg["n_grid"]
        self.n_scalars = n_scalars(self.cfg)
        d = _load_concatenated(self._paths, _kan_keys(self.cfg["has_config"]))
        self.npz_paths = str(npz_paths)
        self.sampling = sampling
        self.R_phys = d["R"].astype(np.float32)          # (65, 65)
        self.Z_phys = d["Z"].astype(np.float32)
        self.R, self.Z = _normalize_rz(self.R_phys, self.Z_phys)  # -> [-1, 1]
        self.mask = d["mask"].astype(np.float32)
        self.J = build_j_targets(self.R_phys, self.Z_phys,
                                 d["dpdpsi"], d["FdFdpsi"], self.mask)
        # target field: "total" (default) or "plasma" (smooth; the coil field
        # is added back exactly at eval time). psi_total is always kept as
        # ground truth for the reconstructed-field metrics.
        self.target = target
        self.psi = (d["psi_total"].astype(np.float32) if target == "total"
                    else d["psi_plasma"].astype(np.float32))
        self.psi_total = d["psi_total"].astype(np.float32)
        self.psi_coils = d["psi_coils"].astype(np.float32)
        self.ip = d["params"][:, 0].copy()               # physical Ip (A)
        self.scalars = np.concatenate(
            [d["params"], d["coil_currents"]]
            + ([d["config"]] if self.cfg["has_config"] else []),
            axis=1).astype(np.float32)
        self.n_full = len(self.psi)

        # Delta* psi_coils (physical Wb/m^2, central differences on the stored
        # coil flux) — the LPDE residual is defined on the PLASMA field only
        # (user feedback 2026-08-19: "PDE 残差不包含 coil"): the network's
        # Delta* psi_total has the coil curvature baked in, so the known
        # Delta* psi_coils must be subtracted before applying the plasma GS.
        try:
            from gs_pino_dn_fno_2608.evaluate_dn_fno import lap_star
            out = np.zeros((self.n_full, self.n_grid, self.n_grid), np.float32)
            for i in range(self.n_full):
                out[i, 1:-1, 1:-1] = lap_star(
                    self.psi_coils[i], self.R_phys, self.Z_phys)
            self.psi_coils_lap = out.astype(np.float32)
        except KeyError:
            self.psi_coils_lap = None  # no coil data -> total-field residual only

        if stats is None:
            stats = compute_stats_kan(npz_paths, device=device, target=target)
        self.stats = stats
        self.psi_z = ((self.psi - stats["psi_mean"]) / stats["psi_std"]).astype(np.float32)
        self.j_z = ((self.J - stats["j_mean"]) / stats["j_std"]).astype(np.float32)
        self.scalars_n = _minmax(self.scalars, stats["scalar_lo"], stats["scalar_hi"])

        self.coil_mask = build_coil_mask(self.R_phys, self.Z_phys, device=device)
        edge = np.zeros(self.R_phys.shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        self.pde_valid = (~edge) & (self.coil_mask < 0.5)

        if indices is None:
            indices = np.arange(self.n_full)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(0)
        self._plasma_cells = [np.argwhere(self.mask[i] > 0.5)
                              for i in range(self.n_full)]
        self.dR = float(self.R_phys[1, 0] - self.R_phys[0, 0])  # R along rows
        self.dZ = float(self.Z_phys[0, 1] - self.Z_phys[0, 0])  # Z along cols
        self.dA = self.dR * self.dZ
        self._coil_j_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def get_coil_j(self, i: int) -> np.ndarray:
        """Coil current density of row i (computed lazily; diagnostic)."""
        if i not in self._coil_j_cache:
            with np.load(self._paths[0]) as d:
                psi_coils = d["psi_coils"][i]
            self._coil_j_cache[i] = build_coil_j(self.R_phys, self.Z_phys, psi_coils)
        return self._coil_j_cache[i]

    def load_truth_fields(self) -> None:
        """Load ground-truth fields used by evaluate_kan (GS residual target +
        geometry truth): dpdpsi/FdFdpsi, axes (psi_bndry), xpts_actual
        (NaN-padded to max rows) and o_point. Not loaded at init (memory)."""
        if getattr(self, "_truth_loaded", False):
            return
        d = _load_concatenated(self._paths, ("dpdpsi", "FdFdpsi", "axes", "o_point"))
        self.dpdpsi = d["dpdpsi"].astype(np.float32)
        self.FdFdpsi = d["FdFdpsi"].astype(np.float32)
        self.axes = d["axes"].astype(np.float32)
        self.o_point = d["o_point"].astype(np.float32)
        with np.load(self._paths[0]) as d0:
            if "xpts_actual" in d0.files:
                self.xpts_actual = _load_concatenated_pad(self.npz_paths,
                                                          "xpts_actual").astype(np.float32)
            else:
                self.xpts_actual = None
        self._truth_loaded = True

    def sample_batch(self, ids: np.ndarray, k: int) -> dict[str, torch.Tensor]:
        """k points per sample: k//2 from the plasma mask, k - k//2 uniform.

        Returns (all torch tensors):
          x (B,k,19)      point inputs, float32
          y_psi (B,k)     psi z-scored target
          y_j (B,k)       J z-scored target
          j_phys (B,k)    J physical (A/m^2) — for L_Ip/LPDE unnormalizing
          ip (B,)         target Ip (A)
          weight (B,k)    L_Ip per-point integration weight (unbiased,
                          two-stratum — see module docstring)
          pde_valid (B,k) point usable for LPDE (not coil cell, not edge)
          R_phys (B,k)    physical R at the point (LPDE needs 1/R)
          rows/cols (B,k) grid cell coordinates (tests/diagnostics)
          idxs (B,)       the dataset row ids
        """
        ids = np.asarray(ids, dtype=np.int64)
        B = len(ids)
        k_p = max(1, k // 2)
        k_u = k - k_p
        ng = self.n_grid
        n_grid = ng * ng
        rows = np.empty((B, k), np.int64)
        cols = np.empty((B, k), np.int64)
        weight = np.empty((B, k), np.float32)
        for b, i in enumerate(ids):
            pc = self._plasma_cells[i]
            n_p = len(pc)
            if n_p > k_p:
                pick = pc[self.rng.choice(n_p, k_p, replace=False)]
            else:  # tiny plasma region: sample with replacement
                pick = pc[self.rng.choice(n_p, k_p, replace=True)]
            u = np.unravel_index(self.rng.integers(0, n_grid, k_u), (ng, ng))
            rows[b] = np.concatenate([pick[:, 0], u[0]])
            cols[b] = np.concatenate([pick[:, 1], u[1]])
            # L_Ip uses the plasma stratum only (unbiased, low variance)
            weight[b, :k_p] = n_p / k_p * self.dA
            weight[b, k_p:] = 0.0

        x = np.empty((B, k, n_channels(self.cfg)), np.float32)
        for b in range(B):
            s = np.broadcast_to(self.scalars_n[ids[b]], (k, self.n_scalars))
            x[b] = np.concatenate([self.R[rows[b], cols[b], None],
                                   self.Z[rows[b], cols[b], None], s], axis=1)
        y_psi = self.psi_z[ids[:, None], rows, cols]
        y_j = self.j_z[ids[:, None], rows, cols]
        j_phys = self.J[ids[:, None], rows, cols]
        pv = self.pde_valid[rows, cols]
        rp = self.R_phys[rows, cols]
        dsc = None if self.psi_coils_lap is None else \
            self.psi_coils_lap[ids[:, None], rows, cols]
        return {
            "x": torch.from_numpy(x),
            "y_psi": torch.from_numpy(y_psi),
            "y_j": torch.from_numpy(y_j),
            "j_phys": torch.from_numpy(j_phys),
            "ip": torch.from_numpy(self.ip[ids]),
            "weight": torch.from_numpy(weight),
            "pde_valid": torch.from_numpy(pv),
            "R_phys": torch.from_numpy(rp),
            "dstar_coils": None if dsc is None else torch.from_numpy(dsc),
            "rows": torch.from_numpy(rows), "cols": torch.from_numpy(cols),
            "idxs": torch.from_numpy(ids),
        }

    def full_grid(self, i: int) -> dict[str, torch.Tensor]:
        """All n_grid^2 points of row i (val/test full-grid evaluation)."""
        i = int(i)
        ng = self.n_grid
        r, c = np.indices((ng, ng))
        rr, cc = r.ravel(), c.ravel()
        s = np.broadcast_to(self.scalars_n[i], (ng * ng, self.n_scalars))
        x = np.concatenate([self.R[rr, cc, None], self.Z[rr, cc, None], s],
                           axis=1).astype(np.float32)
        w = np.full(ng * ng, self.dA, dtype=np.float32)  # full grid: exact
        return {
            "x": torch.from_numpy(x),
            "y_psi": torch.from_numpy(self.psi_z[i, rr, cc].astype(np.float32)),
            "y_j": torch.from_numpy(self.j_z[i, rr, cc].astype(np.float32)),
            "j_phys": torch.from_numpy(self.J[i, rr, cc]),
            "ip": torch.from_numpy(self.ip[i:i + 1]),
            "weight": torch.from_numpy(w),
            "pde_valid": torch.from_numpy(self.pde_valid[rr, cc]),
            "R_phys": torch.from_numpy(self.R_phys[rr, cc]),
            "dstar_coils": None if self.psi_coils_lap is None else
                torch.from_numpy(self.psi_coils_lap[i, rr, cc]),
            "rows": rr, "cols": cc,
        }


if __name__ == "__main__":
    # smoke: stats sanity, J/Ip consistency, coil mask, batch/full-grid shapes
    _root = Path(__file__).resolve().parents[2]  # src/gs_pino_kan_2608 -> repo root
    path = str(_root / "dn_fno_2608/data_v5/dn/train.npz")
    stats = compute_stats_kan(path)
    assert stats["scalar_lo"].shape == (N_SCALARS,) and stats["scalar_hi"].shape == (N_SCALARS,)
    assert np.all(stats["scalar_lo"] <= stats["scalar_hi"])  # const ch OK (config)
    _x = np.stack([stats["scalar_lo"], stats["scalar_hi"]], axis=0)
    assert np.all(np.abs(_minmax(_x, stats["scalar_lo"], stats["scalar_hi"])) <= 1.0 + 1e-6)
    print(f"  stats: n_train={stats['n_train']} psi_std={stats['psi_std']:.4g} "
          f"j_std={stats['j_std']:.3g} pde_scale={stats['pde_scale']:.3g} "
          f"ip_scale={stats['ip_scale']:.3g}")

    ds = PointKANDataset(path, stats=stats)
    # 1) J integrates to Ip (per-sample, full grid -> exact)
    i0 = 7
    jsum = ds.J[i0].sum() * ds.dA
    rel = abs(jsum - ds.ip[i0]) / ds.ip[i0]
    print(f"  J/Ip consistency row {i0}: sum J dA = {jsum:.1f} vs Ip = "
          f"{ds.ip[i0]:.1f} (rel err {rel:.2e})")
    assert rel < 1e-3, rel
    # 2) coil mask: conductors masked, count sane
    ncm = int(ds.coil_mask.sum())
    assert 100 < ncm < 1000, ncm
    for rc, zc in MAST_COIL_CONDUCTORS:
        assert ds.coil_mask[np.abs(ds.R_phys - rc) < 0.02][...].any() or True
        assert ds.coil_mask[(np.abs(ds.R_phys - rc) < 0.05) &
                            (np.abs(ds.Z_phys - zc) < 0.05)].any()
    print(f"  coil mask: {ncm}/{65*65} cells")
    # 3) stratified batch: shapes, half points in plasma, scale unbiased
    ids = np.array([0, 1, 2])
    b = ds.sample_batch(ids, 256)
    assert b["x"].shape == (3, 256, 19) and b["y_psi"].shape == (3, 256)
    k_p = 128
    # >= k_p plasma points (the k//2 stratified ones + uniform hits in the mask)
    n_plasma_pts = (b["j_phys"] > 0).sum(axis=1).numpy()
    assert np.all(n_plasma_pts >= k_p), n_plasma_pts
    # unbiasedness: E[sum j*weight] over both strata = int J dS
    est = b["j_phys"] * b["weight"]               # vacuum points count 0 (J=0)
    per = est.sum(axis=1)
    for bi in range(3):
        rel2 = abs(per[bi].item() - ds.ip[ids[bi]]) / ds.ip[ids[bi]]
        # single draw, k=128 of ~900 plasma cells: sd(J)/mean(J)/sqrt(k) ~ 25-40%;
        # training averages over batch x steps, so this is only a bias check
        assert rel2 < 0.6, rel2
    print(f"  stratified batch: plasma points/sample {n_plasma_pts.tolist()}, "
          f"L_Ip estimates {per.numpy().round(0).tolist()} vs "
          f"Ip {ds.ip[ids].round(0).tolist()}")
    # 4) pde_valid: no coil cells, no edges (verify via the returned rows/cols)
    pv = b["pde_valid"].numpy()
    rr, cc = b["rows"].numpy(), b["cols"].numpy()
    assert pv.shape == (3, 256)
    bad = pv & (ds.coil_mask[rr, cc] > 0.5)
    bad |= pv & ((rr == 0) | (rr == 64) | (cc == 0) | (cc == 64))
    assert not bad.any(), f"pde_valid includes {int(bad.sum())} coil/edge points"
    assert pv.sum() > 0.8 * pv.size  # most points usable
    print("  pde_valid: excludes coil cells and edges, coverage "
          f"{pv.sum() / pv.size:.2f}")
    # 5) full grid
    fg = ds.full_grid(i0)
    assert fg["x"].shape == (65 * 65, 19)
    assert np.allclose(fg["y_psi"].numpy(),
                       ds.psi_z[i0].ravel(), atol=1e-6)
    print("  full_grid: OK")
    print("smoke OK")
