"""Dataset generation CLI reproducing arXiv:2608.05555 (double-null FNO surrogate).

Solves double-null free-boundary GS equilibria with FREEGS (TestTokamak geometry)
following the paper's data-generation config:
  - domain R in [0.1, 2.0], Z in [-2.0, 2.0], 65x65 grid, freeBoundaryHagenow
  - profiles: ConstrainPaxisIp(paxis, Ip, fvac) (default shapes alpha_m=1, alpha_n=2;
    --alpha-sampling also samples the exponents: alpha_m ~ U[1,2], alpha_n ~ U[1.5,2.5])
  - params: paxis ~ U[200, 3000] Pa, Ip ~ U[5e4, 4e5] A, fvac ~ U[0.5, 3.0]
  - X-points: reference (1.1, +-0.6) m with symmetric jitter |dR|,|dZ| <= 0.02 m
    (--xpt-jitter / --xpt-jitter-z scale the amplitudes; data_v3 uses 0.20 m,
    data_v4 uses R 0.10 / Z 0.15 around (1.2, 0.6) via --xpt-r0)
  - constraints: double X-points + isoflux to the outboard midplane reference
    (1.5, 0.0); --isoflux-sampling also samples the anchor point
    R~U[1.2,1.8] x Z~U[-0.3,0.3] (data_v3) or, with --anchor-midplane (data_v4),
    (R, 0.0) with R~U[1.35,1.65] (outer midplane separatrix radius; the separatrix
    must pass through both X-points and the anchor, so anchor variation changes
    the boundary shape), gamma=1e-12; Picard solve rtol=1e-3, maxits=50
  - acceptance: converged AND |Ip_sol - Ip_tgt|/Ip_tgt <= 10% AND >= 2 X-points
    AND L finite AND 0 < Beta0 < 1 (profile degeneracy guard)
  - data_v4 physical-validity checks (all off by default, --<flag> gates):
    magnetic axis inside the (Xpt_lo, Xpt_up, anchor) triangle; the three points
    inside the coil-quadrilateral hull with margin and (--require-wall) inside
    the wall; isoflux residual |psi(Xpt)-psi(anchor)|/core <= --max-isoflux-residual;
    actual vs target X-point deviation <= --max-xpt-deviation;
    anchor-X-point distance >= --min-anchor-xpt-dist; core depth
    psi_axis-psi_bndry >= --min-core-depth

Per-sample saved fields (float32; full set kept for the future PINO stage):
  psi_total, psi_plasma, psi_plasma_norm, psi_coils, mask,
  greens (4 coils), coil_currents (4), dpdpsi, FdFdpsi,
  params [Ip, paxis, fvac], x_coords [R_lo, Z_lo, R_up, Z_up],
  axes [R_axis, Z_axis, psi_bndry, psi_axis], L, Beta0, solve_time,
  anchor [R_anc, Z_anc] (only saved with --isoflux-sampling; data_v3),
  + 8 constraint-diagnostic fields under --save-constraint-diag (data_v4):
  xpts_actual (2,3) greedy-paired critical X-points [R,Z,psi],
  o_point (3), xpt_constraint_res (4: Br/Bz at the targets),
  isoflux_res (2: psi_lo/psi_up minus psi(anchor)), psi_at_constraints (3),
  n_iter (1), psi_relchange_final (1)

Generation is chunked (default 500 samples/chunk): each chunk is a standalone
.npz in <out-dir>/<split>/chunk_XXX.npz and acts as a resume checkpoint (existing
chunks are skipped on re-run).

Usage:
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
      --out-dir dn_fno_2608/data --chunk-size 500 --n-jobs 24
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir dn_fno_2608/data --merge
  # data_v3: wider X-point jitter + sampled isoflux anchor
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 2000 --seed 123 \
      --out-dir dn_fno_2608/data_v3 --alpha-sampling --xpt-jitter 0.20 \
      --isoflux-sampling --max-retries 20 --n-jobs 24
  # data_v4: feasible-region sampling + physical-validity acceptance + diagnostics
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
      --out-dir dn_fno_2608/data_v4 --alpha-sampling --xpt-r0 1.2 --xpt-jitter 0.10 \
      --xpt-jitter-z 0.15 --isoflux-sampling --anchor-midplane \
      --max-isoflux-residual 0.25 --max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 \
      --require-wall --min-core-depth 0.005 --save-constraint-diag --max-retries 20 --n-jobs 24
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import trange

try:
    import freegs
    from freegs import boundary, control, critical, jtor
    HAS_FREEGS = True
except ImportError:
    HAS_FREEGS = False

# ---- paper settings (arXiv:2608.05555) ----
RMIN, RMAX = 0.1, 2.0
ZMIN, ZMAX = -2.0, 2.0
NX, NY = 65, 65
XPT_R, XPT_Z = 1.1, 0.6          # reference X-point targets (up/down symmetric)
XPT_JITTER = 0.02                 # default: paper Eq. 4 |dR|, |dZ| <= 0.02 m (--xpt-jitter)
ISOFLUX_REF = (1.5, 0.0)          # default isoflux anchor (outboard midplane, gauge anchor)
# data_v3 (--isoflux-sampling): sampled anchor, separatrix must pass through it
ANCHOR_R_RANGE = (1.2, 1.8)
ANCHOR_Z_RANGE = (-0.3, 0.3)
# data_v4 (--anchor-midplane): anchor = (R, 0.0), R sampled here = outer midplane
# separatrix radius; within the coil-quadrilateral Z=0 slice [1.0, 1.75] m with
# >= 0.1 m margin (COIL_MARGIN), and always outside the X-point band
# (R_xpt max 1.30) so the anchor strictly bounds the axis from the outboard side.
ANCHOR_MIDPLANE_R_RANGE = (1.35, 1.65)
GAMMA = 1e-12                     # Tikhonov regularization (paper)
RTOL, MAXITS = 1e-3, 50           # Picard tolerance / max iterations (paper)
IP_TOL = 0.10                     # acceptance: |Ip_sol - Ip_tgt| / Ip_tgt <= 10%
MIN_XPTS = 2                      # acceptance: find_critical finds >= 2 X-points
COIL_MARGIN = 0.10                # data_v4: min distance of X-pts/anchor to the
                                  # coil-quadrilateral hull edges
# TestTokamak control-coil centers (from the machine definition; pinned for the
# merge sanity print): P1L/P1U = ShapedCoil shape-point mean ~ (1.0, +-1.1),
# P2L/P2U = Coil at (1.75, +-0.6). Convex hull = the "coil quadrilateral".
COIL_HULL = [(1.0, -1.1), (1.75, -0.6), (1.75, 0.6), (1.0, 1.1)]

PARAM_RANGES = {
    "paxis": (200.0, 3000.0),     # Pa
    "Ip": (5e4, 4e5),             # A
    "fvac": (0.5, 3.0),
}

# data_v5 (--machine mast): MAST parameter ranges (probe-verified feasible
# band; 03-mast.py uses paxis 3 kPa / Ip 0.7 MA / fvac 0.4)
MAST_PARAM_RANGES = {
    "paxis": (1e3, 5e3),          # Pa
    "Ip": (3e5, 8e5),             # A
    "fvac": (0.3, 0.8),
}
MAST_ANCHOR_R_RANGE = (1.2, 1.6)  # midplane anchor R (MAST outer midplane)

# ---- data_v5: machine x configuration registry ----
# Extension points for v6 (not implemented in v5, see PLAN_v5_mixed_configs.md):
#   ("mastu_simple", "snow"): snowflake — needs the freegs_snow fork backend
#       (second-order constraints; solve maxits=200 / rtol=5e-3, probe-verified)
#   ("diiid", "limiter"): limiter — the egg's check_limited flow does not
#       converge (official 12-limited.py config replicated: 0/20), needs
#       dedicated debugging or a fixed-boundary approximation
MACHINE_FACTORIES = {
    "test": lambda: freegs.machine.TestTokamak(),
    "mast": lambda: freegs.machine.MAST(),
}
CONFIG_SPECS = {
    ("test", "dn"): {},   # defaults = paper/data_v4 behaviour
    ("test", "sn"): {"xpt_n": 1},
    ("mast", "dn"): {"xpt_n": 2, "has_wall": False, "param_ranges": MAST_PARAM_RANGES,
                     "anchor_r_range": MAST_ANCHOR_R_RANGE, "xpt_r0": 0.7, "xpt_z0": 1.1},
    ("mast", "sn"): {"xpt_n": 1, "has_wall": False, "param_ranges": MAST_PARAM_RANGES,
                     "anchor_r_range": MAST_ANCHOR_R_RANGE, "xpt_r0": 0.7, "xpt_z0": 1.1},
}

# data_v2 extension (--alpha-sampling): profile-shape exponents
# shape(psi_n) = (1 - psi_n^alpha_m)^alpha_n; freegs checks alpha_m/alpha_n >= 0
# (alpha_m = 0 divides by zero), literature never exceeds ~3. Narrow range
# centred on the paper default (1.0, 2.0) keeps the solver acceptance high.
ALPHA_RANGES = {
    "alpha_m": (1.0, 2.0),
    "alpha_n": (1.5, 2.5),
}

COIL_NAMES = ["P1L", "P1U", "P2L", "P2U"]

# scalar fields stacked per-sample on merge
# ("anchor" only present when --isoflux-sampling, the 8 constraint-diagnostic
# fields only under --save-constraint-diag; save/merge skip missing keys)
STACKED_KEYS = [
    "psi_total", "psi_plasma", "psi_plasma_norm", "psi_coils", "mask",
    "dpdpsi", "FdFdpsi", "greens", "coil_currents", "params", "x_coords",
    "axes", "L", "Beta0", "solve_time", "anchor", "config",
    "xpts_actual", "o_point", "xpt_constraint_res", "isoflux_res",
    "psi_at_constraints", "n_iter", "psi_relchange_final",
]
SHARED_KEYS = ["R", "Z"]  # stored once per chunk, not stacked


def sample_params(rng: np.random.Generator, alpha: bool = False,
                  ranges: dict | None = None) -> dict[str, float]:
    """Sample one equilibrium parameter vector (paper Eq. 2-4 ranges).

    alpha=True additionally samples the profile-shape exponents alpha_m/alpha_n
    (data_v2 extension; the paper fixes them at 1.0/2.0). ranges overrides the
    parameter ranges (data_v5 MAST band).
    """
    ranges = dict(PARAM_RANGES if ranges is None else ranges)
    if alpha:
        ranges.update(ALPHA_RANGES)
    return {name: float(rng.uniform(*ranges[name])) for name in ranges}


def sample_xpoints(rng: np.random.Generator, jitter: float = XPT_JITTER,
                   jitter_z: float | None = None, r0: float = XPT_R,
                   z0: float = XPT_Z):
    """Symmetric X-point jitter preserving exact up-down symmetry (paper Eq. 4).

    jitter = half-width of the |dR| band around r0; jitter_z (default = jitter)
    is the |dZ| band around z0. rng draw order (dR then dZ) is fixed so the
    baseline stream is unchanged for the defaults. data_v3: 0.20 around (1.1, 0.6);
    data_v4: R 0.10 / Z 0.15 around (1.2, 0.6) (--xpt-r0/--xpt-jitter/--xpt-jitter-z).
    """
    if jitter_z is None:
        jitter_z = jitter
    dR = float(rng.uniform(-jitter, jitter))
    dZ = float(rng.uniform(-jitter, jitter))
    lo = (r0 + dR, -(z0 + dZ))
    up = (r0 + dR, +(z0 + dZ))
    return lo, up


def sample_anchor(rng: np.random.Generator, isoflux: bool = False,
                  midplane: bool = False,
                  r_range: tuple[float, float] = ANCHOR_MIDPLANE_R_RANGE) -> tuple[float, float]:
    """Sample the isoflux anchor point; fixed (1.5, 0.0) unless --isoflux-sampling.

    CRITICAL: draws nothing from rng when isoflux=False, so the baseline rng
    stream (and hence the whole baseline dataset) is unchanged. With
    midplane=True (data_v4 --anchor-midplane) the anchor is (R, 0.0) with
    R ~ U[r_range] (default ANCHOR_MIDPLANE_R_RANGE; data_v5 MAST uses
    MAST_ANCHOR_R_RANGE): the separatrix's outer midplane radius.
    """
    if not isoflux:
        return ISOFLUX_REF
    if midplane:
        return (float(rng.uniform(*r_range)), 0.0)
    return (float(rng.uniform(*ANCHOR_R_RANGE)), float(rng.uniform(*ANCHOR_Z_RANGE)))


# ---- data_v4 physical-validity helpers (pure numpy, no new deps) ----
def _control_coil_centers(tokamak) -> list[tuple[float, float]]:
    """Centers of the control coils: plain Coil -> (R, Z), ShapedCoil -> shape-point
    mean, Solenoid (MAST central stack) -> (Rs, Z midpoint).

    Returned in convex-hull (counter-clockwise) order: tokamak.coils iterates
    P1L, P1U, P2L, P2U, which is a self-intersecting bowtie as a polygon.
    """
    centers = []
    for label, coil in tokamak.coils:
        if not coil.control:
            continue
        if hasattr(coil, "Rs"):  # Solenoid: distributed stack along Z
            centers.append((float(coil.Rs), 0.5 * (float(coil.Zsmin) + float(coil.Zsmax))))
        elif np.isscalar(coil.R):
            centers.append((float(coil.R), float(coil.Z)))
        else:
            centers.append((float(np.mean(coil.R)), float(np.mean(coil.Z))))
    # sort by polar angle around the centroid -> CCW convex order (exact for quads)
    c = np.asarray(centers, dtype=np.float64)
    c0 = c.mean(axis=0)
    idx = np.argsort(np.arctan2(c[:, 1] - c0[1], c[:, 0] - c0[0]))
    return [(float(c[j, 0]), float(c[j, 1])) for j in idx]


def _point_in_triangle(p, tri, eps: float = 1e-9) -> bool:
    """Barycentric point-in-triangle test (strict inside, tolerance eps)."""
    x, y = float(p[0]), float(p[1])
    (x1, y1), (x2, y2), (x3, y3) = [(float(a), float(b)) for a, b in tri]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-15:
        return False  # degenerate triangle
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    return a > -eps and b > -eps and (1.0 - a - b) > -eps


def _point_in_polygon(p, verts) -> bool:
    """Ray-casting point-in-polygon for a simple (here convex) polygon."""
    x, y = float(p[0]), float(p[1])
    inside = False
    for i in range(len(verts)):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % len(verts)]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_int > x:
                inside = not inside
    return inside


def _dist_to_poly_edges(p, verts) -> float:
    """Min Euclidean distance from point to the boundary segments of a polygon."""
    x, y = float(p[0]), float(p[1])
    n = len(verts)
    best = np.inf
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 > 0:
            t = min(1.0, max(0.0, ((x - x1) * dx + (y - y1) * dy) / L2))
            best = min(best, np.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return float(best)


def _inside_with_margin(p, verts, margin: float) -> bool:
    """Point inside polygon and at distance >= margin from every edge."""
    return _point_in_polygon(p, verts) and _dist_to_poly_edges(p, verts) >= margin


def _at(eq, p, method: str = "psiRZ") -> float:
    """Evaluate eq.<method>(R, Z) at p and return a plain float (any array shape)."""
    return float(np.asarray(getattr(eq, method)(*p)).reshape(-1)[0])


def _solve_one(args: tuple) -> dict | None:
    """Solve a single DN equilibrium. Module-level for joblib on Windows.

    args = (params, i_seed, cfg); cfg is a dict of sampling/acceptance options
    (all defaults keep the baseline/data_v3 behaviour bit-identical).
    """
    params, i_seed, cfg = args
    t0 = time.perf_counter()

    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    # data_v2: alpha exponents sampled (absent -> paper default 1.0/2.0)
    alpha_m = params.get("alpha_m", 1.0)
    alpha_n = params.get("alpha_n", 2.0)
    rng = np.random.default_rng(i_seed)
    lo, up = sample_xpoints(rng, jitter=cfg["xpt_jitter"], jitter_z=cfg["xpt_jitter_z"],
                            r0=cfg["xpt_r0"], z0=cfg["xpt_z0"])
    anchor = sample_anchor(rng, isoflux=cfg["isoflux_sampling"],
                           midplane=cfg["anchor_midplane"],
                           r_range=cfg["anchor_r_range"])

    try:
        tokamak = cfg["machine_factory"]()

        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=RMIN, Rmax=RMAX, Zmin=ZMIN, Zmax=ZMAX,
            nx=NX, ny=NY,
            boundary=boundary.freeBoundaryHagenow,
        )

        # Profile shapes: paper fixes alpha_m=1, alpha_n=2 ("shapes fixed by the
        # FREEGS setup"); --alpha-sampling additionally varies the exponents.
        profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac,
                                         alpha_m=alpha_m, alpha_n=alpha_n)

        if cfg["xpt_n"] == 1:
            # data_v5 SN: single X-point + single isoflux to the anchor
            # (3 constraints vs 4 control coils — no longer over-determined;
            # asymmetric anchors with Z != 0 are infeasible, probe-verified 0/10)
            constrain = control.constrain(
                xpoints=[lo],
                isoflux=[(*lo, *anchor)],
                gamma=GAMMA,
            )
        else:
            constrain = control.constrain(
                xpoints=[lo, up],
                isoflux=[(*lo, *anchor), (*up, *anchor)],
                gamma=GAMMA,
            )

        # convergenceInfo only under --save-constraint-diag (returns the Picard
        # psi-relative-change history used for the n_iter / psi_relchange fields)
        conv = freegs.solve(eq, profiles, constrain, rtol=RTOL, maxits=MAXITS,
                            show=False, convergenceInfo=cfg["save_constraint_diag"])

        # ---- acceptance checks (common) ----
        if eq.psi_axis is None or eq.psi_bndry is None:
            return None  # reject: no psi axis/bndry
        ip_err = abs(eq.plasmaCurrent() - Ip) / Ip
        if ip_err > IP_TOL:
            return None  # reject: |Ip_sol-Ip_tgt|/Ip_tgt > IP_TOL
        opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
        # data_v5 SN: count only separatrix X-points (psi >= psi_bndry).
        # find_critical also reports vacuum-region saddle points below the
        # boundary (TestTokamak ~2, MAST 5-6 per sample) — a single-null
        # separatrix passes through exactly one X-point.
        sep_xpt = [p for p in xpt if float(p[2]) >= eq.psi_bndry - 1e-6]
        if cfg["xpt_n"] == 1:
            if len(sep_xpt) != 1:
                return None  # reject: separatrix X-point count != 1 (SN)
        elif len(xpt) < MIN_XPTS:
            return None  # reject: < MIN_XPTS X-points
        # boundary flux definition: DN (paper) = mean of the two X-point fluxes;
        # SN (data_v5) = the single separatrix X-point flux
        psi_bndry = sep_xpt[0][2] if cfg["xpt_n"] == 1 else 0.5 * (xpt[0][2] + xpt[1][2])
        # magnetic axis (SN): find_critical reports vacuum-region extrema whose
        # total psi (incl. coil field) can exceed the core psi — e.g. a fixed
        # point near (1.00, -1.10) in TestTokamak and R > 1.8 near the wall.
        # The true axis is the highest-psi O-point *between* the lower X-point
        # and the midplane anchor; DN keeps opt[0] (unchanged behaviour).
        if cfg["xpt_n"] == 1 and opt:
            cand = [o for o in opt if lo[0] < o[0] < anchor[0] and abs(o[1]) < abs(lo[1])]
            axis_pt = max(cand, key=lambda o: o[2]) if cand else opt[0]
        else:
            axis_pt = opt[0] if opt else None
        # profile degeneracy guard (e.g. extreme shapes): L finite, Beta0 in (0,1)
        if not (np.isfinite(profiles.L) and 0.0 < profiles.Beta0 < 1.0):
            return None  # reject: degenerate profile

        # ---- data_v4 physical-validity checks (active only under --<flag>) ----
        diag = cfg["save_constraint_diag"]
        v4_active = (diag
                     or cfg["max_isoflux_residual"] is not None
                     or cfg["max_xpt_deviation"] is not None
                     or cfg["min_anchor_xpt_dist"] is not None
                     or cfg["require_wall"]
                     or cfg["min_core_depth"] is not None)
        if v4_active:
            # core depth (same psi_bndry definition as the saved axes field)
            core = eq.psi_axis - psi_bndry
            if core <= 0 or not np.isfinite(core):
                return None  # reject: core depth <= 0
            if cfg["min_core_depth"] is not None and core < cfg["min_core_depth"]:
                return None  # reject: core depth < threshold

            p_lo = _at(eq, lo)
            p_up = _at(eq, up)
            p_anc = _at(eq, anchor)

            # isoflux residual: the 4-coil Tikhonov fixed point keeps residual
            # orthogonal to the reachable subspace, so |psi(Xpt)-psi(anchor)|/core
            # can be large (data_v3 mean 0.50, max 2.10); reject above threshold
            # (SN has no upper X-point: the residual is |psi(lo)-psi(anchor)|/core)
            if cfg["max_isoflux_residual"] is not None:
                res_terms = [abs(p_lo - p_anc)]
                if cfg["xpt_n"] == 2:
                    res_terms.append(abs(p_up - p_anc))
                if max(res_terms) / core > cfg["max_isoflux_residual"]:
                    return None  # reject: isoflux residual too high

            # geometry: coil-quadrilateral convex hull (with margin) + wall
            # (SN: no upper X-point — x_coords up is a (0,0) placeholder)
            pts = [lo, anchor] if cfg["xpt_n"] == 1 else [lo, up, anchor]
            hull_verts = np.asarray(_control_coil_centers(tokamak), dtype=np.float64)
            # MAST has no wall (tokamak.wall is None): the wall check only
            # applies to machines with one and --require-wall
            wall_verts = None
            if tokamak.wall is not None:
                wall_verts = np.column_stack([
                    np.asarray(tokamak.wall.R, dtype=np.float64),
                    np.asarray(tokamak.wall.Z, dtype=np.float64),
                ])
            for p in pts:
                if not _inside_with_margin(p, hull_verts, cfg["coil_margin"]):
                    return None  # reject: outside coil hull
                if cfg["require_wall"] and not _point_in_polygon(p, wall_verts):
                    return None  # reject: outside wall
            # magnetic axis: DN inside the (Xpt_lo, Xpt_up, anchor) triangle;
            # SN between the lower X-point and the midplane anchor (Z compared
            # against |Z_lo| — x_coords Z is negative for the lower X-point)
            if cfg["xpt_n"] == 1:
                ra, za = axis_pt[0], axis_pt[1]
                if not (lo[0] < ra < anchor[0] and abs(za) < abs(lo[1])):
                    return None  # reject: SN axis not between Xpt and anchor
            elif not _point_in_triangle(opt[0][:2], (lo, up, anchor)):
                return None  # reject: axis outside triangle
            # anchor not too close to either X-point (SN: the lower one)
            if cfg["min_anchor_xpt_dist"] is not None:
                if np.hypot(anchor[0] - lo[0], anchor[1] - lo[1]) < cfg["min_anchor_xpt_dist"]:
                    return None  # reject: anchor too close to lo X-pt
                if cfg["xpt_n"] == 2 and np.hypot(anchor[0] - up[0], anchor[1] - up[1]) < cfg["min_anchor_xpt_dist"]:
                    return None  # reject: anchor too close to up X-pt
            # each target X-point within --max-xpt-deviation of a separatrix
            # (SN) / any actual (DN) X-point
            if cfg["max_xpt_deviation"] is not None:
                cand = sep_xpt if cfg["xpt_n"] == 1 else xpt
                xpt_xy = np.asarray([(r, z) for r, z, _ in cand], dtype=np.float64)
                for tgt in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):
                    if np.hypot(xpt_xy[:, 0] - tgt[0], xpt_xy[:, 1] - tgt[1]).min() > cfg["max_xpt_deviation"]:
                        return None  # reject: X-pt deviation > threshold

            # ---- constraint diagnostics (all from values computed above) ----
            if diag:
                cand = sep_xpt if cfg["xpt_n"] == 1 else xpt
                xpt_xy = np.asarray([(r, z) for r, z, _ in cand], dtype=np.float64)
                remaining = list(range(len(xpt_xy)))
                xpts_actual = []
                for tgt in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):  # greedy pairing
                    j = min(remaining, key=lambda j: np.hypot(xpt_xy[j, 0] - tgt[0],
                                                              xpt_xy[j, 1] - tgt[1]))
                    xpts_actual.append(cand[j][:3])
                    remaining.remove(j)
                xpts_actual = np.asarray(xpts_actual, dtype=np.float64)
                o_point = np.asarray(axis_pt[:3], dtype=np.float64)

        # ---- extract fields ----
        psi_total = eq.psi()
        psi_plasma = eq.plasma_psi
        psi_coils = eq.tokamak.psi(eq.R, eq.Z)
        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)

        mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)

        # GS RHS ingredients for residual diagnostics / future PINO stage:
        # RHS = -mu0 R^2 dp/dpsi - F dF/dpsi, with p'/ff' evaluated on psi_norm
        psi_norm = (psi_total - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis + 1e-30)
        dpdpsi = profiles.pprime(psi_norm)
        fdFdpsi = profiles.ffprime(psi_norm)

        greens, coil_currents = [], []
        for label, coil in tokamak.coils:
            if coil.control:
                greens.append(coil.createPsiGreens(eq.R, eq.Z).astype(np.float32))
                coil_currents.append(float(coil.current))

        R_axis = axis_pt[0] if opt else 1.0
        Z_axis = axis_pt[1] if opt else 0.0
        # boundary flux: DN = average of the two X-point fluxes (paper);
        # SN = the single separatrix X-point flux (computed above)

        result = {
            "psi_total": psi_total.astype(np.float32),
            "psi_plasma": psi_plasma.astype(np.float32),
            "psi_plasma_norm": psi_plasma_norm.astype(np.float32),
            "psi_coils": psi_coils.astype(np.float32),
            "mask": mask.astype(np.float32),
            "dpdpsi": dpdpsi.astype(np.float32),
            "FdFdpsi": fdFdpsi.astype(np.float32),
            "greens": np.stack(greens),
            "coil_currents": np.array(coil_currents, dtype=np.float32),
            "params": np.array(
                [Ip, paxis, fvac]
                + ([alpha_m, alpha_n] if "alpha_m" in params else []),
                dtype=np.float32),
            # SN: x_coords stays 4-ch with the upper pair as (0,0) placeholders
            # (constant channel -> z-scored 0; the config field disambiguates)
            "x_coords": np.array([*lo, *up] if cfg["xpt_n"] == 2 else [*lo, 0.0, 0.0],
                                 dtype=np.float32),
            "config": np.array([0.0 if cfg["xpt_n"] == 2 else 1.0], dtype=np.float32),
            "axes": np.array([R_axis, Z_axis, psi_bndry, eq.psi_axis], dtype=np.float32),
            "L": np.array([profiles.L], dtype=np.float32),
            "Beta0": np.array([profiles.Beta0], dtype=np.float32),
            "solve_time": np.array([time.perf_counter() - t0], dtype=np.float32),
        }
        if cfg["isoflux_sampling"]:  # data_v3: anchor saved only when sampled
            result["anchor"] = np.array(anchor, dtype=np.float32)
        if diag:  # data_v4: constraint diagnostics (over-determined info)
            result.update({
                "xpts_actual": xpts_actual.astype(np.float32),
                "o_point": o_point.astype(np.float32),
                "xpt_constraint_res": np.array(
                    [_at(eq, lo, "Br"), _at(eq, lo, "Bz")]
                    + ([_at(eq, up, "Br"), _at(eq, up, "Bz")] if cfg["xpt_n"] == 2 else []),
                    dtype=np.float32),
                "isoflux_res": np.array([p_lo - p_anc]
                                        + ([p_up - p_anc] if cfg["xpt_n"] == 2 else []),
                                        dtype=np.float32),
                "psi_at_constraints": np.array(
                    [p_lo, p_up, p_anc] if cfg["xpt_n"] == 2 else [p_lo, p_anc],
                    dtype=np.float32),
                "n_iter": np.array([len(conv[1])], dtype=np.float32),
                "psi_relchange_final": np.array([conv[1][-1]], dtype=np.float32),
            })
        return result
    except Exception as e:
        return None


def _solve_with_retry(args: tuple, max_retries: int = 5, rng: np.random.Generator | None = None) -> dict | None:
    """Solve with full parameter resampling on rejection (paper: 100% acceptance expected).

    When rng is None a per-sample deterministic generator is derived from the
    seed (callers don't pass a shared generator to joblib workers). On rejection
    the whole parameter vector is re-drawn (same family: alpha keys survive the
    worker-process boundary; the CLI flag does not) — rejection is usually a
    deterministic property of the (paxis, Ip, fvac, alpha_m, alpha_n) combo
    (e.g. Beta0 outside (0,1)), which jitter-only retries can never fix.
    With --isoflux-sampling the anchor is re-drawn inside _solve_one from the
    per-attempt rng (same mechanism as X-point jitter), so retries also
    resample the anchor.
    """
    params, i_seed, cfg = args
    if rng is None:
        rng = np.random.default_rng(i_seed + 7_000_003)
    for attempt in range(max_retries):
        result = _solve_one((params, i_seed + attempt * 1_000_000, cfg))
        if result is not None:
            return result
        params = sample_params(rng, alpha=("alpha_m" in params),
                               ranges=cfg["param_ranges"])
    return None


def save_chunk(chunk_dir: Path, chunk_idx: int, results: list[dict]) -> Path:
    """Save one chunk as a standalone .npz (resume checkpoint).

    Only keys present in the samples are stored ("anchor" appears only under
    --isoflux-sampling).
    """
    arrays = {key: np.stack([r[key] for r in results])
              for key in STACKED_KEYS if key in results[0]}
    arrays["R"] = R_GLOBAL
    arrays["Z"] = Z_GLOBAL
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / f"chunk_{chunk_idx:03d}.npz"
    np.savez(path, **arrays)
    return path


# shared grid arrays (set in generate())
R_GLOBAL = None
Z_GLOBAL = None


def build_cfg(machine: str = "test", config: str = "dn",
              xpt_jitter: float = XPT_JITTER, xpt_jitter_z: float | None = None,
              isoflux_sampling: bool = False, anchor_midplane: bool = False,
              coil_margin: float = COIL_MARGIN,
              max_isoflux_residual: float | None = None,
              max_xpt_deviation: float | None = None,
              min_anchor_xpt_dist: float | None = None,
              require_wall: bool = False,
              min_core_depth: float | None = None,
              save_constraint_diag: bool = False,
              xpt_r0: float | None = None, xpt_z0: float | None = None) -> dict:
    """Resolve the (machine, config) spec into a solver/acceptance cfg dict.

    data_v5: machine x configuration registry (CONFIG_SPECS); defaults keep
    the paper/data_v4 behaviour. Shared by generate() and the probe script.
    """
    if xpt_jitter_z is None:
        xpt_jitter_z = xpt_jitter
    spec = CONFIG_SPECS.get((machine, config), {})
    xpt_n = spec.get("xpt_n", 2)
    if xpt_r0 is None:
        xpt_r0 = spec.get("xpt_r0", XPT_R)
    if xpt_z0 is None:
        xpt_z0 = spec.get("xpt_z0", XPT_Z)
    has_wall = spec.get("has_wall", True)
    if require_wall and not has_wall:
        print(f"  WARNING: --require-wall ignored ({machine} has no wall)")
    return {
        "xpt_n": xpt_n,
        "machine_factory": MACHINE_FACTORIES[machine],
        "has_wall": has_wall,
        "param_ranges": spec.get("param_ranges", PARAM_RANGES),
        "anchor_r_range": spec.get("anchor_r_range", ANCHOR_MIDPLANE_R_RANGE),
        "xpt_r0": xpt_r0, "xpt_z0": xpt_z0,
        "xpt_jitter": xpt_jitter, "xpt_jitter_z": xpt_jitter_z,
        "isoflux_sampling": isoflux_sampling, "anchor_midplane": anchor_midplane,
        "coil_margin": coil_margin,
        "max_isoflux_residual": max_isoflux_residual,
        "max_xpt_deviation": max_xpt_deviation,
        "min_anchor_xpt_dist": min_anchor_xpt_dist,
        "require_wall": require_wall and has_wall,
        "min_core_depth": min_core_depth,
        "save_constraint_diag": save_constraint_diag,
    }


def generate(out_dir: str, split: str, n_samples: int, seed: int,
             chunk_size: int, n_jobs: int, alpha_sampling: bool = False,
             max_retries: int = 5, xpt_jitter: float = XPT_JITTER,
             isoflux_sampling: bool = False, xpt_r0: float | None = None,
             xpt_z0: float | None = None, xpt_jitter_z: float | None = None,
             anchor_midplane: bool = False, max_isoflux_residual: float | None = None,
             max_xpt_deviation: float | None = None,
             min_anchor_xpt_dist: float | None = None, require_wall: bool = False,
             coil_margin: float = COIL_MARGIN,
             min_core_depth: float | None = None,
             save_constraint_diag: bool = False,
             config: str = "dn", machine: str = "test") -> None:
    """Generate one split in resumable chunks."""
    global R_GLOBAL, Z_GLOBAL
    cfg = build_cfg(machine, config, xpt_jitter, xpt_jitter_z, isoflux_sampling,
                    anchor_midplane, coil_margin, max_isoflux_residual,
                    max_xpt_deviation, min_anchor_xpt_dist, require_wall,
                    min_core_depth, save_constraint_diag, xpt_r0, xpt_z0)

    if not HAS_FREEGS:
        raise RuntimeError("freegs not found. Install freegs first.")

    out_dir = Path(out_dir)
    chunk_dir = out_dir / split
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # shared grid (identical across samples)
    eq_tmp = freegs.Equilibrium(
        tokamak=cfg["machine_factory"](),
        Rmin=RMIN, Rmax=RMAX, Zmin=ZMIN, Zmax=ZMAX, nx=NX, ny=NY,
        boundary=boundary.freeBoundaryHagenow,
    )
    R_GLOBAL = eq_tmp.R.astype(np.float32)
    Z_GLOBAL = eq_tmp.Z.astype(np.float32)

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    print(f"\n{'='*70}")
    print(f"  {config.upper()} dataset generation | machine={machine} | split={split} | n={n_samples} | seed={seed}")
    print(f"  grid {NX}x{NY}, R [{RMIN},{RMAX}], Z [{ZMIN},{ZMAX}], chunk_size={chunk_size}")
    if xpt_jitter_z != xpt_jitter:
        print(f"  X-pts ({cfg['xpt_r0']},+-{cfg['xpt_z0']}) R+-{xpt_jitter} / Z+-{xpt_jitter_z} m, "
              f"isoflux->{'sampled' if isoflux_sampling else ISOFLUX_REF}, "
              f"gamma={GAMMA}, maxits={MAXITS}")
    else:
        print(f"  X-pts ({cfg['xpt_r0']},+-{cfg['xpt_z0']})+-{xpt_jitter} m, "
              f"isoflux->{'sampled' if isoflux_sampling else ISOFLUX_REF}, "
              f"gamma={GAMMA}, maxits={MAXITS}")
    if isoflux_sampling:
        if anchor_midplane:
            print(f"  anchor: (R, 0.0), R ~ U{ANCHOR_MIDPLANE_R_RANGE} (data_v4 midplane)")
        else:
            print(f"  anchor: R ~ U{ANCHOR_R_RANGE}, Z ~ U{ANCHOR_Z_RANGE} (data_v3)")
    if alpha_sampling:
        print(f"  profile shapes: alpha_m ~ U{ALPHA_RANGES['alpha_m']}, "
              f"alpha_n ~ U{ALPHA_RANGES['alpha_n']} (data_v2)")
    else:
        print("  profile shapes: alpha_m=1.0, alpha_n=2.0 fixed (paper)")
    checks = [f"isoflux_res<={max_isoflux_residual}" if max_isoflux_residual is not None else "",
              f"xpt_dev<={max_xpt_deviation}" if max_xpt_deviation is not None else "",
              f"anc_xpt_dist>={min_anchor_xpt_dist}" if min_anchor_xpt_dist is not None else "",
              "wall" if require_wall else "",
              f"core>={min_core_depth}" if min_core_depth is not None else ""]
    if max_isoflux_residual is not None or max_xpt_deviation is not None or \
            min_anchor_xpt_dist is not None or require_wall or min_core_depth is not None:
        checks.append("axis in tri(lo,up,anchor)")
        print(f"  acceptance+ : {', '.join(checks)} (data_v4)")
    if save_constraint_diag:
        print(f"  saving constraint diagnostics (8 fields, --save-constraint-diag)")
    print(f"{'='*70}")

    from joblib import Parallel, delayed

    total_accepted, total_solves = 0, 0
    for ci in trange(n_chunks, desc=f"{split} chunks"):
        chunk_path = chunk_dir / f"chunk_{ci:03d}.npz"
        if chunk_path.exists():
            print(f"  chunk {ci:03d} exists, skipping (resume)")
            with np.load(chunk_path) as d:
                total_accepted += d["psi_total"].shape[0]
            continue

        i0 = ci * chunk_size
        idx = range(i0, min(i0 + chunk_size, n_samples))
        # pre-sample deterministic params for this chunk
        chunk_params = [sample_params(rng, alpha=alpha_sampling, ranges=cfg["param_ranges"])
                        for _ in idx]

        t0 = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_solve_with_retry)((p, seed * 100_000 + i, cfg), max_retries=max_retries)
            for i, p in zip(idx, chunk_params)
        )
        dt = time.perf_counter() - t0

        valid = [r for r in results if r is not None]
        total_accepted += len(valid)
        total_solves += len(results)
        if not valid:
            raise RuntimeError(
                f"chunk {ci:03d}: 0/{len(results)} accepted — adjust sampling "
                f"ranges / acceptance thresholds")
        save_chunk(chunk_dir, ci, valid)
        print(f"  chunk {ci:03d}: {len(valid)}/{len(results)} accepted, "
              f"{dt:.1f}s ({dt/max(len(results),1):.2f}s/solve)")

    print(f"\n  {split}: {total_accepted}/{total_solves} accepted")
    print(f"  chunks saved to {chunk_dir}")


def merge(out_dir: str, split: str) -> None:
    """Merge all chunks of a split into a single .npz."""
    chunk_dir = Path(out_dir) / split
    chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunks found in {chunk_dir}")

    merged, total = {}, 0
    for path in chunks:
        with np.load(path) as d:
            for key in STACKED_KEYS:
                if key not in d:
                    continue
                merged.setdefault(key, []).append(d[key])
            merged.setdefault("R", d["R"])
            merged.setdefault("Z", d["Z"])
            total += d["psi_total"].shape[0]

    out_path = Path(out_dir) / f"{split}.npz"
    arrays = {key: np.concatenate(v) for key, v in merged.items() if isinstance(v, list)}
    arrays["R"] = merged["R"]
    arrays["Z"] = merged["Z"]
    np.savez(out_path, **arrays)

    print(f"\n  merged {len(chunks)} chunks -> {out_path} ({total} samples)")
    for key in sorted(arrays):
        print(f"    {key}: {arrays[key].shape} {arrays[key].dtype}")

    # parameter-range sanity print
    p = arrays["params"]
    print("\n  Parameter statistics:")
    print(f"    Ip:    [{p[:,0].min():.2e}, {p[:,0].max():.2e}] A   (paper [5e4, 4e5])")
    print(f"    paxis: [{p[:,1].min():.1f}, {p[:,1].max():.1f}] Pa (paper [200, 3000])")
    print(f"    fvac:  [{p[:,2].min():.2f}, {p[:,2].max():.2f}]     (paper [0.5, 3.0])")
    x = arrays["x_coords"]
    print(f"    X-pt R: [{x[:,0].min():.3f}, {x[:,0].max():.3f}] m (paper 1.1 +- 0.02)")
    if "config" in arrays and arrays["config"].max() > 0.5:
        print(f"    (SN samples: x_coords up channels are (0,0) placeholders)")
    if p.shape[1] >= 5:
        print(f"    alpha_m: [{p[:,3].min():.3f}, {p[:,3].max():.3f}] (v2 [1.0, 2.0])")
        print(f"    alpha_n: [{p[:,4].min():.3f}, {p[:,4].max():.3f}] (v2 [1.5, 2.5])")
    if "anchor" in arrays:
        a = arrays["anchor"]
        print(f"    anchor R: [{a[:,0].min():.3f}, {a[:,0].max():.3f}] m (v3 [1.2, 1.8])")
        print(f"    anchor Z: [{a[:,1].min():.3f}, {a[:,1].max():.3f}] m (v3 [-0.3, 0.3])")
    if "config" in arrays:
        c = arrays["config"]
        print(f"    config: DN {int((c == 0).sum())} / SN {int((c == 1).sum())}")

    # data_v4 constraint diagnostics (present when --save-constraint-diag)
    if "isoflux_res" in arrays:
        res = np.abs(arrays["isoflux_res"]).max(axis=1)
        core = arrays["axes"][:, 3] - arrays["axes"][:, 2]
        norm = res / np.maximum(core, 1e-30)
        print("\n  Constraint diagnostics (--save-constraint-diag):")
        print(f"    isoflux res |psi(Xpt)-psi(anchor)|/core: mean {norm.mean():.3f}, "
              f"median {np.median(norm):.3f}, p95 {np.percentile(norm, 95):.3f}, "
              f"max {norm.max():.3f} (v3 mean 0.50 / max 2.10)")
        # SN samples carry x_coords up=(0,0) placeholders (n_actual==1): pair
        # against the lower X-point only, or the placeholder inflates the dev
        n_actual = arrays["xpts_actual"].shape[1]
        tgt = arrays["x_coords"][:, :2] if n_actual == 1 \
            else arrays["x_coords"][:, [0, 2]]
        dev = np.hypot(arrays["xpts_actual"][:, :, 0] - tgt[:, 0, None],
                       arrays["xpts_actual"][:, :, 1] - tgt[:, 1, None]).max(axis=1)
        print(f"    X-pt |actual - target| max: mean {dev.mean():.4f} m, "
              f"p95 {np.percentile(dev, 95):.4f} m, max {dev.max():.4f} m")
        print(f"    n_iter: [{arrays['n_iter'].min():.0f}, {arrays['n_iter'].max():.0f}], "
              f"psi_relchange_final max {arrays['psi_relchange_final'].max():.3e} "
              f"(rtol {RTOL})")
        # coil-quadrilateral hull sanity applies only to the 4-coil TestTokamak
        # (COIL_HULL is pinned); MAST (11 coils) uses the machine's own hull
        if arrays["coil_currents"].shape[1] == 4:
            hull = np.asarray(COIL_HULL, dtype=np.float64)
            n_ok = 0
            for i in range(len(arrays["x_coords"])):
                pts = [arrays["x_coords"][i, :2], arrays["x_coords"][i, 2:]]
                if "anchor" in arrays:
                    pts.append(arrays["anchor"][i])
                if all(_inside_with_margin(p, hull, 0.0) for p in pts):
                    n_ok += 1
            print(f"    three points inside coil quadrilateral: {n_ok}/{len(arrays['x_coords'])}")
        else:
            print(f"    coil-quadrilateral hull sanity: skipped "
                  f"({arrays['coil_currents'].shape[1]}-coil machine)")


def main() -> None:
    parser = argparse.ArgumentParser(description="DN dataset generation (arXiv:2608.05555)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="dn_fno_2608/data")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--alpha-sampling", action="store_true",
                        help="sample profile-shape exponents alpha_m/alpha_n "
                             "(data_v2; default off keeps paper shapes 1.0/2.0)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="solve attempts per sample with full parameter "
                             "resampling on rejection (v2 uses 20 for 6000/6000)")
    parser.add_argument("--xpt-jitter", type=float, default=XPT_JITTER,
                        help="X-point jitter half-width in m (paper default 0.02; "
                             "data_v3 uses 0.20)")
    parser.add_argument("--isoflux-sampling", action="store_true",
                        help="sample the isoflux anchor R~U[1.2,1.8] x Z~U[-0.3,0.3] "
                             "and save the per-sample anchor field (data_v3; "
                             "default off keeps the fixed (1.5, 0.0) anchor)")
    # ---- data_v4: feasible-region sampling + physical-validity acceptance ----
    parser.add_argument("--xpt-r0", type=float, default=None,
                        help="X-point reference R (default 1.1; data_v4 uses 1.2; "
                             "MAST spec default 0.7)")
    parser.add_argument("--xpt-z0", type=float, default=None,
                        help="X-point reference |Z| (default 0.6; MAST spec default 1.1)")
    parser.add_argument("--xpt-jitter-z", type=float, default=None,
                        help="X-point |dZ| jitter half-width (default = --xpt-jitter; "
                             "data_v4 uses 0.15 against --xpt-jitter 0.10)")
    parser.add_argument("--anchor-midplane", action="store_true",
                        help="anchor = (R, 0.0), R~U[1.35,1.65] = separatrix outer "
                             "midplane radius (data_v4; default off keeps the v3 "
                             "R x Z anchor sampling)")
    parser.add_argument("--max-isoflux-residual", type=float, default=None,
                        help="reject |psi(Xpt)-psi(anchor)|/core above this "
                             "(data_v4 uses 0.25; default off = no check, v3 path)")
    parser.add_argument("--max-xpt-deviation", type=float, default=None,
                        help="reject when no actual critical X-point is within this "
                             "distance of a target (data_v4 uses 0.10 m; default off)")
    parser.add_argument("--min-anchor-xpt-dist", type=float, default=None,
                        help="reject anchors closer than this to either X-point "
                             "(data_v4 uses 0.15 m; default off)")
    parser.add_argument("--require-wall", action="store_true",
                        help="reject when X-points/anchor fall outside the wall "
                             "(data_v4; default off)")
    parser.add_argument("--coil-margin", type=float, default=COIL_MARGIN,
                        help="min distance of X-pts/anchor to the coil-quadrilateral "
                             "edges (default 0.10 m; data_v4 uses 0.05 — 0.10 rejects "
                             "98% of data_v2 whose X-pt R0=1.1 is 0.10 from the left edge)")
    parser.add_argument("--min-core-depth", type=float, default=None,
                        help="reject psi_axis - psi_bndry < this Wb "
                             "(data_v4 uses 0.005; default off)")
    parser.add_argument("--save-constraint-diag", action="store_true",
                        help="save 8 per-sample constraint-diagnostic fields "
                             "(xpts_actual, o_point, xpt_constraint_res, isoflux_res, "
                             "psi_at_constraints, n_iter, psi_relchange_final; data_v4)")
    parser.add_argument("--merge", action="store_true",
                        help="merge existing chunks of --split into a single npz")
    # ---- data_v5: machine x configuration ----
    parser.add_argument("--config", choices=["dn", "sn"], default="dn",
                        help="equilibrium configuration: dn = double null "
                             "(default), sn = single null (data_v5; single "
                             "X-point + isoflux to the midplane anchor)")
    parser.add_argument("--machine", choices=["test", "mast"], default="test",
                        help="machine geometry: test = TestTokamak (default), "
                             "mast = MAST (data_v5; MAST has no wall, 11 "
                             "control coils)")
    args = parser.parse_args()

    if args.merge:
        merge(args.out_dir, args.split)
    else:
        generate(args.out_dir, args.split, args.n_samples, args.seed,
                 args.chunk_size, args.n_jobs, args.alpha_sampling,
                 args.max_retries, args.xpt_jitter, args.isoflux_sampling,
                 args.xpt_r0, args.xpt_z0, args.xpt_jitter_z,
                 args.anchor_midplane, args.max_isoflux_residual,
                 args.max_xpt_deviation, args.min_anchor_xpt_dist,
                 args.require_wall, args.coil_margin, args.min_core_depth,
                 args.save_constraint_diag, args.config, args.machine)


if __name__ == "__main__":
    main()
