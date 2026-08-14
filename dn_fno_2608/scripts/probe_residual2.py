"""Probe 2: norm breakdown of the GS residual terms on one truth sample.

Prints ||lap*psi||, ||mu0 R^2 p'||, ||FF'||, ||RHS|| over several masks and
tests a few more RHS conventions against the paper's 2.29 baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import freegs
from freegs import boundary, control, critical, jtor

MU0 = 4.0 * np.pi * 1e-7
RMIN, RMAX, ZMIN, ZMAX = 0.1, 2.0, -2.0, 2.0
ISOFLUX_REF = (1.5, 0.0)
GAMMA, RTOL, MAXITS = 1e-12, 1e-3, 50
DATA = Path(__file__).resolve().parents[1] / "data" / "test.npz"
SAMPLE = 0


def lap_star(psi, R, Z):
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    d2r = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dR**2
    dr = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    d2z = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dZ**2
    return d2r - dr / (R[1:-1, 1:-1] + 1e-8) + d2z


def main() -> None:
    with np.load(DATA) as d:
        Ip, paxis, fvac = d["params"][SAMPLE]
        lo = tuple(d["x_coords"][SAMPLE][:2])
        up = tuple(d["x_coords"][SAMPLE][2:])
        R, Z = d["R"], d["Z"]

    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=RMIN, Rmax=RMAX,
                            Zmin=ZMIN, Zmax=ZMAX, nx=65, ny=65,
                            boundary=boundary.freeBoundaryHagenow)
    profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac)
    constrain = control.constrain(
        xpoints=[lo, up],
        isoflux=[(*lo, *ISOFLUX_REF), (*up, *ISOFLUX_REF)],
        gamma=GAMMA)
    freegs.solve(eq, profiles, constrain, rtol=RTOL, maxits=MAXITS, show=False)

    psi = eq.psi()
    psi_a, psi_b = eq.psi_axis, eq.psi_bndry
    dpsi = psi_b - psi_a
    opt, xpt = critical.find_critical(R, Z, psi)
    n_freegs = (psi - psi_a) / dpsi
    dp = profiles.pprime(n_freegs)
    ff = profiles.ffprime(n_freegs)
    lap = lap_star(psi, R, Z)
    Rc = R[1:-1, 1:-1]
    mu0r2 = MU0 * Rc**2
    rhs = -mu0r2 * dp[1:-1, 1:-1] - ff[1:-1, 1:-1]

    print(f"psi_axis={psi_a:.4f} psi_bndry={psi_b:.4f} dpsi={dpsi:.4f}")

    # masks: core_mask + dilations (count points with |psi_n| <= 1+eps)
    masks = {"core_mask": critical.core_mask(R, Z, psi, opt, xpt)}
    for eps in (0.0, 0.05, 0.10, 0.25):
        masks[f"psin<=1+{eps}"] = (n_freegs <= 1.0 + eps) & (n_freegs >= -eps)
    for eps in (0.05, 0.25):
        masks[f"psin<=1+{eps} (no lower)"] = (n_freegs <= 1.0 + eps)
    masks["all_interior"] = np.ones_like(psi, dtype=bool)

    for tag, mfull in masks.items():
        core = mfull[1:-1, 1:-1].astype(bool)
        if core.sum() < 10:
            continue
        n = core.sum()
        n_lap = np.linalg.norm(lap[core])
        n_p = np.linalg.norm((mu0r2 * dp[1:-1, 1:-1])[core])
        n_ff = np.linalg.norm(ff[1:-1, 1:-1][core])
        n_rhs = np.linalg.norm(rhs[core])
        r = n_lap * 0  # placeholder
        r_v0 = np.linalg.norm((lap - rhs)[core]) / n_rhs
        print(f"mask={tag:22s} npts={n:5d} ||lap||={n_lap:10.4f} "
              f"||mu0R2p'||={n_p:10.4f} ||FF'||={n_ff:10.4f} ||RHS||={n_rhs:10.4f} "
              f"R_V0={r_v0:8.4f}")

    # sign-flip and normalization variants on core_mask
    core = (critical.core_mask(R, Z, psi, opt, xpt)[1:-1, 1:-1]).astype(bool)
    variants = {
        "RHS +FF' (sign flip)": -mu0r2 * dp[1:-1, 1:-1] + ff[1:-1, 1:-1],
        "RHS -FF' only": -ff[1:-1, 1:-1],
        "RHS -mu0R2p' only": -mu0r2 * dp[1:-1, 1:-1],
        "RHS -mu0R2p'-FF' (V0)": rhs,
        "RHS on psi/psi_a": -mu0r2 * profiles.pprime(psi / psi_a)[1:-1, 1:-1]
                            - profiles.ffprime(psi / psi_a)[1:-1, 1:-1],
        "RHS on psi_n=psi/psi_b": -mu0r2 * profiles.pprime(psi / psi_b)[1:-1, 1:-1]
                                   - profiles.ffprime(psi / psi_b)[1:-1, 1:-1],
        "RHS p'/dpsi only": -mu0r2 * (dp / dpsi)[1:-1, 1:-1] - ff[1:-1, 1:-1],
        "RHS ff'/dpsi only": -mu0r2 * dp[1:-1, 1:-1] - (ff / dpsi)[1:-1, 1:-1],
    }
    for tag, rhs_v in variants.items():
        r = np.linalg.norm((lap - rhs_v)[core]) / np.linalg.norm(rhs_v[core])
        print(f"{tag:26s} R={r:8.4f}  (x100={r*100:7.2f})")


if __name__ == "__main__":
    main()
