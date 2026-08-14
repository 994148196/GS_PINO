"""Probe: which RHS / psi-norm / mask convention reproduces the paper's
GS residual baseline of 2.29 on a freegs TRUTH field (ours measures 0.012)?

Re-solves one test sample with the exact generation config, then evaluates
R = || Delta* psi - RHS ||_2,Omega / || RHS ||_2,Omega under many variants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for gs_pino

import freegs
from freegs import boundary, control, critical, jtor

MU0 = 4.0 * np.pi * 1e-7

# same solver config as generate_dn_dataset.py
RMIN, RMAX, ZMIN, ZMAX = 0.1, 2.0, -2.0, 2.0
NX = NY = 65
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


def ratio(lap, R, dpdpsi, fdFdpsi, mask_full):
    R_c = R[1:-1, 1:-1]
    rhs = -MU0 * R_c**2 * dpdpsi[1:-1, 1:-1] - fdFdpsi[1:-1, 1:-1]
    for tag, mask in mask_full:
        core = mask[1:-1, 1:-1] > 0.5
        if core.sum() < 10:
            continue
        r = np.linalg.norm((lap - rhs)[core]) / np.linalg.norm(rhs[core])
        print(f"    mask={tag:10s} R={r:8.4f}  (R*100={r*100:7.2f}%)")
    return rhs


def main() -> None:
    with np.load(DATA) as d:
        Ip, paxis, fvac = d["params"][SAMPLE]
        lo = tuple(d["x_coords"][SAMPLE][:2])
        up = tuple(d["x_coords"][SAMPLE][2:])
        R, Z = d["R"], d["Z"]
    print(f"sample {SAMPLE}: Ip={Ip:.1f} paxis={paxis:.1f} fvac={fvac:.3f} "
          f"lo={tuple(round(v,4) for v in lo)} up={tuple(round(v,4) for v in up)}")

    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=RMIN, Rmax=RMAX,
                            Zmin=ZMIN, Zmax=ZMAX, nx=NX, ny=NY,
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
    print(f"  solved: psi_axis={psi_a:.4f} psi_bndry={psi_b:.4f} dpsi={dpsi:.4f} "
          f"nxpt={len(xpt)}")

    masks = {
        "core_mask": critical.core_mask(R, Z, psi, opt, xpt),
        "full_dom": np.ones_like(psi),
        "psin_0_1": ((psi - psi_a) / dpsi > 0) & ((psi - psi_a) / dpsi < 1),
    }
    mask_list = list(masks.items())

    lap = lap_star(psi, R, Z)
    print(f"  norms: ||lap*psi||={np.linalg.norm(lap):.4f}")

    # --- variants of the p'/FF' evaluation argument ---
    n_freegs = (psi - psi_a) / dpsi            # 0 at axis, 1 at bndry (ours)
    variants = {
        "V0 psi_norm freegs (ours)": n_freegs,
        "V1 psi_phys [Wb]": psi,
        "V2 z-scored (train stats)": None,      # filled below
        "V3 psi_norm inverted (bndry=0)": 1.0 - n_freegs,
        "V4 psi_norm from psi_plasma": None,    # filled below
    }
    with np.load(DATA) as d:
        variants["V2 z-scored (train stats)"] = \
            (psi - float(d["psi_total"].mean())) / float(d["psi_total"].std())
    psi_pl = eq.plasma_psi
    variants["V4 psi_norm from psi_plasma"] = \
        (psi_pl - psi_a) / (psi_b - psi_a)

    for tag, arg in variants.items():
        print(f"\n[{tag}]")
        dp = profiles.pprime(arg)
        ff = profiles.ffprime(arg)
        ratio(lap, R, dp, ff, mask_list)
        # also test chain-rule and unit variants for the freegs-normalized case
        if tag.startswith("V0"):
            print("  -- V0 + chain rule /dpsi:")
            ratio(lap, R, dp / dpsi, ff / dpsi, mask_list)
            print("  -- V0 without mu0 R^2 on p':")
            R_c = R[1:-1, 1:-1]
            rhs = -dp[1:-1, 1:-1] - ff[1:-1, 1:-1]
            core = masks["core_mask"][1:-1, 1:-1] > 0.5
            print(f"    mask=core_mask R={np.linalg.norm((lap-rhs)[core])/np.linalg.norm(rhs[core]):8.4f}")
            print("  -- V0 times dpsi (both terms):")
            rhs = -MU0 * R_c**2 * (dp / 1.0)[1:-1, 1:-1] * dpsi - ff[1:-1, 1:-1] * dpsi
            print(f"    mask=core_mask R={np.linalg.norm((lap-rhs)[core])/np.linalg.norm(rhs[core]):8.4f}")

    # --- dataset-stored fields (should equal V0) ---
    with np.load(DATA) as d:
        dp_st = d["dpdpsi"][SAMPLE]
        ff_st = d["FdFdpsi"][SAMPLE]
        mask_st = d["mask"][SAMPLE]
    print("\n[dataset-stored dpdpsi/FdFdpsi (ours)]")
    rhs = ratio(lap, R, dp_st, ff_st, [("stored_mask", mask_st)])
    print(f"  max|dp_st - dp_V0|={np.abs(dp_st - profiles.pprime(n_freegs)).max():.3e} "
          f"max|ff_st - ff_V0|={np.abs(ff_st - profiles.ffprime(n_freegs)).max():.3e}")


if __name__ == "__main__":
    main()
