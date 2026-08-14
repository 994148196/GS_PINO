"""Smoke test (Phase 0.2): solve ONE double-null equilibrium per paper 2608.05555.

Based on freegs 06-xpoints.py pattern (TestTokamak + double X-point constrain),
with the paper's stated solver config: Picard, gamma=1e-12, isoflux to a fixed
outboard midplane reference, maxits=50, rtol=1e-3.

Checks:
  1. solve converges within maxits
  2. find_critical finds >= 2 distinct X-points
  3. |Ip_sol - Ip_tgt| / Ip_tgt <= 10%
  4. psi_bndry = 0.5*(psi_lo^X + psi_up^X) computable
  5. profiles.pprime/ffprime produce dpdpsi / FdFdpsi fields
  6. solve wall time (paper reference: ~1.77 s median)

Usage: python smoke_dn_solve.py [--with-isoflux] [--no-isoflux]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import freegs
from freegs import boundary, control, critical, jtor

# Paper settings
RMIN, RMAX = 0.1, 2.0
ZMIN, ZMAX = -2.0, 2.0
NX, NY = 65, 65
XPT_R, XPT_Z = 1.1, 0.6  # reference X-point targets (upper/lower symmetric)
JITTER = 0.02
ISOFLUX_REF = (1.5, 0.0)  # fixed outboard midplane reference (gauge anchor)
GAMMA = 1e-12
RTOL, MAXITS = 1e-3, 50

# Representative sample from paper Fig. 1
PAXIS, IP, FVAC = 1200.0, 2.0e5, 1.75


def build_and_solve(paxis, Ip, fvac, with_isoflux=True, verbose=True):
    t0 = time.perf_counter()

    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(
        tokamak=tokamak,
        Rmin=RMIN,
        Rmax=RMAX,
        Zmin=ZMIN,
        Zmax=ZMAX,
        nx=NX,
        ny=NY,
        boundary=boundary.freeBoundaryHagenow,
    )

    # Profile shapes fixed by the FREEGS setup (paper: "shapes fixed by the
    # FREEGS setup used here"); only scalar amplitudes vary.
    profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac)

    # Symmetric X-point jitter: lower (1.1+dR, -(0.6+dZ)), upper (1.1+dR, +(0.6+dZ))
    rng = np.random.default_rng(0)
    dR = float(rng.uniform(-JITTER, JITTER))
    dZ = float(rng.uniform(-JITTER, JITTER))
    lo = (XPT_R + dR, -(XPT_Z + dZ))
    up = (XPT_R + dR, +(XPT_Z + dZ))

    if with_isoflux:
        constrain = control.constrain(
            xpoints=[lo, up],
            isoflux=[(*lo, *ISOFLUX_REF), (*up, *ISOFLUX_REF)],
            gamma=GAMMA,
        )
    else:
        constrain = control.constrain(xpoints=[lo, up], gamma=GAMMA)

    freegs.solve(eq, profiles, constrain, rtol=RTOL, maxits=MAXITS, show=False)
    dt = time.perf_counter() - t0

    psi = eq.psi()
    opt, xpt = critical.find_critical(eq.R, eq.Z, psi)

    if verbose:
        print(f"X-point targets: lo={lo}, up={up}")
        print(f"solve time: {dt:.3f} s")
        print(f"O-points: {len(opt)}, X-points: {len(xpt)}")
        for i, (r, z, p) in enumerate(xpt):
            print(f"  xpt[{i}]: R={r:.4f}, Z={z:.4f}, psi={p:.6e}")
        for i, (r, z, p) in enumerate(opt):
            print(f"  opt[{i}]: R={r:.4f}, Z={z:.4f}, psi={p:.6e}")
        Ip_sol = eq.plasmaCurrent()
        print(f"Ip_tgt={Ip:.4e}, Ip_sol={Ip_sol:.4e}, rel err={abs(Ip_sol-Ip)/Ip*100:.3f}%")
        print(f"eq.psi_bndry={eq.psi_bndry:.6e}")
        if len(xpt) >= 2:
            psi_bndry_avg = 0.5 * (xpt[0][2] + xpt[1][2])
            print(f"0.5*(psi_lo+psi_up)={psi_bndry_avg:.6e}, "
                  f"|diff|={abs(psi_bndry_avg - eq.psi_bndry):.3e}")
        for label, coil in tokamak.coils:
            print(f"  coil {label}: I={coil.current:.4e} A")

    return eq, profiles, tokamak, dt, len(xpt)


def check_profiles(eq, profiles):
    """Build dpdpsi / FdFdpsi fields from the constrained profiles."""
    psi = eq.psi()
    psi_axis = eq.psi_axis
    psi_bndry = eq.psi_bndry
    psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
    dpdpsi = profiles.pprime(psi_norm)
    fdFdpsi = profiles.ffprime(psi_norm)
    print(f"\ndpdpsi: shape={dpdpsi.shape}, range=[{dpdpsi.min():.4e}, {dpdpsi.max():.4e}]")
    print(f"FdFdpsi: shape={fdFdpsi.shape}, range=[{fdFdpsi.min():.4e}, {fdFdpsi.max():.4e}]")
    print(f"L={profiles.L:.6e}, Beta0={profiles.Beta0:.6e}, fvac={profiles.fvac():.4f}")
    return dpdpsi, fdFdpsi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-isoflux", action="store_true",
                    help="run without isoflux constraints (06-xpoints.py pure mode)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"DN smoke solve: paxis={PAXIS}, Ip={IP}, fvac={FVAC}, "
          f"isoflux={'off' if args.no_isoflux else 'on'}")
    print("=" * 70)
    eq, profiles, tokamak, dt, n_xpt = build_and_solve(
        PAXIS, IP, FVAC, with_isoflux=not args.no_isoflux)

    ok = True
    if n_xpt < 2:
        print("FAIL: fewer than 2 X-points found")
        ok = False
    Ip_err = abs(eq.plasmaCurrent() - IP) / IP
    if Ip_err > 0.10:
        print(f"FAIL: Ip relative error {Ip_err*100:.2f}% > 10%")
        ok = False
    check_profiles(eq, profiles)
    print("\nSMOKE RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
