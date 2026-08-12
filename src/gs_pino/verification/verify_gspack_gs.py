"""Verify that gspack's numerical solution satisfies GS equation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import warnings
import contextlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

gspack_dir = Path("d:/D_F/Fusion/AI/PINN/gspack2_TRAE")
if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
    sys.path.insert(0, str(gspack_dir))

from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles import ConstrainBetapIp
from gspack import picard


def compute_gs_residual(psi, R, Z, L, Beta0, alpha_m, alpha_n, psi_axis):
    """Compute GS residual for given psi and parameters."""
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    
    psi_inner = psi[1:-1, 1:-1]
    R_c = R[1:-1, 1:-1]
    
    dpsi_dR_plus = (psi[2:, 1:-1] - psi[1:-1, 1:-1]) / dR
    dpsi_dR_minus = (psi[1:-1, 1:-1] - psi[:-2, 1:-1]) / dR
    
    R_plus = R[2:, 1:-1]
    R_minus = R[:-2, 1:-1]
    
    R_dpsi_dR_plus = R_plus * dpsi_dR_plus
    R_dpsi_dR_minus = R_minus * dpsi_dR_minus
    
    d_R_dpsi_dR_dR = (R_dpsi_dR_plus - R_dpsi_dR_minus) / (2.0 * dR)
    
    d2psi_dZ2 = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dZ**2)
    
    lap_psi = d_R_dpsi_dR_dR / R_c + d2psi_dZ2
    
    mu0 = 4 * np.pi * 1e-7
    
    pb = psi_inner / psi_axis
    pb = np.clip(pb, 1e-30, 1.0)
    
    core = np.clip(1.0 - pb**alpha_m, 0.0, None)
    
    dP_dpsi = (L * Beta0 / R_c) * core**alpha_n / psi_axis
    
    FF_prime = mu0 * L * (1.0 - Beta0) * R_c * core**alpha_n
    
    rhs = (mu0 / R_c) * (dP_dpsi * psi_inner + FF_prime)
    
    residual = -lap_psi - rhs
    
    return residual


def verify_gspack_solution():
    """Verify gspack solution satisfies GS equation."""
    R0 = 1.0
    a = 0.5
    kappa = 1.0
    delta = 0.0
    Ip = 2e5
    betap = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    margin = max(0.15, 0.3 * a)
    Rmin = R0 - a - margin
    Rmax = R0 + a + margin
    Zmax_val = kappa * a + margin
    Zmin = -Zmax_val
    
    eq = FixedBoundaryEquilibrium(
        R0=R0, a=a, kappa=kappa, delta=delta,
        Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax_val,
        nx=65, ny=65, order=2, method="lu",
        fix_bndry_zero=True,
    )
    
    pro = ConstrainBetapIp(
        betap=betap, Ip=Ip, fvac=1.0,
        alpha_m=alpha_m, alpha_n=alpha_n, Raxis=R0,
    )
    
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        _errs = picard.solve(
            eq, pro, constrain=None,
            maxits=50, rtol=5e-3,
            anderson_m=5,
            convergenceInfo=True, verbose=False,
        )
    
    psi = np.asarray(eq.psi(), dtype=np.float32)
    R = np.asarray(eq.R, dtype=np.float32)
    Z = np.asarray(eq.Z, dtype=np.float32)
    psi_axis = float(eq.psi_axis)
    
    L = float(pro.L)
    Beta0 = float(pro.Beta0)
    
    print(f"gspack solution:")
    print(f"  L: {L:.6e}")
    print(f"  Beta0: {Beta0:.6e}")
    print(f"  psi_axis: {psi_axis:.6e}")
    R_axis, Z_axis, _ = eq.magneticAxis()
    print(f"  R_axis: {R_axis:.4f}")
    print(f"  Z_axis: {Z_axis:.4f}")
    
    residual = compute_gs_residual(psi, R, Z, L, Beta0, alpha_m, alpha_n, psi_axis)
    
    print(f"\nGS Residual Statistics:")
    print(f"  Mean: {np.mean(residual):.6e}")
    print(f"  Max: {np.max(residual):.6e}")
    print(f"  Min: {np.min(residual):.6e}")
    print(f"  RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].contourf(R, Z, psi, levels=50, cmap="viridis")
    axes[0].set_title("gspack psi")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], residual, levels=50, cmap="RdBu_r")
    axes[1].set_title("GS Residual")
    axes[1].set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gspack_gs_residual.png', dpi=150)
    plt.close()
    
    return psi, R, Z, residual, L, Beta0, psi_axis


def compute_psi_bar_analytic(R, Z, R_axis, Z_axis, a, L, Beta0, alpha_m, alpha_n):
    """Compute analytic psi_bar using gspack parameters."""
    rho = np.sqrt((R - R_axis)**2 + (Z - Z_axis)**2) / a
    core = np.clip(1.0 - rho**alpha_m, 0.0, None)
    psi_bar_raw = core * (1.0 + Beta0 * core**alpha_n)
    psi_bar_raw = np.where(rho <= 1.0, psi_bar_raw, 0.0)
    psi_bar = psi_bar_raw / (1.0 + Beta0)
    return psi_bar


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    psi, R, Z, residual, L, Beta0, psi_axis = verify_gspack_solution()
    
    R_axis, Z_axis, _ = FixedBoundaryEquilibrium(
        R0=1.0, a=0.5, kappa=1.0, delta=0.0,
        Rmin=0.35, Rmax=1.65, Zmin=-0.65, Zmax=0.65,
        nx=65, ny=65, order=2, method="lu",
        fix_bndry_zero=True,
    ).magneticAxis()
    
    psi_bar_analytic = compute_psi_bar_analytic(R, Z, R_axis, Z_axis, 0.5, L, Beta0, 1.0, 2.0)
    psi_bar_gspack = psi / psi_axis
    
    diff = psi_bar_analytic - psi_bar_gspack
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.contourf(R, Z, diff, levels=50, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax.set_title("Analytic Solov'ev - gspack psi_bar")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    plt.colorbar(im, ax=ax)
    plt.savefig(OUTPUT_DIR / 'solovev_vs_gspack_diff.png', dpi=150)
    plt.close()
    
    print(f"\nSolov'ev vs gspack psi_bar diff:")
    print(f"  Mean: {np.mean(diff):.6e}")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.6e}")