"""Verify Solov'ev equilibrium using gspack's canonical Jtor form."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import warnings
import contextlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def verify_solovev_canonical():
    """Verify Solov'ev with correct Jtor formula."""
    gspack_dir = Path("d:/D_F/Fusion/AI/PINN/gspack2_TRAE")
    if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
        sys.path.insert(0, str(gspack_dir))
    
    from gspack.equilibrium import FixedBoundaryEquilibrium
    from gspack.profiles import ConstrainBetapIp
    from gspack import picard
    from gspack.greens import MU0, gs_sparse_2nd
    
    R0 = 1.0
    a = 0.5
    kappa = 1.0
    delta = 0.0
    Ip = 2e5
    betap = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = MU0
    
    margin = max(0.15, 0.3 * a)
    Rmin = R0 - a - margin
    Rmax = R0 + a + margin
    Zmin = -kappa * a - margin
    Zmax = kappa * a + margin
    
    eq = FixedBoundaryEquilibrium(
        R0=R0, a=a, kappa=kappa, delta=delta,
        Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
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
    psi_bndry = float(eq.psi_bndry)
    dpsi = psi_bndry - psi_axis
    
    psiN = (psi - psi_axis) / dpsi
    
    L = float(pro.L)
    Beta0 = float(pro.Beta0)
    Raxis = float(pro.Raxis)
    
    R_axis, Z_axis, _ = eq.magneticAxis()
    
    print(f"gspack Solution:")
    print(f"=" * 60)
    print(f"  L: {L:.6e}")
    print(f"  Beta0: {Beta0:.6e}")
    print(f"  Raxis (input): {Raxis:.4f}")
    print(f"  magnetic_axis R: {R_axis:.4f}")
    print(f"  magnetic_axis Z: {Z_axis:.4f}")
    print(f"  psi_axis: {psi_axis:.6e}")
    print(f"  psi_bndry: {psi_bndry:.6e}")
    print(f"  dpsi: {dpsi:.6e}")
    print(f"=" * 60)
    
    jtorshape = (1.0 - np.clip(psiN, 0.0, 1.0)**alpha_m)**alpha_n
    
    Jtor_gspack = L * (Beta0 * R / Raxis + (1.0 - Beta0) * Raxis / R) * jtorshape
    
    rhs = -mu0 * R * Jtor_gspack
    
    A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, 65, 65)
    lap_star_psi = A @ psi.ravel()
    lap_star_psi = lap_star_psi.reshape(65, 65)
    
    lap_star_inner = lap_star_psi[1:-1, 1:-1]
    rhs_inner = rhs[1:-1, 1:-1]
    
    residual = lap_star_inner - rhs_inner
    
    print(f"\ngspack GS Equation Verification (canonical Jtor):")
    print(f"=" * 60)
    print(f"  Residual Statistics (interior):")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"=" * 60)
    
    rho = np.sqrt((R - R_axis)**2 + Z**2) / a
    rho = np.clip(rho, 0.0, None)
    
    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar_analytic = core * (1.0 + Beta0 * core**(alpha_n - 1)) / (1.0 + Beta0)
    
    psi_bar_gspack = psi / psi_axis
    
    diff = psi_bar_analytic - psi_bar_gspack
    
    print(f"\nSolov'ev vs gspack Comparison:")
    print(f"=" * 60)
    print(f"  psi_bar diff statistics:")
    print(f"    Mean: {np.mean(diff):.6e}")
    print(f"    Max: {np.max(diff):.6e}")
    print(f"    Min: {np.min(diff):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(diff**2)):.6e}")
    print(f"=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].contourf(R, Z, psi_bar_analytic, levels=50, cmap="viridis")
    axes[0].set_title("Analytic Solov'ev psi_bar")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].scatter(R_axis, Z_axis, c="red", s=50, marker="x")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(R, Z, psi_bar_gspack, levels=50, cmap="viridis")
    axes[1].set_title("gspack psi_bar")
    axes[1].set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].scatter(R_axis, Z_axis, c="red", s=50, marker="x")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(R, Z, diff, levels=50, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    axes[2].set_title("Solov'ev - gspack Difference")
    axes[2].set_xlabel("R (m)")
    axes[2].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_canonical_verification.png', dpi=150)
    plt.close()
    
    return psi, psiN, Jtor_gspack, residual, diff


def verify_analytic_solovev():
    """Verify Solov'ev analytically using correct Jtor form."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    Raxis = R0
    
    nr = 500
    rho = np.linspace(1e-6, 0.999, nr)
    
    psi_bar = (1.0 - rho**2) * (1.0 + Beta0 * (1.0 - rho**2)**(alpha_n - 1)) / (1.0 + Beta0)
    
    psi_axis_val = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    psi = psi_bar * psi_axis_val
    
    dpsi = -psi_axis_val
    
    psiN = (psi - psi_axis_val) / dpsi
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    Jtor = L * (Beta0 * R0 / Raxis + (1.0 - Beta0) * Raxis / R0) * jtorshape
    
    rhs = mu0 * R0 * Jtor
    
    x = 1.0 - rho**2
    
    dpsi_bar_dx = (1.0 + Beta0 * alpha_n * x**(alpha_n - 1)) / (1.0 + Beta0)
    d2psi_bar_dx2 = -(Beta0 * alpha_n * (alpha_n - 1) * x**(alpha_n - 2)) / (1.0 + Beta0)
    
    dpsi_bar_drho = dpsi_bar_dx * (-2.0 * rho)
    d2psi_bar_drho2 = d2psi_bar_dx2 * (4.0 * rho**2) + dpsi_bar_dx * (-2.0)
    
    lap_psi_bar_rho = (1.0 / rho) * dpsi_bar_drho + d2psi_bar_drho2
    lap_psi_r0 = lap_psi_bar_rho * psi_axis_val / a**2
    
    residual = lap_psi_r0 + rhs
    
    print(f"\nAnalytic Solov'ev Verification (canonical Jtor):")
    print(f"=" * 60)
    print(f"  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"=" * 60)
    
    return psi, psiN, Jtor, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    psi, psiN, Jtor_gspack, residual_gspack, diff = verify_solovev_canonical()
    psi_analytic, psiN_analytic, Jtor_analytic, residual_analytic = verify_analytic_solovev()