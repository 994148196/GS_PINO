"""Final verification: Solov'ev vs gspack numerical solution."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import warnings
import contextlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def verify_solovev_with_gspack():
    """Compare Solov'ev analytic solution with gspack numerical solution."""
    gspack_dir = Path("d:/D_F/Fusion/AI/PINN/gspack2_TRAE")
    if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
        sys.path.insert(0, str(gspack_dir))
    
    from gspack.equilibrium import FixedBoundaryEquilibrium
    from gspack.profiles import ConstrainBetapIp
    from gspack import picard
    from gspack.greens import MU0
    
    R0 = 1.0
    a = 0.5
    kappa = 1.0
    delta = 0.0
    Ip = 2e5
    betap = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = MU0
    L = Ip / (2 * np.pi * R0)
    
    margin = max(0.15, 0.3 * a)
    eq = FixedBoundaryEquilibrium(
        R0=R0, a=a, kappa=kappa, delta=delta,
        Rmin=R0 - a - margin, Rmax=R0 + a + margin,
        Zmin=-kappa * a - margin, Zmax=kappa * a + margin,
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
    
    psi_gspack = np.asarray(eq.psi(), dtype=np.float32)
    psi_axis_gspack = float(eq.psi_axis)
    R = np.asarray(eq.R, dtype=np.float32)
    Z = np.asarray(eq.Z, dtype=np.float32)
    
    L_gspack = float(pro.L)
    Beta0_gspack = float(pro.Beta0)
    
    R_axis, Z_axis, _ = eq.magneticAxis()
    
    print(f"gspack Solution:")
    print(f"=" * 60)
    print(f"  L: {L_gspack:.6e}")
    print(f"  Beta0: {Beta0_gspack:.6e}")
    print(f"  psi_axis: {psi_axis_gspack:.6e}")
    print(f"  R_axis: {R_axis:.4f}")
    print(f"  Z_axis: {Z_axis:.4f}")
    print(f"=" * 60)
    
    rho = np.sqrt((R - R_axis)**2 + Z**2) / a
    rho = np.clip(rho, 0.0, None)
    
    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar_analytic = core * (1.0 + Beta0_gspack * core**(alpha_n - 1)) / (1.0 + Beta0_gspack)
    
    psi_bar_gspack = psi_gspack / psi_axis_gspack
    
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
    
    im2 = axes[2].contourf(R, Z, diff, levels=50, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    axes[2].set_title("Solov'ev - gspack Difference")
    axes[2].set_xlabel("R (m)")
    axes[2].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_vs_gspack_final.png', dpi=150)
    plt.close()
    
    return psi_bar_analytic, psi_bar_gspack, diff


def verify_gspack_satisfies_gs():
    """Verify gspack's solution satisfies GS equation."""
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
    
    A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, 65, 65)
    lap_star_psi = A @ psi.ravel()
    lap_star_psi = lap_star_psi.reshape(65, 65)
    
    psi_axis = float(eq.psi_axis)
    psiN = psi / psi_axis
    
    L = float(pro.L)
    Beta0 = float(pro.Beta0)
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R * pprime_val + ffprime_val / (mu0 * R)
    
    rhs = -mu0 * R * Jtor
    
    lap_star_inner = lap_star_psi[1:-1, 1:-1]
    rhs_inner = rhs[1:-1, 1:-1]
    
    residual = lap_star_inner - rhs_inner
    
    print(f"\ngspack GS Equation Verification:")
    print(f"=" * 60)
    print(f"  Residual Statistics (interior):")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"=" * 60)
    
    return psi, R, Z, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    psi_bar_analytic, psi_bar_gspack, diff = verify_solovev_with_gspack()
    psi, R, Z, residual = verify_gspack_satisfies_gs()