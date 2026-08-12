"""Verify gspack solution is self-consistent with GS equation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import warnings
import contextlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def verify_gspack_self_consistent():
    """Verify gspack solution satisfies GS equation using gspack's own functions."""
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
    psiN = np.clip(psiN, 0.0, 1.0)
    
    L = float(pro.L)
    Beta0 = float(pro.Beta0)
    Raxis = float(pro.Raxis)
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    Jtor = L * (Beta0 * R / Raxis + (1.0 - Beta0) * Raxis / R) * jtorshape
    
    A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, 65, 65)
    
    lap_star_psi = A @ psi.ravel()
    lap_star_psi = lap_star_psi.reshape(65, 65)
    
    rhs = mu0 * R * Jtor
    
    residual = lap_star_psi + rhs
    
    interior_mask = (psiN > 0.0) & (psiN < 1.0)
    
    residual_interior = residual[interior_mask]
    
    print(f"gspack Self-Consistency Check:")
    print(f"=" * 60)
    print(f"  Parameters:")
    print(f"    L = {L:.6e}")
    print(f"    Beta0 = {Beta0:.6e}")
    print(f"    psi_axis = {psi_axis:.6e}")
    print(f"    psi_bndry = {psi_bndry:.6e}")
    print(f"    dpsi = {dpsi:.6e}")
    print(f"=" * 60)
    print(f"  GS Residual Statistics (interior only):")
    print(f"    Mean: {np.mean(residual_interior):.6e}")
    print(f"    Max: {np.max(residual_interior):.6e}")
    print(f"    Min: {np.min(residual_interior):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual_interior**2)):.6e}")
    print(f"    |Mean| / |rhs_mean|: {np.abs(np.mean(residual_interior)) / (np.mean(np.abs(rhs[interior_mask])) + 1e-30):.6e}")
    print(f"=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].contourf(R, Z, psiN, levels=20, cmap="viridis")
    axes[0].set_title("psiN")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(R, Z, lap_star_psi, levels=50, cmap="RdBu_r")
    axes[1].set_title("A·psi (-Delta*psi)")
    axes[1].set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(R, Z, rhs, levels=50, cmap="RdBu_r")
    axes[2].set_title("mu0*R*Jtor")
    axes[2].set_xlabel("R (m)")
    axes[2].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gspack_consistency_check.png', dpi=150)
    plt.close()
    
    return psi, psiN, lap_star_psi, rhs, residual


def benchmark_with_different_grids():
    """Benchmark gspack with different grid resolutions."""
    gspack_dir = Path("d:/D_F/Fusion/AI/PINN/gspack2_TRAE")
    if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
        sys.path.insert(0, str(gspack_dir))
    
    from gspack.equilibrium import FixedBoundaryEquilibrium
    from gspack.profiles import ConstrainBetapIp
    from gspack import picard
    from gspack.greens import MU0, gs_sparse_2nd
    
    mu0 = MU0
    
    R0 = 1.0
    a = 0.5
    kappa = 1.0
    delta = 0.0
    Ip = 2e5
    betap = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    margin = max(0.15, 0.3 * a)
    
    grid_sizes = [33, 65, 129]
    
    print(f"\ngspack Grid Convergence Test:")
    print(f"=" * 60)
    
    for nx in grid_sizes:
        ny = nx
        Rmin = R0 - a - margin
        Rmax = R0 + a + margin
        Zmin = -kappa * a - margin
        Zmax = kappa * a + margin
        
        eq = FixedBoundaryEquilibrium(
            R0=R0, a=a, kappa=kappa, delta=delta,
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nx=nx, ny=ny, order=2, method="lu",
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
                maxits=100, rtol=1e-5,
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
        psiN = np.clip(psiN, 0.0, 1.0)
        
        L = float(pro.L)
        Beta0 = float(pro.Beta0)
        Raxis = float(pro.Raxis)
        
        jtorshape = (1.0 - psiN**alpha_m)**alpha_n
        Jtor = L * (Beta0 * R / Raxis + (1.0 - Beta0) * Raxis / R) * jtorshape
        
        A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, nx, ny)
        lap_star_psi = A @ psi.ravel()
        lap_star_psi = lap_star_psi.reshape(nx, ny)
        
        rhs = mu0 * R * Jtor
        residual = lap_star_psi + rhs
        
        interior_mask = (psiN > 0.01) & (psiN < 0.99)
        rmse = np.sqrt(np.mean(residual[interior_mask]**2))
        
        R_axis, Z_axis, _ = eq.magneticAxis()
        
        print(f"  Grid {nx}x{ny}:")
        print(f"    RMSE: {rmse:.6e}")
        print(f"    R_axis: {R_axis:.4f}")
        print(f"    psi_axis: {psi_axis:.6e}")
    
    print(f"=" * 60)


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    verify_gspack_self_consistent()
    benchmark_with_different_grids()