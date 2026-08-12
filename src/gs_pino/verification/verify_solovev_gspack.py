"""Verify Solov'ev equilibrium using gspack's exact GS operator."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

gspack_dir = Path("d:/D_F/Fusion/AI/PINN/gspack2_TRAE")
if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
    sys.path.insert(0, str(gspack_dir))

from gspack.greens import gs_sparse_2nd, MU0


def solovev_psi(R, Z, R0, a, Ip, Beta0, alpha_m=1.0, alpha_n=2.0):
    """Compute Solov'ev psi on 2D grid."""
    mu0 = MU0
    L = Ip / (2 * np.pi * R0)
    
    rho = np.sqrt((R - R0)**2 + Z**2) / a
    rho = np.clip(rho, 0.0, None)
    
    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar = core * (1.0 + Beta0 * core**(alpha_n - 1)) / (1.0 + Beta0)
    
    psi_axis_val = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    
    psi = psi_bar * psi_axis_val
    psiN = 1.0 - psi_bar
    
    return psi, psiN, psi_axis_val


def verify_with_gspack_operator():
    """Verify Solov'ev using gspack's exact GS sparse matrix."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = MU0
    L = Ip / (2 * np.pi * R0)
    
    nx, ny = 65, 65
    Rmin, Rmax = R0 - a - 0.15, R0 + a + 0.15
    Zmin, Zmax = -a - 0.15, a + 0.15
    
    R_grid = np.linspace(Rmin, Rmax, nx)
    Z_grid = np.linspace(Zmin, Zmax, ny)
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')
    
    psi, psiN, psi_axis_val = solovev_psi(R, Z, R0, a, Ip, Beta0, alpha_m, alpha_n)
    
    A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, nx, ny)
    
    lap_star_psi = A @ psi.ravel()
    lap_star_psi = lap_star_psi.reshape(nx, ny)
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R * pprime_val + ffprime_val / (mu0 * R)
    
    rhs = -mu0 * R * Jtor
    
    lap_star_psi_inner = lap_star_psi[1:-1, 1:-1]
    rhs_inner = rhs[1:-1, 1:-1]
    
    residual = lap_star_psi_inner - rhs_inner
    
    print(f"Solov'ev Verification with gspack GS Operator:")
    print(f"=" * 60)
    print(f"  Parameters:")
    print(f"    R0 = {R0} m")
    print(f"    a = {a} m")
    print(f"    Ip = {Ip} A")
    print(f"    Beta0 = {Beta0}")
    print(f"    L = {L:.6e}")
    print(f"    psi_axis = {psi_axis_val:.6e} Wb/rad")
    print(f"=" * 60)
    print(f"  GS Residual Statistics (interior):")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].contourf(R, Z, psi, levels=50, cmap="viridis")
    axes[0].set_title("Solov'ev psi")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(R, Z, lap_star_psi, levels=50, cmap="RdBu_r")
    axes[1].set_title("Delta*psi (gspack)")
    axes[1].set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], residual, levels=50, cmap="RdBu_r")
    axes[2].set_title("GS Residual")
    axes[2].set_xlabel("R (m)")
    axes[2].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    # 使用仓库根目录下的 outputs/，与从任何 CWD 运行一致
    out_dir = Path(__file__).resolve().parents[3] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'solovev_gspack_operator.png', dpi=150)
    plt.close()

    return psi, psiN, lap_star_psi, rhs, residual


if __name__ == '__main__':
    verify_with_gspack_operator()