"""2D verification of Solov'ev equilibrium with proper GS Laplacian."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def solovev_psi(R, Z, R0, a, Ip, Beta0, alpha_m=1.0, alpha_n=2.0):
    """Compute Solov'ev psi on 2D grid."""
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    rho = np.sqrt((R - R0)**2 + Z**2) / a
    rho = np.clip(rho, 0.0, None)
    
    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar = core * (1.0 + Beta0 * core**(alpha_n - 1)) / (1.0 + Beta0)
    
    psi_axis_val = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    
    psi = psi_bar * psi_axis_val
    
    psiN = 1.0 - psi_bar
    
    return psi, psiN, psi_axis_val


def compute_modified_laplacian(psi, R, Z):
    """Compute modified Laplacian for GS equation.
    
    Delta*psi = R * d/dR (1/R * dpsi/dR) + d2psi/dZ2
    """
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    
    psi_inner = psi[1:-1, 1:-1]
    R_c = R[1:-1, 1:-1]
    
    dpsi_dR_plus = (psi[2:, 1:-1] - psi[1:-1, 1:-1]) / dR
    dpsi_dR_minus = (psi[1:-1, 1:-1] - psi[:-2, 1:-1]) / dR
    
    R_plus = R[2:, 1:-1]
    R_minus = R[:-2, 1:-1]
    
    term1 = R_plus * dpsi_dR_plus / R_plus
    term2 = R_minus * dpsi_dR_minus / R_minus
    
    d_R_inv_dpsi_dR_dR = (term1 - term2) / (2.0 * dR)
    
    d2psi_dZ2 = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dZ**2)
    
    lap_star_psi = R_c * d_R_inv_dpsi_dR_dR + d2psi_dZ2
    
    return lap_star_psi


def verify_solovev_2d():
    """Verify Solov'ev in 2D with correct GS Laplacian."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    nx, ny = 65, 65
    Rmin, Rmax = R0 - a - 0.15, R0 + a + 0.15
    Zmin, Zmax = -a - 0.15, a + 0.15
    
    R_grid = np.linspace(Rmin, Rmax, nx)
    Z_grid = np.linspace(Zmin, Zmax, ny)
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')
    
    psi, psiN, psi_axis_val = solovev_psi(R, Z, R0, a, Ip, Beta0, alpha_m, alpha_n)
    
    lap_star_psi = compute_modified_laplacian(psi, R, Z)
    
    psiN_inner = psiN[1:-1, 1:-1]
    R_inner = R[1:-1, 1:-1]
    
    jtorshape = (1.0 - psiN_inner**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R_inner * pprime_val + ffprime_val / (mu0 * R_inner)
    
    rhs = mu0 * R_inner * Jtor
    
    residual = lap_star_psi + rhs
    
    print(f"2D Solov'ev Verification:")
    print(f"=" * 60)
    print(f"  Parameters:")
    print(f"    R0 = {R0} m")
    print(f"    a = {a} m")
    print(f"    Ip = {Ip} A")
    print(f"    Beta0 = {Beta0}")
    print(f"    L = {L:.6e}")
    print(f"    psi_axis = {psi_axis_val:.6e} Wb/rad")
    print(f"=" * 60)
    print(f"  GS Residual Statistics:")
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
    
    im1 = axes[1].contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], lap_star_psi, levels=50, cmap="RdBu_r")
    axes[1].set_title("Delta*psi")
    axes[1].set_xlabel("R (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], residual, levels=50, cmap="RdBu_r")
    axes[2].set_title("GS Residual")
    axes[2].set_xlabel("R (m)")
    axes[2].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_2d_verification.png', dpi=150)
    plt.close()
    
    return psi, psiN, lap_star_psi, rhs, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    verify_solovev_2d()