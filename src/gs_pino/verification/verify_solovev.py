"""Direct verification of generalized Solov'ev equilibrium.

This script verifies that the analytical Solov'ev solution satisfies
the Grad-Shafranov equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def solovev_psi_bar(rho, n=2.0, betap=0.5):
    """Compute Solov'ev normalized psi."""
    x = np.clip(1.0 - rho**2, 0.0, None)
    return x * (1.0 + betap * x**(n - 1)) / (1.0 + betap)


def verify_solovev_normalized():
    """Verify Solov'ev in normalized coordinates."""
    betap = 0.5
    n = 2.0
    
    nr = 1000
    rho = np.linspace(1e-6, 0.999, nr)
    drho = rho[1] - rho[0]
    
    psi_bar = solovev_psi_bar(rho, n, betap)
    
    dpsi_drho = np.gradient(psi_bar, drho)
    d2psi_drho2 = np.gradient(dpsi_drho, drho)
    
    lap_psi_bar = (1.0 / rho) * dpsi_drho + d2psi_drho2
    
    x = np.clip(1.0 - psi_bar, 1e-30, None)
    
    dP_dpsi_bar = -betap * n * x ** (n - 1)
    
    FF_prime_bar = -betap * (1.0 - x)
    
    rhs = -dP_dpsi_bar * psi_bar - FF_prime_bar
    
    residual = lap_psi_bar - rhs
    
    print(f"Solov'ev Normalized Verification:")
    print(f"  Parameters: betap={betap}, n={n}")
    print(f"\n  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(rho, psi_bar, 'b-')
    axes[0, 0].set_title('psi_bar')
    axes[0, 0].set_xlabel('rho')
    
    axes[0, 1].plot(rho, lap_psi_bar, 'r-')
    axes[0, 1].set_title('Laplacian of psi_bar')
    axes[0, 1].set_xlabel('rho')
    
    axes[1, 0].plot(rho, rhs, 'g-')
    axes[1, 0].set_title('RHS (normalized)')
    axes[1, 0].set_xlabel('rho')
    
    axes[1, 1].plot(rho, residual, 'k-')
    axes[1, 1].set_title('Residual')
    axes[1, 1].set_xlabel('rho')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_normalized_verification.png', dpi=150)
    plt.close()
    
    return psi_bar, lap_psi_bar, rhs, residual


def verify_solovev_analytic_derivatives():
    """Verify using analytical derivatives."""
    betap = 0.5
    n = 2.0
    
    nr = 1000
    rho = np.linspace(1e-6, 0.999, nr)
    
    psi_bar = solovev_psi_bar(rho, n, betap)
    
    x = np.clip(1.0 - rho**2, 0.0, None)
    
    dpsi_dx = (1.0 + betap * n * x**(n - 1)) / (1.0 + betap)
    d2psi_dx2 = (betap * n * (n - 1) * x**(n - 2)) / (1.0 + betap)
    
    dpsi_drho = dpsi_dx * (-2.0 * rho)
    d2psi_drho2 = d2psi_dx2 * (4.0 * rho**2) + dpsi_dx * (-2.0)
    
    lap_psi_bar = (1.0 / rho) * dpsi_drho + d2psi_drho2
    
    x_psi = np.clip(1.0 - psi_bar, 1e-30, None)
    
    dP_dpsi_bar = -betap * n * x_psi ** (n - 1)
    
    FF_prime_bar = -betap * (1.0 - x_psi)
    
    rhs = -dP_dpsi_bar * psi_bar - FF_prime_bar
    
    residual = lap_psi_bar - rhs
    
    print(f"\nSolov'ev Analytical Derivatives:")
    print(f"\n  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    return psi_bar, lap_psi_bar, rhs, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    verify_solovev_normalized()
    verify_solovev_analytic_derivatives()