"""Analytical verification of Solov'ev equilibrium."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def verify_solovev_analytic():
    """Verify Solov'ev satisfies GS equation analytically."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    nr = 500
    rho = np.linspace(1e-6, 0.999, nr)
    
    psi_bar = (1.0 - rho**2) * (1.0 + Beta0 * (1.0 - rho**2)**(alpha_n - 1)) / (1.0 + Beta0)
    psiN = 1.0 - psi_bar
    
    psi_axis = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    psi = psi_bar * psi_axis
    
    x = 1.0 - rho**2
    
    dpsi_bar_dx = (1.0 + Beta0 * alpha_n * x**(alpha_n - 1)) / (1.0 + Beta0)
    d2psi_bar_dx2 = -(Beta0 * alpha_n * (alpha_n - 1) * x**(alpha_n - 2)) / (1.0 + Beta0)
    
    dpsi_bar_drho = dpsi_bar_dx * (-2.0 * rho)
    d2psi_bar_drho2 = d2psi_bar_dx2 * (4.0 * rho**2) + dpsi_bar_dx * (-2.0)
    
    lap_psi_bar_rho = (1.0 / rho) * dpsi_bar_drho + d2psi_bar_drho2
    
    lap_psi_rho = lap_psi_bar_rho * psi_axis
    
    lap_psi_r0 = lap_psi_rho / a**2
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    Jtor = L * (Beta0 * rho**2 + (1.0 - Beta0)) / (R0 * (1.0 - rho**2)) * jtorshape
    
    rhs = mu0 * R0 * Jtor
    
    residual = lap_psi_r0 + rhs
    
    print(f"Analytical Solov'ev Verification (circular, axisymmetric):")
    print(f"=" * 60)
    print(f"  Parameters:")
    print(f"    R0 = {R0} m")
    print(f"    a = {a} m")
    print(f"    Ip = {Ip} A")
    print(f"    Beta0 = {Beta0}")
    print(f"    alpha_m = {alpha_m}")
    print(f"    alpha_n = {alpha_n}")
    print(f"    L = {L:.6e}")
    print(f"    psi_axis = {psi_axis:.6e} Wb/rad")
    print(f"=" * 60)
    print(f"  GS Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    print(f"=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(rho, psiN, 'b-')
    axes[0, 0].set_title('psiN')
    axes[0, 0].set_xlabel('rho')
    
    axes[0, 1].plot(rho, lap_psi_r0, 'r-')
    axes[0, 1].set_title('Delta*psi (R=R0)')
    axes[0, 1].set_xlabel('rho')
    
    axes[1, 0].plot(rho, rhs, 'g-')
    axes[1, 0].set_title('-mu0*R*Jtor')
    axes[1, 0].set_xlabel('rho')
    
    axes[1, 1].plot(rho, residual, 'k-')
    axes[1, 1].set_title('GS Residual')
    axes[1, 1].set_xlabel('rho')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_analytic_verification.png', dpi=150)
    plt.close()
    
    return psiN, lap_psi_r0, rhs, residual


def verify_simple_case():
    """Test simple case: Beta0=0, alpha_m=1, alpha_n=1."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.0
    alpha_m = 1.0
    alpha_n = 1.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    nr = 500
    rho = np.linspace(1e-6, 0.999, nr)
    
    psi_bar = 1.0 - rho**2
    psiN = rho**2
    
    psi_axis = mu0 * L * R0**2 / 2
    psi = psi_bar * psi_axis
    
    dpsi_bar_drho = -2.0 * rho
    d2psi_bar_drho2 = -2.0
    
    lap_psi_bar_rho = (1.0 / rho) * dpsi_bar_drho + d2psi_bar_drho2
    lap_psi_r0 = lap_psi_bar_rho * psi_axis / a**2
    
    jtorshape = (1.0 - psiN)**1
    
    Jtor = L * (1.0 - Beta0) / (R0 * (1.0 - rho**2)) * jtorshape
    
    rhs = mu0 * R0 * Jtor
    
    residual = lap_psi_r0 + rhs
    
    print(f"\nSimple Case (Beta0=0, n=1):")
    print(f"  psi_bar = {psi_bar[0]:.6f} at axis, {psi_bar[-1]:.6f} at boundary")
    print(f"  psiN = {psiN[0]:.6f} at axis, {psiN[-1]:.6f} at boundary")
    print(f"  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    return psiN, lap_psi_r0, rhs, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    verify_solovev_analytic()
    verify_simple_case()