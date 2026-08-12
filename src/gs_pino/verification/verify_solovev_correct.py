"""Correct verification of Solov'ev equilibrium matching gspack's conventions."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def solovev_psiN(rho, alpha_m=1.0, alpha_n=2.0, Beta0=0.5):
    """Compute Solov'ev psiN matching gspack's convention.
    
    gspack definition: psiN = (psi - psi_axis) / (psi_bndry - psi_axis)
    psiN = 0 at magnetic axis, psiN = 1 at LCFS boundary.
    
    For Solov'ev equilibrium:
        psi_bar = (1 - rho^2) * (1 + Beta0 * (1 - rho^2)^(alpha_n-1)) / (1 + Beta0)
        psi_bar = 1 at axis, 0 at boundary
        
        psiN = 1 - psi_bar
        psiN = 0 at axis, 1 at boundary
    """
    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar = core * (1.0 + Beta0 * core**(alpha_n - 1)) / (1.0 + Beta0)
    psiN = 1.0 - psi_bar
    return psiN


def verify_solovev_correct():
    """Verify Solov'ev satisfies GS equation using gspack's exact conventions."""
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
    drho = rho[1] - rho[0]
    
    psiN = solovev_psiN(rho, alpha_m, alpha_n, Beta0)
    
    dpsiN_drho = np.gradient(psiN, drho)
    d2psiN_drho2 = np.gradient(dpsiN_drho, drho)
    
    lap_psiN_rho = (1.0 / rho) * dpsiN_drho + d2psiN_drho2
    
    psi_axis = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    
    dpsi = psi_axis
    
    lap_psi = lap_psiN_rho * dpsi / a**2
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R0 * pprime_val + ffprime_val / (mu0 * R0)
    
    rhs = mu0 * Jtor
    
    residual = -lap_psi - rhs
    
    print(f"Correct Solov'ev Verification (gspack convention):")
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
    axes[0, 0].set_title('psiN (gspack convention)')
    axes[0, 0].set_xlabel('rho')
    
    axes[0, 1].plot(rho, lap_psi, 'r-')
    axes[0, 1].set_title('Laplacian of psi')
    axes[0, 1].set_xlabel('rho')
    
    axes[1, 0].plot(rho, rhs, 'g-')
    axes[1, 0].set_title('RHS (mu0 * Jtor)')
    axes[1, 0].set_xlabel('rho')
    
    axes[1, 1].plot(rho, residual, 'k-')
    axes[1, 1].set_title('GS Residual')
    axes[1, 1].set_xlabel('rho')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'solovev_correct_verification.png', dpi=150)
    plt.close()
    
    return psiN, lap_psi, rhs, residual


def verify_with_analytic_derivatives():
    """Verify using analytical derivatives."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.5
    alpha_m = 1.0
    alpha_n = 2.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    psi_axis = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    
    nr = 500
    rho = np.linspace(1e-6, 0.999, nr)
    
    psiN = solovev_psiN(rho, alpha_m, alpha_n, Beta0)
    
    x = np.clip(1.0 - rho**2, 0.0, None)
    
    dpsi_bar_dx = (1.0 + Beta0 * alpha_n * x**(alpha_n - 1)) / (1.0 + Beta0)
    psi_bar = x * (1.0 + Beta0 * x**(alpha_n - 1)) / (1.0 + Beta0)
    
    dpsiN_dx = -dpsi_bar_dx
    d2psiN_dx2 = -(Beta0 * alpha_n * (alpha_n - 1) * x**(alpha_n - 2)) / (1.0 + Beta0)
    
    dpsiN_drho = dpsiN_dx * (-2.0 * rho)
    d2psiN_drho2 = d2psiN_dx2 * (4.0 * rho**2) + dpsiN_dx * (-2.0)
    
    lap_psiN_rho = (1.0 / rho) * dpsiN_drho + d2psiN_drho2
    
    lap_psi = lap_psiN_rho * psi_axis / a**2
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R0 * pprime_val + ffprime_val / (mu0 * R0)
    rhs = mu0 * Jtor
    
    residual = -lap_psi - rhs
    
    print(f"\nAnalytical Derivatives Verification:")
    print(f"  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    return psiN, lap_psi, rhs, residual


def test_simple_case():
    """Test simple case: Beta0=0."""
    R0 = 1.0
    a = 0.5
    Ip = 2e5
    Beta0 = 0.0
    alpha_m = 1.0
    alpha_n = 1.0
    
    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)
    
    psi_axis = mu0 * L * R0**2 / 2 * (1.0 + Beta0)
    
    nr = 500
    rho = np.linspace(1e-6, 0.999, nr)
    drho = rho[1] - rho[0]
    
    psiN = solovev_psiN(rho, alpha_m, alpha_n, Beta0)
    
    dpsiN_drho = np.gradient(psiN, drho)
    d2psiN_drho2 = np.gradient(dpsiN_drho, drho)
    
    lap_psiN_rho = (1.0 / rho) * dpsiN_drho + d2psiN_drho2
    
    lap_psi = lap_psiN_rho * psi_axis / a**2
    
    jtorshape = (1.0 - psiN**alpha_m)**alpha_n
    
    pprime_val = L * Beta0 / R0 * jtorshape
    ffprime_val = mu0 * L * (1.0 - Beta0) * R0 * jtorshape
    
    Jtor = R0 * pprime_val + ffprime_val / (mu0 * R0)
    rhs = mu0 * Jtor
    
    residual = -lap_psi - rhs
    
    print(f"\nSimple Case (Beta0=0, n=1):")
    print(f"  Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")
    
    return psiN, lap_psi, rhs, residual


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    verify_solovev_correct()
    verify_with_analytic_derivatives()
    test_simple_case()