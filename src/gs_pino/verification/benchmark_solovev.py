"""Benchmark for fixed-boundary Grad-Shafranov using generalized Solov'ev equilibrium.

This script compares:
1. Analytic generalized Solov'ev equilibrium solution
2. Numerical solution from gspack FixedBoundaryEquilibrium (if available)
3. Numerical solution from freegs with fixed boundary (if available)

The generalized Solov'ev equilibrium is given by:
    psi(rho, theta) = psi_axis * (1 - rho^2)^(alpha_m) * [1 + beta_p * (1 - rho^2)^(alpha_n)]
    
where rho is normalized poloidal flux coordinate and theta is poloidal angle.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "solovev_benchmark"


def generalized_solovev_psi(
    R: np.ndarray,
    Z: np.ndarray,
    params: dict[str, float],
    use_gspack_params: bool = False,
) -> dict[str, np.ndarray | float]:
    """Compute generalized Solov'ev equilibrium analytically.

    The generalized Solov'ev solution satisfies the Grad-Shafranov equation:
       -∇*ψ = μ₀/R * dP/dψ * ψ + μ₀/R * FF'(ψ)

    where:
       ψ(rho) = ψ_axis * (1 - rho^2)^m * [1 + beta_p * (1 - rho^2)^n]
       rho = r/a is normalized radial coordinate
       p(ψN) = p0 * (1 - ψN^m)^n where ψN = ψ/ψ_axis
       FF'(ψN) = μ₀ * L * (1 - beta_p) * R0 * (1 - ψN^m)^n

    Parameters
    ----------
    R, Z : 2D arrays of shape (nr, nz)
        Computational grid coordinates.
    params : dict
        Equilibrium parameters: R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n.
    use_gspack_params : bool
        If True, compute L and Beta0 using gspack's ConstrainBetapIp formula.

    Returns
    -------
    dict with psi_bar, psi, psi_axis, psi_lcfs, R_axis, Z_axis, plasma_mask.
    """
    R0 = params["R0"]
    a = params["a"]
    kappa = params["kappa"]
    delta = params["delta"]
    Ip = params["Ip"]
    betap = params["betap"]
    alpha_m = params["alpha_m"]
    alpha_n = params["alpha_n"]

    r = np.sqrt((R - R0) ** 2 + Z**2)
    rho = r / a

    mu0 = 4 * np.pi * 1e-7
    L = Ip / (2 * np.pi * R0)

    psi_bar = np.zeros_like(rho)
    valid = rho <= 1.0
    core = 1.0 - rho[valid] ** 2
    psi_bar[valid] = (core**alpha_m) * (1.0 + betap * core**alpha_n)
    psi_bar = np.clip(psi_bar, 0.0, None)

    psi_bar_max = float(np.max(psi_bar)) if np.max(psi_bar) > 0 else 1.0
    psi_bar = psi_bar / psi_bar_max

    psi_axis_val = (mu0 * L * R0**2 / 2) * (1.0 + betap)
    psi = psi_bar * psi_axis_val

    axis_index = np.unravel_index(np.argmax(psi_bar), psi_bar.shape)
    R_axis = float(R[axis_index])
    Z_axis = float(Z[axis_index])

    plasma_mask = np.where(rho <= 1.0, 1.0, 0.0).astype(np.float32)

    print(f"  Solov'ev L: {L:.6e}, Beta0: {betap:.6e}, psi_axis: {psi_axis_val:.6e}")

    return {
        "R": R.astype(np.float32),
        "Z": Z.astype(np.float32),
        "psi_bar": psi_bar.astype(np.float32),
        "psi": psi.astype(np.float32),
        "psi_axis": psi_axis_val,
        "psi_lcfs": 0.0,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "plasma_mask": plasma_mask,
        "profile_params": np.array([float(L), float(betap)], dtype=np.float32),
    }


def compute_grad_shafranov_residual(
    psi_bar: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    psi_axis: float,
    L: float,
    betap: float,
    alpha_m: float,
    alpha_n: float,
) -> np.ndarray:
    """Compute Grad-Shafranov equation residual using direct verification method.

    For Solov'ev equilibrium, we verify that:
       -∇²ψ = μ₀/R * (dP/dψ * ψ + FF'(ψ))

    where in flux coordinates:
       ψ = psi_axis * psi_bar
       psi_bar = (1 - ρ²)^m * (1 + β_p * (1 - ρ²)^n) / (1 + β_p)
       ρ = r/a = sqrt((R-R0)^2 + Z^2) / a

    The Laplacian in cylindrical coordinates for axisymmetric ψ:
       ∇²ψ = (1/R) ∂/∂R (R ∂ψ/∂R) + ∂²ψ/∂Z²
    """
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]

    psi = psi_bar * psi_axis

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
    R0 = float(R[R.shape[0] // 2, R.shape[1] // 2])

    pb = psi_bar[1:-1, 1:-1]
    pb = np.clip(pb, 1e-30, 1.0)

    core = 1.0 - pb**alpha_m
    core = np.clip(core, 0.0, None)

    dP_dpsi = (L * betap / R0) * core**alpha_n / psi_axis

    FF_prime = mu0 * L * (1.0 - betap) * R0 * core**alpha_n

    psi_phys = pb * psi_axis

    rhs = (mu0 / R_c) * (dP_dpsi * psi_phys + FF_prime)
    residual = -lap_psi - rhs

    return residual


def benchmark_with_gspack(params: dict[str, float], nr: int = 65, nz: int = 65) -> dict | None:
    """Run benchmark using gspack FixedBoundaryEquilibrium."""
    try:
        import sys
        import warnings
        import io
        import contextlib

        gspack_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "gspack2_TRAE"
        if gspack_dir.is_dir() and str(gspack_dir) not in sys.path:
            sys.path.insert(0, str(gspack_dir))

        from gspack.equilibrium import FixedBoundaryEquilibrium
        from gspack.profiles import ConstrainBetapIp
        from gspack import picard

        R0 = float(params["R0"])
        a = float(params["a"])
        kappa = float(params["kappa"])
        delta = float(params["delta"])
        Ip = float(params["Ip"])
        betap = float(params["betap"])
        alpha_m = float(params["alpha_m"])
        alpha_n = float(params["alpha_n"])

        margin = max(0.15, 0.3 * a)
        Rmin = R0 - a - margin
        Rmax = R0 + a + margin
        Zmax_val = kappa * a + margin
        Zmin = -Zmax_val

        eq = FixedBoundaryEquilibrium(
            R0=R0, a=a, kappa=kappa, delta=delta,
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax_val,
            nx=nr, ny=nz, order=2, method="lu",
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

        psi_native = np.asarray(eq.psi(), dtype=np.float32)
        psi_bndry = float(eq.psi_bndry)
        psi_axis_val = float(eq.psi_axis)
        dpsi = psi_axis_val - psi_bndry

        if abs(dpsi) < 1e-30:
            return None

        psi_bar = (psi_native - psi_bndry) / dpsi
        psi_bar = np.clip(psi_bar, 0.0, None).astype(np.float32)

        R_axis, Z_axis, _ = eq.magneticAxis()
        plasma_mask = eq.plasma_mask.astype(np.float32)
        profile_params = np.array([float(pro.L), float(pro.Beta0)], dtype=np.float32)

        print(f"  gspack L: {pro.L:.6e}, Beta0: {pro.Beta0:.6e}")

        return {
            "R": np.asarray(eq.R, dtype=np.float32),
            "Z": np.asarray(eq.Z, dtype=np.float32),
            "psi_bar": psi_bar,
            "psi": psi_native,
            "psi_axis": psi_axis_val,
            "psi_lcfs": psi_bndry,
            "R_axis": float(R_axis),
            "Z_axis": float(Z_axis),
            "plasma_mask": plasma_mask,
            "profile_params": profile_params,
        }

    except ImportError:
        return None
    except Exception as e:
        print(f"gspack benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_with_freegs(params: dict[str, float], nx: int = 65, ny: int = 65) -> dict | None:
    """Run benchmark using freegs with fixed boundary."""
    try:
        import freegs
        from freegs import jtor, control, boundary

        tokamak = freegs.machine.TestTokamak()

        R0 = params["R0"]
        a = params["a"]
        kappa = params["kappa"]
        delta = params["delta"]

        margin = max(0.15, 0.3 * a)
        Rmin = R0 - a - margin
        Rmax = R0 + a + margin
        Zmax_val = kappa * a + margin
        Zmin = -Zmax_val

        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=Rmin,
            Rmax=Rmax,
            Zmin=Zmin,
            Zmax=Zmax_val,
            nx=nx,
            ny=ny,
            boundary=boundary.fixedBoundary,
        )

        profiles = jtor.ConstrainPaxisIp(
            eq,
            paxis=params.get("paxis", 1000.0),
            Ip=params["Ip"],
            fvac=params.get("fvac", 1.0),
            alpha_m=params["alpha_m"],
            alpha_n=params["alpha_n"],
            Raxis=R0,
        )

        constrain = control.constrain(xpoints=[])

        freegs.solve(eq, profiles, constrain, show=False)

        if eq.psi_axis is None or eq.psi_bndry is None:
            return None

        psi_total = eq.psi()
        psi_plasma = eq.plasma_psi

        rtol = 1e-3
        psi_coils = eq.tokamak.psi(eq.R, eq.Z)
        if not np.allclose(psi_total, psi_plasma + psi_coils, rtol=rtol):
            return None

        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)

        opt, xpt = freegs.critical.find_critical(eq.R, eq.Z, psi_total)
        R_axis = opt[0][0] if opt else R0
        Z_axis = opt[0][1] if opt else 0.0

        mask = freegs.critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt) if xpt else np.ones_like(psi_total)

        if np.isnan(psi_plasma_norm).any() or np.isinf(psi_plasma_norm).any():
            return None

        return {
            "R": eq.R.astype(np.float32),
            "Z": eq.Z.astype(np.float32),
            "psi_bar": psi_plasma_norm.astype(np.float32),
            "psi": psi_plasma.astype(np.float32),
            "psi_axis": float(eq.psi_axis),
            "psi_lcfs": float(eq.psi_bndry),
            "R_axis": float(R_axis),
            "Z_axis": float(Z_axis),
            "plasma_mask": mask.astype(np.float32),
            "profile_params": np.array([getattr(profiles, "L", 0.0), getattr(profiles, "Beta0", 0.0)], dtype=np.float32),
        }

    except ImportError:
        return None
    except Exception as e:
        print(f"freegs benchmark failed: {e}")
        return None


def compute_metrics(solovev: dict, numerical: dict, label: str) -> dict:
    """Compute comparison metrics between Solov'ev and numerical solution."""
    mask = solovev["plasma_mask"] > 0.5

    psi_bar_sol = solovev["psi_bar"][mask]
    psi_bar_num = numerical["psi_bar"][mask]

    if psi_bar_sol.size == 0 or psi_bar_num.size == 0:
        return {"rmse": np.nan, "max_error": np.nan, "rel_l2": np.nan}

    rmse = np.sqrt(np.mean((psi_bar_sol - psi_bar_num) ** 2))
    max_error = np.max(np.abs(psi_bar_sol - psi_bar_num))
    rel_l2 = np.sqrt(np.sum((psi_bar_sol - psi_bar_num) ** 2)) / np.sqrt(np.sum(psi_bar_sol**2))

    R_axis_diff = abs(solovev["R_axis"] - numerical["R_axis"])
    Z_axis_diff = abs(solovev["Z_axis"] - numerical["Z_axis"])
    psi_axis_ratio = numerical["psi_axis"] / solovev["psi_axis"]

    print(f"\n{label} Comparison Metrics:")
    print(f"  RMSE (psi_bar): {rmse:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    print(f"  Relative L2: {rel_l2 * 100:.2f}%")
    print(f"  R_axis diff: {R_axis_diff:.4f} m")
    print(f"  Z_axis diff: {Z_axis_diff:.4f} m")
    print(f"  psi_axis ratio: {psi_axis_ratio:.4f}")

    return {
        "rmse": rmse,
        "max_error": max_error,
        "rel_l2": rel_l2,
        "R_axis_diff": R_axis_diff,
        "Z_axis_diff": Z_axis_diff,
        "psi_axis_ratio": psi_axis_ratio,
    }


def solovev_from_gspack(gspack_result: dict, params: dict[str, float]) -> dict[str, np.ndarray | float]:
    """Compute Solov'ev solution using gspack's actual grid, L, and Beta0.

    This ensures a fair comparison by using gspack's coordinate system
    and profile parameters.
    """
    R = gspack_result["R"]
    Z = gspack_result["Z"]
    L = float(gspack_result["profile_params"][0])
    Beta0 = float(gspack_result["profile_params"][1])
    R_axis = gspack_result["R_axis"]
    Z_axis = gspack_result["Z_axis"]

    alpha_m = params["alpha_m"]
    alpha_n = params["alpha_n"]

    rho = np.sqrt((R - R_axis) ** 2 + (Z - Z_axis) ** 2)
    rho = rho / params["a"]

    core = np.clip(1.0 - rho**2, 0.0, None)
    psi_bar_raw = (core**alpha_m) * (1.0 + Beta0 * core**alpha_n)
    psi_bar_raw = np.where(rho <= 1.0, psi_bar_raw, 0.0)
    psi_bar_raw = np.clip(psi_bar_raw, 0.0, None)

    psi_bar = psi_bar_raw / (1.0 + Beta0)

    mu0 = 4 * np.pi * 1e-7
    psi_axis_val = (mu0 * L * R_axis**2 / 2) * (1.0 + Beta0)
    psi = psi_bar * psi_axis_val

    plasma_mask = np.where(rho <= 1.0, 1.0, 0.0).astype(np.float32)

    print(f"  Solov'ev (gspack-compat): L={L:.6e}, Beta0={Beta0:.6e}, psi_axis={psi_axis_val:.6e}")

    return {
        "R": R,
        "Z": Z,
        "psi_bar": psi_bar.astype(np.float32),
        "psi": psi.astype(np.float32),
        "psi_axis": psi_axis_val,
        "psi_lcfs": 0.0,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "plasma_mask": plasma_mask,
        "profile_params": gspack_result["profile_params"],
    }


def plot_comparison(
    solovev: dict,
    gspack_result: dict | None,
    freegs_result: dict | None,
    params: dict[str, float],
    out_dir: str,
) -> None:
    """Generate comparison plots."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    R, Z = solovev["R"], solovev["Z"]

    im0 = axes[0, 0].contourf(R, Z, solovev["psi_bar"], levels=50, cmap="viridis")
    axes[0, 0].contour(R, Z, solovev["psi_bar"], levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], colors="white", linewidths=0.5)
    axes[0, 0].set_title(f"Solov'ev Analytic\n(R0={params['R0']}, a={params['a']}, βp={params['betap']})")
    axes[0, 0].set_xlabel("R (m)")
    axes[0, 0].set_ylabel("Z (m)")
    axes[0, 0].scatter(solovev["R_axis"], solovev["Z_axis"], c="red", s=50, marker="x", label="Magnetic Axis")
    axes[0, 0].legend()
    plt.colorbar(im0, ax=axes[0, 0])

    if gspack_result is not None:
        im1 = axes[0, 1].contourf(gspack_result["R"], gspack_result["Z"], gspack_result["psi_bar"], levels=50, cmap="viridis")
        axes[0, 1].contour(gspack_result["R"], gspack_result["Z"], gspack_result["psi_bar"], levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], colors="white", linewidths=0.5)
        axes[0, 1].set_title("gspack FixedBoundaryEquilibrium")
        axes[0, 1].set_xlabel("R (m)")
        axes[0, 1].set_ylabel("Z (m)")
        axes[0, 1].scatter(gspack_result["R_axis"], gspack_result["Z_axis"], c="red", s=50, marker="x")
        plt.colorbar(im1, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, "gspack not available", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].axis("off")

    if freegs_result is not None:
        im2 = axes[1, 0].contourf(freegs_result["R"], freegs_result["Z"], freegs_result["psi_bar"], levels=50, cmap="viridis")
        axes[1, 0].contour(freegs_result["R"], freegs_result["Z"], freegs_result["psi_bar"], levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], colors="white", linewidths=0.5)
        axes[1, 0].set_title("freegs Fixed Boundary")
        axes[1, 0].set_xlabel("R (m)")
        axes[1, 0].set_ylabel("Z (m)")
        axes[1, 0].scatter(freegs_result["R_axis"], freegs_result["Z_axis"], c="red", s=50, marker="x")
        plt.colorbar(im2, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "freegs not available", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].axis("off")

    if gspack_result is not None:
        diff = solovev["psi_bar"] - gspack_result["psi_bar"]
        im3 = axes[1, 1].contourf(R, Z, diff, levels=50, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        axes[1, 1].set_title("Solov'ev - gspack Difference")
        axes[1, 1].set_xlabel("R (m)")
        axes[1, 1].set_ylabel("Z (m)")
        plt.colorbar(im3, ax=axes[1, 1])
    elif freegs_result is not None:
        diff = solovev["psi_bar"] - freegs_result["psi_bar"]
        im3 = axes[1, 1].contourf(R, Z, diff, levels=50, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        axes[1, 1].set_title("Solov'ev - freegs Difference")
        axes[1, 1].set_xlabel("R (m)")
        axes[1, 1].set_ylabel("Z (m)")
        plt.colorbar(im3, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "No numerical results", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/solovev_benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    L = float(solovev["profile_params"][0])
    betap = float(solovev["profile_params"][1])
    residual = compute_grad_shafranov_residual(
        solovev["psi_bar"], R, Z, solovev["psi_axis"], L, betap, params["alpha_m"], params["alpha_n"]
    )
    im = ax.contourf(R[1:-1, 1:-1], Z[1:-1, 1:-1], residual, levels=50, cmap="RdBu_r")
    ax.set_title("GS Residual (Solov'ev)")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    plt.colorbar(im, ax=ax)
    plt.savefig(f"{out_dir}/solovev_gs_residual.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Solov'ev GS Residual Statistics:")
    print(f"    Mean: {np.mean(residual):.6e}")
    print(f"    Max: {np.max(residual):.6e}")
    print(f"    Min: {np.min(residual):.6e}")
    print(f"    RMSE: {np.sqrt(np.mean(residual**2)):.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generalized Solov'ev equilibrium against numerical solvers")
    parser.add_argument("--R0", type=float, default=1.0, help="Magnetic axis major radius (m)")
    parser.add_argument("--a", type=float, default=0.5, help="Plasma minor radius (m)")
    parser.add_argument("--kappa", type=float, default=1.0, help="Elongation")
    parser.add_argument("--delta", type=float, default=0.0, help="Triangularity")
    parser.add_argument("--Ip", type=float, default=2e5, help="Plasma current (A)")
    parser.add_argument("--betap", type=float, default=0.5, help="Poloidal beta")
    parser.add_argument("--alpha-m", type=float, default=1.0, help="Profile exponent m")
    parser.add_argument("--alpha-n", type=float, default=2.0, help="Profile exponent n")
    parser.add_argument("--paxis", type=float, default=1000.0, help="On-axis pressure (Pa)")
    parser.add_argument("--nx", type=int, default=65, help="Number of R grid points")
    parser.add_argument("--ny", type=int, default=65, help="Number of Z grid points")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--gspack-compat", action="store_true", help="Use gspack-compatible parameter calculation")
    args = parser.parse_args()

    params = {
        "R0": args.R0,
        "a": args.a,
        "kappa": args.kappa,
        "delta": args.delta,
        "Ip": args.Ip,
        "betap": args.betap,
        "alpha_m": args.alpha_m,
        "alpha_n": args.alpha_n,
        "paxis": args.paxis,
    }

    print(f"\n{'='*60}")
    print(f"  Generalized Solov'ev Equilibrium Benchmark")
    print(f"{'='*60}")
    print(f"  Parameters:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(f"  Grid: {args.nx}x{args.ny}")
    print(f"  gspack-compat: {args.gspack_compat}")
    print(f"{'='*60}")

    Rmin = params["R0"] - params["a"] - 0.15
    Rmax = params["R0"] + params["a"] + 0.15
    Zmax_val = params["kappa"] * params["a"] + 0.15
    Zmin = -Zmax_val

    rr = np.linspace(Rmin, Rmax, args.nx)
    zz = np.linspace(Zmin, Zmax_val, args.ny)
    R, Z = np.meshgrid(rr, zz, indexing="ij")

    print("\n[1/3] Computing Solov'ev analytic solution...")
    solovev = generalized_solovev_psi(R, Z, params, use_gspack_params=args.gspack_compat)
    print(f"  Solov'ev psi_axis: {solovev['psi_axis']:.6e}")
    print(f"  Solov'ev R_axis: {solovev['R_axis']:.4f} m")
    print(f"  Solov'ev Z_axis: {solovev['Z_axis']:.4f} m")

    print("\n[2/3] Running gspack benchmark...")
    gspack_result = benchmark_with_gspack(params, args.nx, args.ny)
    if gspack_result is not None:
        print(f"  gspack psi_axis: {gspack_result['psi_axis']:.6e}")
        print(f"  gspack R_axis: {gspack_result['R_axis']:.4f} m")
        print(f"  gspack Z_axis: {gspack_result['Z_axis']:.4f} m")

        print("\n  Computing Solov'ev using gspack's actual parameters...")
        solovev_gspack = solovev_from_gspack(gspack_result, params)

        print("\n  === Direct comparison (gspack grid + parameters) ===")
        compute_metrics(solovev_gspack, gspack_result, "gspack")
    else:
        print("  gspack not available, skipping...")

    print("\n[3/3] Running freegs benchmark...")
    freegs_result = benchmark_with_freegs(params, args.nx, args.ny)
    if freegs_result is not None:
        print(f"  freegs psi_axis: {freegs_result['psi_axis']:.6e}")
        print(f"  freegs R_axis: {freegs_result['R_axis']:.4f} m")
        print(f"  freegs Z_axis: {freegs_result['Z_axis']:.4f} m")
        compute_metrics(solovev, freegs_result, "freegs")
    else:
        print("  freegs not available or failed, skipping...")

    print(f"\n{'='*60}")
    print(f"  Generating visualization...")
    plot_comparison(solovev, gspack_result, freegs_result, params, args.out)
    print(f"  Output saved to {args.out}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()