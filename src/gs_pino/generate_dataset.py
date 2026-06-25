"""Dataset generation CLI for fixed-boundary GS surrogate experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import trange

from .geometry import PARAM_NAMES, geometry_channels
from .solvers import _HAS_GSPACK, AnalyticFixedBoundarySolver, GSSolverAdapter


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    """Sample one equilibrium parameter vector from gspack2_TRAE reference ranges."""
    # Match parameter bounds from gspack2_TRAE/examples/04_generate_dataset.py
    ranges = {
        "R0": (0.8, 1.5),       # major radius [m]
        "a": (0.3, 0.7),        # minor radius [m]
        "kappa": (1.0, 2.0),    # elongation
        "delta": (0.0, 0.5),    # triangularity (positive only)
        "Ip": (1e5, 5e5),       # plasma current [A]
        "betap": (0.3, 1.5),    # poloidal beta
        "alpha_m": (0.5, 3.0),  # current profile exponent 1
        "alpha_n": (0.5, 3.0),  # current profile exponent 2
    }
    return {name: float(rng.uniform(*ranges[name])) for name in PARAM_NAMES}


def make_case_grid(R0: float, a: float, kappa: float, delta: float, nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a case-specific rectangular grid that encloses the shaped LCFS."""
    r_extent = a * (1.25 + abs(delta))
    z_extent = kappa * a * 1.25

    r = np.linspace(R0 - r_extent, R0 + r_extent, nr, dtype=np.float32)
    z = np.linspace(-z_extent, z_extent, nz, dtype=np.float32)
    return np.meshgrid(r, z, indexing="ij")


def generate(out: str, n_samples: int, nr: int, nz: int, seed: int, rtol: float = 1e-5) -> None:
    """Generate a compressed `.npz` dataset with fields documented in PROJECT.md."""

    # —— 选择求解器 ——
    if _HAS_GSPACK:
        print(f"  Using GSSolverAdapter (gspack2_TRAE real GS solver), rtol={rtol:.0e}.")
        solver = GSSolverAdapter(nr=nr, nz=nz, rtol=rtol)
        use_real_solver = True
    else:
        print("  gspack2_TRAE not found; falling back to AnalyticFixedBoundarySolver.")
        solver = AnalyticFixedBoundarySolver()
        use_real_solver = False

    rng = np.random.default_rng(seed)

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["R", "Z", "params", "psi", "psi_bar", "mask", "sdf", "rho", "theta"]}
    axes: list[list[float]] = []
    profile_params_list: list[np.ndarray] = []

    n_failed = 0
    for _ in trange(n_samples, desc="Generating"):
        params = sample_params(rng)

        if use_real_solver:
            # 真正的 GS 求解器：创建自己的计算网格
            sol = solver.solve(params)
            if sol is None:
                n_failed += 1
                continue

            R = sol["R"]   # (nr, nz)
            Z = sol["Z"]   # (nr, nz)

            # 使用求解器返回的真实 LCFS 掩码
            geom = geometry_channels(R, Z, params, plasma_mask=sol["plasma_mask"])
        else:
            # 备用求解器：我们构造网格
            R, Z = make_case_grid(params["R0"], params["a"], params["kappa"], params["delta"], nr, nz)
            geom = geometry_channels(R, Z, params)
            sol = solver.solve(R, Z, params)

        # 检查 psi_bar 是否有 NaN 或 Inf
        if np.isnan(sol["psi_bar"]).any() or np.isinf(sol["psi_bar"]).any():
            n_failed += 1
            continue

        # 检查 profile_params 是否有 NaN 或 Inf
        if np.isnan(sol.get("profile_params", np.array([0.0, 0.0]))).any() or np.isinf(sol.get("profile_params", np.array([0.0, 0.0]))).any():
            n_failed += 1
            continue

        # 检查 psi_axis 和 psi_lcfs 是否异常大
        if abs(sol["psi_axis"]) > 1e10 or abs(sol["psi_lcfs"]) > 1e10:
            n_failed += 1
            continue

        arrays["R"].append(R)
        arrays["Z"].append(Z)
        arrays["params"].append(np.array([params[name] for name in PARAM_NAMES], dtype=np.float32))

        # 求解器已用 fix_bndry_zero=True，psi 中 LCFS=0
        arrays["psi"].append(sol["psi"])
        arrays["psi_bar"].append(sol["psi_bar"])

        for key in ["mask", "sdf", "rho", "theta"]:
            arrays[key].append(geom[key])

        axes.append([sol["R_axis"], sol["Z_axis"], sol["psi_lcfs"], sol["psi_axis"]])
        profile_params_list.append(sol.get("profile_params", np.array([0.0, 0.0], dtype=np.float32)))

    if n_failed > 0:
        print(f"  Warning: {n_failed} solves failed and were skipped.")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{key: np.stack(value) for key, value in arrays.items()},
        axes=np.array(axes, dtype=np.float32),
        param_names=np.array(PARAM_NAMES),
        profile_params=np.stack(profile_params_list),
    )


def main() -> None:
    """Parse command-line arguments and generate a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gs_fixed_boundary.npz", help="Output .npz dataset path.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of equilibrium cases to generate.")
    parser.add_argument("--nr", type=int, default=64, help="Number of R grid points per case.")
    parser.add_argument("--nz", type=int, default=64, help="Number of Z grid points per case.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="GS solver Picard convergence tolerance.")
    args = parser.parse_args()
    generate(args.out, args.n_samples, args.nr, args.nz, args.seed, args.rtol)


if __name__ == "__main__":
    main()
