from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from tqdm import trange

from .geometry import PARAM_NAMES, geometry_channels
from .solvers import AnalyticFixedBoundarySolver


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    ranges = {
        "R0": (1.5, 1.9), "a": (0.35, 0.55), "kappa": (1.2, 1.9), "delta": (-0.45, 0.45),
        "Ip": (8.0e5, 1.6e6), "betap": (0.2, 1.0), "alpha_m": (1.0, 4.0), "alpha_n": (1.0, 4.0),
    }
    return {k: float(rng.uniform(*ranges[k])) for k in PARAM_NAMES}


def make_case_grid(R0: float, a: float, kappa: float, delta: float, nr: int, nz: int):
    r_extent = a * (1.25 + abs(delta))
    z_extent = kappa * a * 1.25
    r = np.linspace(R0 - r_extent, R0 + r_extent, nr, dtype=np.float32)
    z = np.linspace(-z_extent, z_extent, nz, dtype=np.float32)
    return np.meshgrid(r, z, indexing="ij")


def generate(out: str, n_samples: int, nr: int, nz: int, seed: int):
    rng = np.random.default_rng(seed)
    solver = AnalyticFixedBoundarySolver()
    arrays: dict[str, list] = {k: [] for k in ["R", "Z", "params", "psi", "psi_bar", "mask", "sdf", "rho", "theta", "meta"]}
    axes = []
    for _ in trange(n_samples, desc="Generating"):
        params = sample_params(rng)
        R, Z = make_case_grid(params["R0"], params["a"], params["kappa"], params["delta"], nr, nz)
        geom = geometry_channels(R, Z, params)
        sol = solver.solve(R, Z, params)
        arrays["R"].append(R); arrays["Z"].append(Z)
        arrays["params"].append(np.array([params[k] for k in PARAM_NAMES], dtype=np.float32))
        arrays["psi"].append(sol["psi"]); arrays["psi_bar"].append(sol["psi_bar"])
        for k in ["mask", "sdf", "rho", "theta"]:
            arrays[k].append(geom[k])
        axes.append([sol["R_axis"], sol["Z_axis"], sol["psi_lcfs"], sol["psi_axis"]])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.stack(v) for k, v in arrays.items() if k != "meta"}, axes=np.array(axes, dtype=np.float32), param_names=np.array(PARAM_NAMES))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/gs_fixed_boundary.npz")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--nr", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate(args.out, args.n_samples, args.nr, args.nz, args.seed)

if __name__ == "__main__":
    main()
