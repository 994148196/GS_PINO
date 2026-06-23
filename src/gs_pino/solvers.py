from __future__ import annotations

import numpy as np

from .geometry import geometry_channels


class AnalyticFixedBoundarySolver:
    """Analytic fallback with the same IO contract as a fixed-boundary GS solver.

    It is not a physics replacement for GS_solver; it enables end-to-end data,
    training, evaluation, and plotting before wiring in the external solver.
    """

    def solve(self, R: np.ndarray, Z: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray | float]:
        geom = geometry_channels(R, Z, params)
        rho = geom["rho"]
        theta = geom["theta"]
        m = params["alpha_m"]
        n = params["alpha_n"]
        beta = params["betap"]
        ip_scale = params["Ip"] / 1.0e6
        core = np.clip(1.0 - rho ** 2, 0.0, None)
        shaping = 1.0 + 0.08 * params["delta"] * np.cos(theta) + 0.04 * (params["kappa"] - 1.0) * np.cos(2 * theta)
        psi_bar = (core ** (0.65 + 0.08 * m)) * (1.0 + 0.06 * beta * core ** (0.5 + 0.1 * n)) * shaping
        psi_bar = np.where(geom["mask"] > 0, psi_bar, 0.0).astype(np.float32)
        max_val = float(np.max(psi_bar)) if np.max(psi_bar) > 0 else 1.0
        psi_bar = psi_bar / max_val
        psi_scale = 2.0e-2 * ip_scale * params["a"]
        return {
            "psi_bar": psi_bar.astype(np.float32),
            "psi": (psi_bar * psi_scale).astype(np.float32),
            "psi_lcfs": 0.0,
            "psi_axis": psi_scale,
            "R_axis": float(R[np.unravel_index(np.argmax(psi_bar), psi_bar.shape)]),
            "Z_axis": float(Z[np.unravel_index(np.argmax(psi_bar), psi_bar.shape)]),
        }
