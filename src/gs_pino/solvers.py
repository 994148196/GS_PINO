"""Solver adapters for fixed-boundary Grad-Shafranov data generation."""

from __future__ import annotations

import numpy as np

from .geometry import geometry_channels


class AnalyticFixedBoundarySolver:
    """Analytic fallback with the same I/O contract as a GS solver adapter.

    This class is deliberately simple: it lets the project run end-to-end before
    the external `GS_solver` package is installed.  For production data, replace
    `solve()` with a call to the real fixed-boundary solver and return the same
    dictionary keys.
    """

    def solve(self, R: np.ndarray, Z: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray | float]:
        """Return synthetic `psi` fields and axis metadata for one equilibrium."""
        # Reuse geometry channels so the fallback and dataset masks are consistent.
        geom = geometry_channels(R, Z, params)
        rho = geom["rho"]
        theta = geom["theta"]

        # Profile parameters shape the synthetic core profile in a smooth way.
        m = params["alpha_m"]
        n = params["alpha_n"]
        beta = params["betap"]
        ip_scale = params["Ip"] / 1.0e6

        # Core goes to zero at LCFS and is clipped outside LCFS.
        core = np.clip(1.0 - rho**2, 0.0, None)

        # Add mild shaping so elongation/triangularity visibly affect contours.
        shaping = 1.0 + 0.08 * params["delta"] * np.cos(theta) + 0.04 * (params["kappa"] - 1.0) * np.cos(2 * theta)

        # Smooth normalized flux-like field, zeroed outside LCFS.
        psi_bar = (core ** (0.65 + 0.08 * m)) * (1.0 + 0.06 * beta * core ** (0.5 + 0.1 * n)) * shaping
        psi_bar = np.where(geom["mask"] > 0, psi_bar, 0.0).astype(np.float32)

        # Normalize axis value to one for stable training targets.
        max_val = float(np.max(psi_bar)) if np.max(psi_bar) > 0 else 1.0
        psi_bar = psi_bar / max_val

        # Give the physical psi a simple current/minor-radius scale for plotting.
        psi_scale = 2.0e-2 * ip_scale * params["a"]
        axis_index = np.unravel_index(np.argmax(psi_bar), psi_bar.shape)
        return {
            "psi_bar": psi_bar.astype(np.float32),
            "psi": (psi_bar * psi_scale).astype(np.float32),
            "psi_lcfs": 0.0,
            "psi_axis": psi_scale,
            "R_axis": float(R[axis_index]),
            "Z_axis": float(Z[axis_index]),
        }
