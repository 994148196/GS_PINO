"""LCFS geometry utilities for shaped fixed-boundary equilibria."""

from __future__ import annotations

import numpy as np

# Keep one canonical parameter order everywhere: dataset, plots, and model input.
PARAM_NAMES = ("R0", "a", "kappa", "delta", "Ip", "betap", "alpha_m", "alpha_n")


def miller_lcfs(theta: np.ndarray, R0: float, a: float, kappa: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Miller-like LCFS coordinates for a poloidal angle array."""
    # Triangularity shifts the cosine angle; elongation scales vertical height.
    return R0 + a * np.cos(theta + delta * np.sin(theta)), kappa * a * np.sin(theta)


def lcfs_level(R: np.ndarray, Z: np.ndarray, R0: float, a: float, kappa: float, delta: float) -> np.ndarray:
    """Approximate normalized flux-surface radius `rho` for a Miller boundary.

    `rho <= 1` is treated as inside LCFS.  This is a geometry feature and mask
    helper, not a replacement for solving the Grad-Shafranov equation.
    """
    # A robust angle estimate in Miller-normalized coordinates.
    theta = np.arctan2(Z / max(kappa * a, 1e-8), (R - R0) / max(a, 1e-8))

    # Boundary radius in normalized `(R-R0)/a, Z/(kappa*a)` coordinates.
    boundary_r = np.sqrt(np.cos(theta + delta * np.sin(theta)) ** 2 + np.sin(theta) ** 2)

    # Point radius in the same normalized coordinate system.
    point_r = np.sqrt(((R - R0) / max(a, 1e-8)) ** 2 + (Z / max(kappa * a, 1e-8)) ** 2)

    # Ratio gives an inexpensive LCFS level-set proxy: LCFS is approximately 1.
    return point_r / np.maximum(boundary_r, 1e-6)


def geometry_channels(R: np.ndarray, Z: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray]:
    """Compute mask/SDF/rho/theta channels used by data generation and training."""
    # `rho` is the primary non-rectangular-domain descriptor.
    rho = lcfs_level(R, Z, params["R0"], params["a"], params["kappa"], params["delta"])

    # Binary mask isolates physically meaningful in-LCFS points.
    mask = (rho <= 1.0).astype(np.float32)

    # Signed-distance proxy is negative inside and positive outside.
    sdf = (rho - 1.0).astype(np.float32)

    # Angle is encoded later as sin/cos to avoid discontinuity at +/- pi.
    theta = np.arctan2(Z / (params["kappa"] * params["a"]), (R - params["R0"]) / params["a"]).astype(np.float32)
    return {"rho": rho.astype(np.float32), "theta": theta, "mask": mask, "sdf": sdf}
