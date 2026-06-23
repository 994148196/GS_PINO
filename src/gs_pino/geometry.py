from __future__ import annotations

import numpy as np

PARAM_NAMES = ("R0", "a", "kappa", "delta", "Ip", "betap", "alpha_m", "alpha_n")


def miller_lcfs(theta: np.ndarray, R0: float, a: float, kappa: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Miller-like fixed-boundary LCFS coordinates."""
    return R0 + a * np.cos(theta + delta * np.sin(theta)), kappa * a * np.sin(theta)


def make_grid(R0: float, a: float, kappa: float, margin: float = 0.18, nr: int = 128, nz: int = 128):
    rmin = R0 - a * (1.0 + abs(delta := 0.0) + margin)
    rmax = R0 + a * (1.0 + abs(delta) + margin)
    zmax = kappa * a * (1.0 + margin)
    r = np.linspace(rmin, rmax, nr, dtype=np.float32)
    z = np.linspace(-zmax, zmax, nz, dtype=np.float32)
    return np.meshgrid(r, z, indexing="ij")


def lcfs_level(R: np.ndarray, Z: np.ndarray, R0: float, a: float, kappa: float, delta: float, n_iter: int = 4) -> np.ndarray:
    """Approximate Miller flux-surface radius rho; LCFS is rho=1.

    The triangularity makes the inverse non-linear. A few fixed-point iterations
    are enough for masks/features used by the neural operator.
    """
    theta = np.arctan2(Z / max(kappa * a, 1e-8), (R - R0) / max(a, 1e-8))
    for _ in range(n_iter):
        rb = R0 + a * np.cos(theta + delta * np.sin(theta))
        zb = kappa * a * np.sin(theta)
        theta = np.arctan2(Z / max(kappa * a, 1e-8), (R - R0) / max(a, 1e-8))
        theta = 0.7 * theta + 0.3 * np.arctan2(zb / max(kappa * a, 1e-8), (rb - R0) / max(a, 1e-8))
    boundary_r = np.sqrt((np.cos(theta + delta * np.sin(theta))) ** 2 + (np.sin(theta)) ** 2)
    point_r = np.sqrt(((R - R0) / max(a, 1e-8)) ** 2 + (Z / max(kappa * a, 1e-8)) ** 2)
    return point_r / np.maximum(boundary_r, 1e-6)


def geometry_channels(R: np.ndarray, Z: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray]:
    rho = lcfs_level(R, Z, params["R0"], params["a"], params["kappa"], params["delta"])
    mask = (rho <= 1.0).astype(np.float32)
    sdf = (rho - 1.0).astype(np.float32)
    theta = np.arctan2(Z / (params["kappa"] * params["a"]), (R - params["R0"]) / params["a"]).astype(np.float32)
    return {"rho": rho.astype(np.float32), "theta": theta, "mask": mask, "sdf": sdf}
