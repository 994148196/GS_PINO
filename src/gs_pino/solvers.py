"""Solver adapters for fixed-boundary Grad-Shafranov data generation."""

from __future__ import annotations

import io
import os
import sys
import warnings
import contextlib

import numpy as np

from .geometry import geometry_channels

# 确保 gspack2_TRAE 在 Python 路径中
_GSPACK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gspack2_TRAE")
)
if os.path.isdir(_GSPACK_DIR) and _GSPACK_DIR not in sys.path:
    sys.path.insert(0, _GSPACK_DIR)

_HAS_GSPACK = False
try:
    from gspack.equilibrium import FixedBoundaryEquilibrium
    from gspack.profiles import ConstrainBetapIp
    from gspack import picard
    import gspack.backend as bk

    _HAS_GSPACK = True
except ImportError:
    pass


# ───────────────────────────────────────────────────────────────────────
#  解析求解器 (保留为备用，当 gspack2 不可用时)
# ───────────────────────────────────────────────────────────────────────

class AnalyticFixedBoundarySolver:
    """解析备用求解器：与 GS 求解器适配器保持相同的 I/O 约定。

    此类有意保持简单：它允许项目在安装外部 `GS_solver` 软件包之前
    端到端运行。对于生产数据，请将 `solve()` 替换为对真正固定边界求解器
    的调用并返回相同的字典键。
    """

    def solve(self, R: np.ndarray, Z: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray | float]:
        """为一个平衡返回合成的 `psi` 场和轴元数据。"""
        geom = geometry_channels(R, Z, params)
        rho = geom["rho"]
        theta = geom["theta"]

        m = params["alpha_m"]
        n = params["alpha_n"]
        beta = params["betap"]
        ip_scale = params["Ip"] / 1.0e6

        core = np.clip(1.0 - rho**2, 0.0, None)
        shaping = 1.0 + 0.08 * params["delta"] * np.cos(theta) + 0.04 * (params["kappa"] - 1.0) * np.cos(2 * theta)

        psi_bar = (core ** (0.65 + 0.08 * m)) * (1.0 + 0.06 * beta * core ** (0.5 + 0.1 * n)) * shaping
        psi_bar = np.where(geom["mask"] > 0, psi_bar, 0.0).astype(np.float32)

        max_val = float(np.max(psi_bar)) if np.max(psi_bar) > 0 else 1.0
        psi_bar = psi_bar / max_val

        psi_scale = 2.0e-2 * ip_scale * params["a"]
        axis_index = np.unravel_index(np.argmax(psi_bar), psi_bar.shape)
        return {
            "psi_bar": psi_bar.astype(np.float32),
            "psi": (psi_bar * psi_scale).astype(np.float32),
            "psi_lcfs": 0.0,
            "psi_axis": psi_scale,
            "R_axis": float(R[axis_index]),
            "Z_axis": float(Z[axis_index]),
            "plasma_mask": geom["mask"],
            "profile_params": np.array([0.0, 0.0], dtype=np.float32),
        }


# ───────────────────────────────────────────────────────────────────────
#  真正的 GS 求解器适配器 — 调用 gspack2_TRAE
# ───────────────────────────────────────────────────────────────────────

class GSSolverAdapter:
    """调用 gspack2_TRAE FixedBoundaryEquilibrium 进行真正的 GS 求解。

    此适配器创建固定边界平衡，使用 Picard 迭代求解，
    并将所有输出规格化为 GS_PINO 训练流水线期望的格式。

    Parameters
    ----------
    nr, nz : 求解器网格的 R、Z 点数。
    maxits : 最大 Picard 迭代次数。
    rtol   : 相对 psi 变化收敛容差。
    anderson_m : Anderson 加速窗口大小。
    """

    def __init__(
        self,
        nr: int = 65,
        nz: int = 65,
        maxits: int = 50,
        rtol: float = 5e-3,
        anderson_m: int = 5,
    ):
        if not _HAS_GSPACK:
            raise ImportError(
                "gspack2_TRAE 未找到。请安装 gspack2_TRAE 或使用 "
                "AnalyticFixedBoundarySolver 作为备用方案。"
            )
        self.nr = nr
        self.nz = nz
        self.maxits = maxits
        self.rtol = rtol
        self.anderson_m = anderson_m
        self._setup_grid = False
        bk.set_backend("cpu")

    def solve(self, params: dict[str, float]) -> dict[str, np.ndarray | float] | None:
        """求解一个固定边界 GS 平衡并返回所有需要的场。

        Returns
        -------
        dict 或 None（如果求解失败），包含：
          R, Z       : 2-D 计算网格
          psi        : 物理尺度极向磁通 (2-D)
          psi_bar    : 归一化磁通，LCFS 处为 0，磁轴处为 1
          psi_lcfs   : LCFS 处的 psi 值
          psi_axis   : 磁轴处的 psi 值
          R_axis     : 磁轴的 R 坐标
          Z_axis     : 磁轴的 Z 坐标
          plasma_mask: LCFS 内部的二值掩码 (1 = 内部)
          profile_params: [L, Beta0] — 用于训练时的 GS 残差 / Ip 约束
        """
        R0 = float(params["R0"])
        a = float(params["a"])
        kappa = float(params["kappa"])
        delta = float(params["delta"])
        Ip = float(params["Ip"])
        betap = float(params["betap"])
        alpha_m = float(params["alpha_m"])
        alpha_n = float(params["alpha_n"])

        # 适配域：1.3 倍等离子体尺寸，最小边距 0.15 m
        margin = max(0.15, 0.3 * a)
        Rmin = R0 - a - margin
        Rmax = R0 + a + margin
        Zmax_val = kappa * a + margin
        Zmin = -Zmax_val
        Zmax_val_ = Zmax_val

        try:
            eq = FixedBoundaryEquilibrium(
                R0=R0, a=a, kappa=kappa, delta=delta,
                Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax_val_,
                nx=self.nr, ny=self.nz, order=2, method="lu",
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
                    maxits=self.maxits, rtol=self.rtol,
                    anderson_m=self.anderson_m,
                    convergenceInfo=True, verbose=False,
                )

            # --- 提取物理 psi 并归一化 ---
            # fix_bndry_zero=True 使得 eq.psi() 中 LCFS 上 psi = 0
            psi_native = np.asarray(eq.psi(), dtype=np.float32)
            psi_bndry = float(eq.psi_bndry)       # 0.0 with fix_bndry_zero
            psi_axis_val = float(eq.psi_axis)     # 已平移：原始 psi_axis - 原始 psi_bndry
            dpsi = psi_axis_val - psi_bndry

            if abs(dpsi) < 1e-30:
                return None

            # psi_bar: LCFS = 0, axis = 1
            psi_bar = (psi_native - psi_bndry) / dpsi
            psi_bar = np.clip(psi_bar, 0.0, None).astype(np.float32)

            # 磁轴位置
            R_axis, Z_axis, _ = eq.magneticAxis()

            # 等离子体 LCFS 内部掩码
            plasma_mask = eq.plasma_mask.astype(np.float32)

            # 轮廓参数：用于 GS 残差中的 p' 和 FF'
            # p'(ψN) = L·Beta0/Raxis · (1-ψN^αm)^αn
            # FF'(ψN) = μ₀·L·(1-Beta0)·Raxis · (1-ψN^αm)^αn
            profile_params = np.array([float(pro.L), float(pro.Beta0)], dtype=np.float32)

            return {
                "R": np.asarray(eq.R, dtype=np.float32),
                "Z": np.asarray(eq.Z, dtype=np.float32),
                "psi_bar": psi_bar,
                "psi": psi_native,              # 已平移：LCFS = 0
                "psi_lcfs": psi_bndry,           # 0.0
                "psi_axis": psi_axis_val,        # 已平移
                "R_axis": float(R_axis),
                "Z_axis": float(Z_axis),
                "plasma_mask": plasma_mask,
                "profile_params": profile_params,
            }

        except Exception:
            return None

    def __call__(self, params: dict[str, float]) -> dict[str, np.ndarray | float] | None:
        return self.solve(params)
