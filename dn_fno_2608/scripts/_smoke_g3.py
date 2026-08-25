"""临时冒烟：g3 生成器 3 样本/配置 + greens 恒等式 + 网格/通道检查。"""
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))
from gs_gspack2_dn_fno_2608 import generate_dn_g3_dataset as g3  # noqa: E402
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5  # noqa: E402

CFGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]

# 门全关（探针模式）+ 必要基线门
def cfg_for(name):
    return g3.build_cfg_g3(
        machine="mastu_g3", config=name,
        isoflux_sampling=True, anchor_midplane=True,
        max_isoflux_residual=0.35, max_xpt_deviation=0.10,
        min_anchor_xpt_dist=0.15, coil_margin=0.05, min_core_depth=0.005,
        save_constraint_diag=True,
        sn_midplane_ratio_min=None, sn_zaxis_ratio_max=None,
        max_gs_true=None, max_snow_xpt_dev=None)

print("=== UnitMachine 单元接口检查 ===")
tok = g3.make_mastu_unit()
print("物理线圈数:", len(tok.coils), "(expect 26)")
print("控制单元数:", len(tok.controlCurrents()), "(expect 14)")
tok.setControlCurrents([1.0] * 14)
cs = tok.controlCurrents()
print("set/get 一致:", np.allclose(cs, 1.0))
tok.controlAdjust(np.ones(14))
print("controlAdjust 后:", np.allclose(tok.controlCurrents(), 2.0))
print("P61 上下号:", tok._by_name["P61U"].current, tok._by_name["P61L"].current,
      "(expect 2.0, -2.0)")
print("Solenoid control:", tok._by_name["Solenoid"].control, "(expect False)")

print("\n=== 求解冒烟（3 样本/配置，门全关）===")
for name in CFGS:
    cfg = cfg_for(name)
    params = g5.sample_params(np.random.default_rng(123), alpha=True,
                              ranges=cfg["param_ranges"])
    n_ok = 0
    for i in range(3):
        r = g3._solve_one((params, 123 * 100_000 + i, cfg))
        if r is None:
            print(f"  {name} sample {i}: REJECTED")
            continue
        n_ok += 1
        # greens 恒等式: Σ I_k·G_k vs psi_coils
        diff = np.abs(np.einsum("k,kij->ij",
                                r["coil_currents"], r["greens"]) - r["psi_coils"]).max()
        print(f"  {name} sample {i}: OK |greens_id maxdiff={diff:.2e} "
              f"greens={r['greens'].shape} coils={r['coil_currents'].shape} "
              f"gs_true={float(r['gs_true'][0]):.3f} "
              f"midplane={float(r['midplane_ratio'][0]):.3f} "
              f"zaxis={float(r['zaxis_ratio'][0]):.3f} "
              f"config={int(r['config'][0])}")
        if diff > 1e-6:
            print(f"    !! greens identity FAILED")
    print(f"  {name}: {n_ok}/3 accepted")

print("\n=== 网格 ===")
eq0 = g3.make_mastu_unit()
# 用一个已构造 eq 拿网格（经 cfg 工厂）
import gspack
tok2 = g3.make_mastu_unit()
e0 = gspack.Equilibrium(tok2, Rmin=0.1, Rmax=2.0, Zmin=-2.0, Zmax=2.0,
                        nx=129, ny=129, order=2, method="lu")
print("R shape:", e0.R.shape, "Z shape:", e0.Z.shape)
print("R[0,0]..R[-1,0]:", e0.R[0, 0], e0.R[-1, 0])
print("Z[0,0]..Z[0,-1]:", e0.Z[0, 0], e0.Z[0, -1])
