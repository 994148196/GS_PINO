#!/usr/bin/env python
"""data_gspack2_v2 数据校验（S2，失败即停）。

用法（仓库根目录）:
    python dn_fno_2608/scripts/validate_g3.py

校验项：
  1. 网格逐点 array_equal data_v6（129² R[0.1,2] Z[-2,2]）
  2. 21ch 通道序（DNPinoDataset 加载 → x.shape[0] == 21；通道序对照
     CHANNEL_NAMES_COILS：R,Z + params[Ip,paxis,fvac,alpha_m,alpha_n] +
     14 单元电流 [Solenoid,Pc,Px,D1,D2,D3,Dp,D5,D6,D7,P4,P5,P61,P62]）
  3. greens 恒等式：Σ_k I_k·G_k ≡ psi_coils（抽查 50 样本/配置，max diff ≤1e-6）
  4. Ip 重构 rel err（verify_pde 同款；gspack 数据固有 ~5e-3 → 报告 + 宽松
     阈值 1e-2，与 data_gspack2_v1 口径一致）
  5. filter_v6 独立复核（对 data_gspack2_v2 跑 score_file：判据值 vs 生成器
     写入指标一致性；应移除 ~0 条；写 data_gspack2_v2/scores.json 入库）
  6. SN 质量对比 vs data_v6（midplane_ratio <0 样本、zaxis_ratio 均值、
     gs_true>15 样本、上瓣薄占比——全部应显著优于 v6 raw）

依赖：torch5060 环境 + PYTHONPATH src + gspack2_TRAE（DNPinoDataset 导入链）。
"""
import json
import sys
from pathlib import Path

import numpy as np

# Windows GBK 控制台无法打印 ²/× 等字符 → 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")

from gs_pino_fno_phys.data_pino import DNPinoDataset  # noqa: E402

sys.path.insert(0, str(ROOT / "dn_fno_2608" / "scripts"))
from filter_v6 import (  # noqa: E402
    gs_true as fv_gs_true,
    midplane_ratio as fv_midplane,
    score_file as fv_score_file,
)

CFGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]
G3 = ROOT / "dn_fno_2608" / "data_gspack2_v2"
V6 = ROOT / "dn_fno_2608" / "data_v6"
GREENS_N = 50
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def main() -> None:
    # ── 1. 网格对照 data_v6 ──
    print("== 1. 网格（vs data_v6）==")
    with np.load(V6 / "sn" / "test.npz") as d6, np.load(G3 / "sn" / "test.npz") as d3:
        ok = np.array_equal(d6["R"], d3["R"]) and np.array_equal(d6["Z"], d3["Z"])
        check("网格逐点一致", ok, f"R{d3['R'].shape} R[0,0]={d3['R'][0,0]:.4f} R[-1,-1]={d3['R'][-1,-1]:.4f}")
        check("129²", d3["R"].shape == (129, 129))

    # ── 2. 21ch 加载 ──
    # 注意: DNPinoDataset 基类只接受 str/list 路径（data_dn_fno_coils:104）→ 传 str
    print("== 2. 通道（21ch）==")
    for c in CFGS:
        ds = DNPinoDataset(str(G3 / c / "test.npz"), indices=np.arange(5))
        item = ds[0]
        check(f"{c}: 21ch", item["x"].shape[0] == 21, f"x{item['x'].shape}")
        check(f"{c}: psi 129²", item["y_psi"].shape[1:] == (129, 129),
              f"y_psi{item['y_psi'].shape}")
    # 通道序: R,Z 恒等（索引 0/1），params 前 5 列
    ds = DNPinoDataset(str(G3 / "dn" / "test.npz"), indices=np.arange(5))
    with np.load(G3 / "dn" / "test.npz") as d:
        p0 = d["params"][0]
    x = ds[0]["x"]
    sm = ds.stats["scalar_mean"]
    ss = ds.stats["scalar_std"]
    # scalar 通道广播到网格 → 取任意单点 (0,0) 比对标量值；
    # x 的 scalar 顺序 = [params(5), coils(14)] → x[2:7] 对应 sm[0:5]
    assert np.allclose(x[2:7, 0, 0].numpy(), (p0 - sm[0:5]) / (ss[0:5] + 1e-8), atol=1e-4)
    check("params 通道序 [Ip,paxis,fvac,alpha_m,alpha_n]", True)

    # ── 3. greens 恒等式 ──
    print("== 3. greens 恒等式（50 样本/配置）==")
    for c in CFGS:
        with np.load(G3 / c / "train.npz") as d:
            N = d["params"].shape[0]
            idx = np.linspace(0, N - 1, GREENS_N).astype(int)
            g = d["greens"][idx]
            icoil = d["coil_currents"][idx]
            pcoil = d["psi_coils"][idx]
            est = np.einsum("nijk,ni->njk", g, icoil)
            md = float(np.abs(est - pcoil).max())
            check(f"{c}: greens 恒等式", md <= 1e-6, f"max diff={md:.2e}")

    # ── 4. Ip 重构 ──
    print("== 4. Ip 重构 ==")
    for c in CFGS:
        ds = DNPinoDataset(str(G3 / c / "train.npz"), indices=np.arange(100))
        ip_est = (ds.j_phys * (ds.mask > 0.5)).sum(axis=(1, 2)) * ds.dR * ds.dZ
        rel = np.abs(ip_est - ds.params[:, 0]) / ds.params[:, 0]
        check(f"{c}: Ip 重构", rel.mean() < 1e-2,
              f"mean rel={rel.mean():.2e} (gspack 固有 ~5e-3)")

    # ── 5. filter_v6 独立复核 ──
    print("== 5. filter_v6 复核（data_gspack2_v2 判据重算）==")
    scores = {}
    for c in CFGS:
        for split in ["train", "val", "test"]:
            _, N, bad_idx, vals = fv_score_file(c, split, src=G3)
            if bad_idx:
                print(f"  [FAIL] {c}/{split}: {len(bad_idx)} 条判据命中 {bad_idx[:5]}")
                FAIL.append(f"filter_v6 {c}/{split}")
            scores[f"{c}/{split}"] = {
                "n": N, "n_removed": len(bad_idx), "removed_idx": bad_idx,
                # 仅记录被移除样本的判据值（期望 0 条 → 文件小）
                "removed_vals": [{k: round(v[k], 4) for k in v} for i, v
                                 in enumerate(vals) if i in bad_idx],
            }
    (G3 / "scores.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    print(f"  [OK] 15 split 判据命中 0 条（scores.json 已写 {G3/'scores.json'}）")

    # ── 6. SN 质量对比 vs data_v6 ──
    print("== 6. SN 质量对比 vs data_v6（test 200 口径）==")
    with np.load(G3 / "sn" / "test.npz") as d3, np.load(V6 / "sn" / "test.npz") as d6:
        # midplane_ratio（v6 无新字段 → filter_v6 公式重算；g3 用生成器写入值）
        def mid_arr(d):
            if "midplane_ratio" in d.files:
                return d["midplane_ratio"]
            return np.array([fv_midplane(d["psi_total"][i], d["R"], d["Z"], d["axes"][i])
                             for i in range(d["params"].shape[0])])

        mid3, mid6 = mid_arr(d3), mid_arr(d6)
        print(f"  g3: midplane<0 = {(mid3 < 0).sum()}/200, mean={mid3.mean():.3f}, min={mid3.min():.3f}")
        print(f"  v6: midplane<0 = {(mid6 < 0).sum()}/200, mean={mid6.mean():.3f}, min={mid6.min():.3f}")
        check("SN 病态 0 条（midplane<0）", float((mid3 < 0).sum()) == 0)
        # zaxis: |Z_axis|/|Z_lo|（x_coords lo 行 = [loR, loZ]；sn 上对占位 0）
        for tag, d in [("g3", d3), ("v6", d6)]:
            zlo = np.abs(d["x_coords"][:, 1])
            zax = np.abs(d["axes"][:, 1]) / np.maximum(zlo, 1e-6)
            print(f"  {tag}: |Z_axis|/|Z_lo| mean={zax.mean():.3f} max={zax.max():.3f} "
                  f"frac>0.5={(zax > 0.5).mean():.3f}")
        # gs_true（同口径重算）
        def gs_arr(d):
            return np.array([fv_gs_true(d["psi_total"][i], d["R"], d["Z"],
                                        d["dpdpsi"][i], d["FdFdpsi"][i], d["axes"][i, 2])
                             for i in range(d["params"].shape[0])])

        gs3, gs6 = gs_arr(d3), gs_arr(d6)
        print(f"  g3: gs_true>15 = {(gs3 > 15).sum()}/200, mean={gs3.mean():.2f}")
        print(f"  v6: gs_true>15 = {(gs6 > 15).sum()}/200, mean={gs6.mean():.2f}")
        check("SN 门后 test 全过（gs_true>15 = 0）", float((gs3 > 15).sum()) == 0)

    print("\n" + ("ALL CHECKS PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
