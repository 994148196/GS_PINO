"""Aggregate evaluation metrics into paper-style tables + a reproduction report.

Reads dn_fno_2608/outputs/fno_n{N}_s{s}/ (best.pt + history.json),
outputs/report/n{N}_s{s}/metrics.json and outputs/report/latency.json,
and writes outputs/report/REPORT.md with Table I/II/III comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # dn_fno_2608/
OUT = ROOT / "outputs"
REPORT = OUT / "report"

# paper Table I (3-seed mean +- std per N)
PAPER_TABLE_I = {
    500:  {"rel_l2": (0.286, 0.034), "rmse_norm": 3.13e-3, "rmse_phys": 8.53e-5},
    1000: {"rel_l2": (0.182, 0.002), "rmse_norm": 1.97e-3, "rmse_phys": 5.33e-5},
    2000: {"rel_l2": (0.109, 0.004), "rmse_norm": 1.18e-3, "rmse_phys": 3.16e-5},
    5000: {"rel_l2": (0.061, 0.006), "rmse_norm": 6.59e-4, "rmse_phys": 1.79e-5},
}
PAPER_TABLE_III = {
    "fno_gpu_ms": (2.765, 2.803), "fno_cpu_ms": (25.587, 25.810),
    "freegs_ms": (1768.0, 2002.0), "speedup_gpu": 640.0, "speedup_cpu": 69.0,
}
PAPER_TABLE_II = {
    "sep_mean_cm": (0.072, 0.059, 0.147), "sep_hausdorff_cm": (2.128, 0.967, 4.693),
    "sep_area_rel_err_pct": (0.465, 0.343, 0.992), "x_lo_cm": (0.161, 0.136, 0.350),
    "x_up_cm": (0.112, 0.097, 0.236), "o_point_cm": (0.031, 0.023, 0.079),
    "dpsi_bndry_Wb": (9.82e-6, None, 7.00e-5),
}


def load_metrics(path: Path) -> dict | None:
    p = path / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if v == v else "nan"


def main() -> None:
    lines = ["# arXiv:2608.05555 复现报告 (Double-Null FNO)",
             "", f"生成时间: 见各日志；本地硬件: GPU 见 latency.json",
             ""]

    # ---- Table I ----
    lines += ["## Table I — 场级精度 vs 训练集大小（复现: 可用 seed 均值±std；论文: 3-seed 均值±std）", "",
              "| N_train | relL2 复现 % | relL2 论文 % | RMSE_phys 复现 (Wb) | RMSE_phys 论文 (Wb) |",
              "|---|---|---|---|---|"]
    table_i = {}
    for n in (500, 1000, 2000, 5000):
        seeds = [load_metrics(REPORT / f"n{n}_s{s}") for s in (1, 2, 3)]
        seeds = [m for m in seeds if m]
        if not seeds:
            lines.append(f"| {n} | (未评估) | {PAPER_TABLE_I[n]['rel_l2'][0]} | | {PAPER_TABLE_I[n]['rmse_phys']:.2e} |")
            continue
        l2 = np.array([m["rel_l2_pct"]["mean"] for m in seeds])
        rmse = np.array([m["rmse_phys_Wb"]["mean"] for m in seeds])
        table_i[n] = {"l2_mean": l2.mean(), "l2_std": l2.std(), "rmse_mean": rmse.mean()}
        ours = f"{l2.mean():.3f}" + (f" ± {l2.std():.3f}" if len(seeds) > 1 else "")
        lines.append(
            f"| {n} | {ours} | {PAPER_TABLE_I[n]['rel_l2'][0]:.3f} ± "
            f"{PAPER_TABLE_I[n]['rel_l2'][1]:.3f} | {rmse.mean():.3e} | "
            f"{PAPER_TABLE_I[n]['rmse_phys']:.2e} |")

    # scaling law fit
    if len(table_i) == 4:
        xs = np.log(np.array(sorted(table_i)))
        ys = np.log(np.array([table_i[n]["l2_mean"] for n in sorted(table_i)]))
        slope = float(np.polyfit(xs, ys, 1)[0])
        lines += ["", f"缩放律拟合: ε ∝ N^{slope:.2f} （论文: N^-0.68）", ""]
    else:
        lines.append("")

    # ---- Table II (best model) ----
    best = load_metrics(REPORT / "n5000_s1")
    lines += ["## Table II — 几何指标（N=5000 seed=1；均值 | 中位 | P95，距离 cm）", "",
              "| 指标 | 复现 | 论文 |", "|---|---|---|"]
    if best:
        for k, (pm, pmed, pp95) in PAPER_TABLE_II.items():
            v = best["geometry"].get(k)
            if v is None:
                continue
            if "Wb" in k and v["mean"] < 1e-4:
                rep = f"{v['mean']:.2e} | {v['median']:.2e} | {v['p95']:.2e}"
            else:
                rep = f"{v['mean']:.4f} | {v['median']:.4f} | {v['p95']:.4f}"
            if pmed is None:
                pap = f"{pm:.2e} | | {pp95:.2e}"
            else:
                pap = f"{pm:.3f} | {pmed:.3f} | {pp95:.3f}"
            lines.append(f"| {k} | {rep} | {pap} |")
        lines.append("")

    # ---- residual + latency ----
    lines += ["## GS 残差诊断（归一化，论文 2.29；freegs 真值基线 2.29±0.06）", ""]
    if best:
        g = best["gs_residual"]
        lines.append(f"- 预测场: {g['pred']['mean']:.3f} (论文 2.29)")
        lines.append(f"- freegs 真值基线: {g['true_freegs']['mean']:.3f}")
        lines.append(f"- 比值: {g['ratio_pred_true']:.3f} (论文 0.998)")
        lines.append(f"- find_critical 失败: {best['n_find_critical_fail']}/{best['n_eval']} (论文 0)")
    lines.append("")

    lines += ["## Table III — 推理延迟（batch 1，500 样本；中位 | p95）", "",
              "| 方法 | 复现 (ms) | 论文 (ms) | 加速比 |", "|---|---|---|---|"]
    lat_path = REPORT / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        if "fno_gpu_ms" in lat:
            lines.append(f"| FNO GPU ({lat['hardware']['gpu']}) | {lat['fno_gpu_ms']['median']:.3f} | "
                         f"{lat['fno_gpu_ms']['p95']:.3f} | "
                         f"{PAPER_TABLE_III['fno_gpu_ms'][0]} | {PAPER_TABLE_III['fno_gpu_ms'][1]} | "
                         f"{lat.get('speedup_gpu', 0):.0f}x (论文 ~{PAPER_TABLE_III['speedup_gpu']:.0f}x) |")
        lines.append(f"| FNO CPU | {lat['fno_cpu_ms']['median']:.3f} | {lat['fno_cpu_ms']['p95']:.3f} | "
                     f"{PAPER_TABLE_III['fno_cpu_ms'][0]} | {PAPER_TABLE_III['fno_cpu_ms'][1]} | "
                     f"{lat.get('speedup_cpu', 0):.0f}x (论文 ~{PAPER_TABLE_III['speedup_cpu']:.0f}x) |")
        if "freegs_ms" in lat:
            lines.append(f"| FREEGS CPU | {lat['freegs_ms']['median']:.0f} | {lat['freegs_ms']['p95']:.0f} | "
                         f"{PAPER_TABLE_III['freegs_ms'][0]} | {PAPER_TABLE_III['freegs_ms'][1]} | — |")
        lines += ["", "> 注: 论文延迟为 A100 SXM4-40GB 实测，本机绝对数字不可直接比较，"
                   "方法论与加速比口径一致。", ""]
    else:
        lines.append("| (latency.json 未生成) | | | |")

    # ---- deviations / assumptions ----
    lines += ["## 与论文的偏差与假设记录", "",
              "- 参数量: 本实现 4,211,649 vs 论文 4,770,241（论文模块组装细节不可恢复，已按用户要求取近似）",
              "- 激活函数: GELU（neuraloperator 默认；论文未指明）",
              "- isoflux 参考点: (1.5, 0.0) m 固定外中平面（论文未给坐标；仅锚定 gauge）",
              "- 数据加载: workers=0（Windows spawn 稳定性；论文 4 workers，不影响结果）",
              "- 延迟绝对值: 本地 RTX 5060 vs 论文 A100，不可直接比",
              "- epoch 上限: 800（论文未说明）；复现中早停 patience 75 未触发（val 持续缓慢改善），"
              "全部跑满 800 epoch，论文最佳验证 epoch 310",
              "- 多 seed: 论文每档 3 seeds；按用户决定主要跑 4 次（每档 seed 1），"
              "N=500/1000 有额外 seed 2/3 可用",
              ""]

    (REPORT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved -> {REPORT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
