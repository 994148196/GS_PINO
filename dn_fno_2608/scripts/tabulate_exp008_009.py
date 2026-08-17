#!/usr/bin/env python
"""Tabulate exp008/exp009 eval metrics.json files into the README comparison table.

Reads the per-bucket metrics.json produced by evaluate_dn_fno.py and prints the
markdown rows for exp008 README §3a / exp009 README §3a + §3b.

Usage: python dn_fno_2608/scripts/tabulate_exp008_009.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E8 = ROOT / "dn_fno_2608/experiments/exp008_mixed_configs/model_a14ch_xa_mix"
E9 = ROOT / "dn_fno_2608/experiments/exp009_split_configs"

BUCKETS = {
    # (label, path to metrics.json)
    "混合 DN桶": E8 / "eval_dn/metrics.json",
    "专职 DN": E9 / "model_dn_13ch/eval/metrics.json",
    "混合 SN桶": E8 / "eval_sn/metrics.json",
    "专职 SN": E9 / "model_sn_13ch/eval/metrics.json",
}

CROSS = {
    "专职 DN -> SN test": E9 / "model_dn_13ch/eval_cross_sn/metrics.json",
    "专职 SN -> DN test": E9 / "model_sn_13ch/eval_cross_dn/metrics.json",
    "混合 -> 整体 test": E8 / "eval_all/metrics.json",
}


def row(m: dict, label: str) -> str:
    rl = m["rel_l2_pct"]
    return (f"| {label} | {rl['mean']:.3f} | {rl['median']:.3f} | {rl['p95']:.3f} | "
            f"{m['rmse_phys_Wb']['mean']:.2e} | "
            f"{m['gs_residual']['ratio_pred_true']:.4f} | "
            f"{m['n_find_critical_fail']} | "
            f"{m['geometry']['sep_mean_cm']['mean']:.3f} |")


def main() -> None:
    print("\n### 3a. 混合 vs 专职（核心对比）")
    header = "| 指标 | 混合 DN桶 | 专职 DN | 混合 SN桶 | 专职 SN |"
    print("(见下方逐行)" if False else "")
    for label, path in BUCKETS.items():
        if not path.exists():
            print(f"# MISSING {path}")
            continue
        print(row(json.load(open(path)), label))

    print("\n### 交叉评估（rel L2 mean %）")
    print("| 模型 → 数据 | rel L2 mean % | median % | RMSE phys (Wb) |")
    for label, path in CROSS.items():
        if not path.exists():
            print(f"# MISSING {path}")
            continue
        m = json.load(open(path))
        print(f"| {label} | {m['rel_l2_pct']['mean']:.3f} | "
              f"{m['rel_l2_pct']['median']:.3f} | {m['rmse_phys_Wb']['mean']:.2e} |")


if __name__ == "__main__":
    main()
