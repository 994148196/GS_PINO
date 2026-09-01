#!/usr/bin/env python
"""exp203 sn 收敛质量分桶诊断（S0-1，机制确认）。

用 exp203 ckpt 在 g3 sn test（200 样本）上按 psi_relchange_final 分桶对比
per-sample rel L2（psi_total 域，与 evaluate_pino 同口径）：
  - 好桶 relchange < 1e-3（对齐 v6：0% 超过）
  - 差桶 relchange >= 1e-3（g3: 20.1%）
  - 细三分桶 <1e-3 / [1e-3,3e-3) / [3e-3,1e-2) 看单调性
  - spearman 相关 relchange vs rel L2

预期（若"收敛质量驱动误差"成立）：rel L2 随 relchange 单调上升，差桶显著
高于好桶（比值 >= 1.5 判绿）。

用法（仓库根目录）:
    python dn_fno_2608/scripts/exp203_diag_bucket.py
输出: experiments/exp203_pino_rhs_g3_n500/diag_bucket.json + 打印
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Windows GBK 控制台无法打印 ²/× 等字符 → 强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")

from gs_pino_dn_fno_2608.data_dn_fno import rel_l2_normalized  # noqa: E402
from gs_pino_fno_phys.data_pino import DNPinoDataset  # noqa: E402
from gs_pino_fno_phys.evaluate_pino import load_checkpoint  # noqa: E402
from gs_pino_fno_phys.losses_pino import denorm_psi  # noqa: E402

EXP203 = ROOT / "dn_fno_2608" / "experiments" / "exp203_pino_rhs_g3_n500"
CKPT = EXP203 / "best.pt"
SN_TEST = ROOT / "dn_fno_2608" / "data_gspack2_v2" / "sn" / "test.npz"
OUT = EXP203 / "diag_bucket.json"

GOOD_TH = 1e-3          # 好/差桶分界（v6 sn max relchange = 9.94e-4）
FINE_EDGES = [0.0, 1e-3, 3e-3, 1e-2]  # 细三分桶


def stats_pct(x: np.ndarray) -> dict[str, float]:
    return {"mean_pct": float(np.mean(x) * 100),
            "median_pct": float(np.median(x) * 100),
            "p90_pct": float(np.percentile(x, 90) * 100)}


def main() -> None:
    model, stats, mode, _ = load_checkpoint(str(CKPT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = DNPinoDataset(str(SN_TEST), stats=stats)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    n = len(ds)
    psi_t_mean, psi_t_std = float(stats["psi_mean"]), float(stats["psi_std"])

    rel2 = np.zeros(n, dtype=np.float64)
    # 顺序索引与 raw 行索引映射（与 evaluate_pino.py:134-136 同款）
    with torch.no_grad():
        start = 0
        for batch in loader:
            x = batch["x"].to(device)
            pred = model(x).cpu()
            psi_plasma_pred = denorm_psi(pred[:, 0:1], stats).numpy()[:, 0]
            for b in range(pred.shape[0]):
                i = start + b
                j = ds.indices[i]
                psi_tot_pred = psi_plasma_pred[b] + np.einsum(
                    "kij,k->ij", ds.greens[j], ds.coil_currents[j])
                psi_tot_true = ds.psi_total[j]
                rel2[i] = float(rel_l2_normalized(
                    torch.from_numpy(((psi_tot_pred - psi_t_mean) / psi_t_std)[None, None]),
                    torch.from_numpy(((psi_tot_true - psi_t_mean) / psi_t_std)[None, None])))
            start += batch["x"].shape[0]

    with np.load(str(SN_TEST)) as d:
        rc_all = np.asarray(d["psi_relchange_final"]).reshape(-1)
    rc = np.asarray(rc_all[ds.indices], dtype=np.float64)

    # ---- 分桶统计 ----
    good = rc < GOOD_TH
    bad = ~good
    fine_ids = np.digitize(rc, FINE_EDGES) - 1
    fine_ids = np.clip(fine_ids, 0, len(FINE_EDGES) - 2)

    res = {
        "n": int(n),
        "mode": mode,
        "relchange": {"mean": float(rc.mean()), "p90": float(np.percentile(rc, 90)),
                      "max": float(rc.max())},
        "rel_l2_overall_pct": {"mean": float(rel2.mean() * 100),
                               "median": float(np.median(rel2) * 100)},
        "buckets": {
            f"good_<{GOOD_TH:g}": {"n": int(good.sum()),
                                   **stats_pct(rel2[good])},
            f"bad_>={GOOD_TH:g}": {"n": int(bad.sum()),
                                   **stats_pct(rel2[bad])},
        },
        "fine_buckets": {
            f"[{FINE_EDGES[k]:g},{FINE_EDGES[k+1]:g})": {"n": int((fine_ids == k).sum()),
                                                         **stats_pct(rel2[fine_ids == k])}
            for k in range(len(FINE_EDGES) - 1)},
        "ratio_bad_over_good": float(rel2[bad].mean() / (rel2[good].mean() + 1e-12)),
        "spearman_rho": float(np.corrcoef(
            np.argsort(np.argsort(rc)), np.argsort(np.argsort(rel2)))[0, 1]),
    }

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print(f"sn test n={n} | mode={mode}")
    print(f"relchange: mean {res['relchange']['mean']:.2e}  p90 "
          f"{res['relchange']['p90']:.2e}  max {res['relchange']['max']:.2e}")
    print(f"overall rel L2 (psi_total): mean {res['rel_l2_overall_pct']['mean']:.3f}%  "
          f"median {res['rel_l2_overall_pct']['median']:.3f}%")
    print("\n两分桶:")
    for k, v in res["buckets"].items():
        print(f"  {k:18s} n={v['n']:3d}  mean {v['mean_pct']:.3f}%  "
              f"median {v['median_pct']:.3f}%  p90 {v['p90_pct']:.3f}%")
    print(f"\n细三分桶（单调性）:")
    for k, v in res["fine_buckets"].items():
        print(f"  {k:14s} n={v['n']:3d}  mean {v['mean_pct']:.3f}%  "
              f"median {v['median_pct']:.3f}%")
    print(f"\nratio bad/good = {res['ratio_bad_over_good']:.2f}   "
          f"spearman rho = {res['spearman_rho']:.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
