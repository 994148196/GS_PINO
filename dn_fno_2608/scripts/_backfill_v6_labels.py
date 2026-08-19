#!/usr/bin/env python
"""data_v6 标注字段补写工具。

首批 chunk 在 STACKED_KEYS 加入 wall_contact/wall_contact_excess/
inwall_sep_frac 之前生成，这三个字段未落盘。本工具从 chunk 内已有的
psi_total / mask / axes 用与 _acceptance_checks 完全相同的逻辑重算
（全部确定性，逐样本可复现），写回新 npz（原 chunk 保留为 .bak 由
run_generate_v6.sh 清理）。

用法（仓库根目录，需要 freegs_snow fork）:
    python dn_fno_2608/scripts/_backfill_v6_labels.py dn_fno_2608/data_v6
重算所有 <out>/<config>/<split>/chunk_*.npz；已有字段的 chunk 跳过。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g

LABEL_KEYS = ("wall_contact", "wall_contact_excess", "inwall_sep_frac")


def _wall_sample_points(machine) -> np.ndarray:
    """Same 60-pts-per-segment wall polygon sampling as _acceptance_checks."""
    Rw = np.asarray(machine.wall.R, dtype=np.float64)
    Zw = np.asarray(machine.wall.Z, dtype=np.float64)
    seg = []
    for i in range(len(Rw) - 1):
        t = np.linspace(0.0, 1.0, 60)
        seg.append(np.c_[Rw[i] + t * (Rw[i + 1] - Rw[i]),
                         Zw[i] + t * (Zw[i + 1] - Zw[i])])
    return np.vstack(seg)


def backfill_chunk(path: Path) -> tuple[str, int]:
    with np.load(path) as d:
        if "wall_contact" in d.files:
            return "skip", 0
        R = d["R"]
        Z = d["Z"]
        psi = d["psi_total"]                     # (N, ny, nx)
        mask = d["mask"]                         # (N, ny, nx)
        psi_bndry = d["axes"][:, 2]              # (N,)
        psi_axis = d["axes"][:, 3]               # (N,)
        axes = d["axes"]
        other = {k: d[k] for k in d.files if k not in ("psi_total", "mask", "axes")}
    n = psi.shape[0]

    machine = g.freegs.machine.MASTU_simple()
    wp = _wall_sample_points(machine)
    wRmin = float(np.asarray(machine.wall.R).min())

    wc = np.zeros(n, np.float32)
    wce = np.zeros(n, np.float32)
    iw = np.zeros(n, np.float32)
    for i in range(n):
        # --- wall contact (same as the generator's gate) ---
        maskf = RectBivariateSpline(R[:, 0], Z[0, :], mask[i])
        touch = maskf(wp[:, 0], wp[:, 1], grid=False) > 0.5
        if touch.any():
            wc[i] = 1.0
            psif = RectBivariateSpline(R[:, 0], Z[0, :], psi[i])
            exc = psif(wp[touch, 0], wp[touch, 1], grid=False) - psi_bndry[i]
            core = psi_axis[i] - psi_bndry[i]
            if core > 0:
                wce[i] = float(exc.max() / core)
        # --- wall-internal structure (crossing-edge fraction, numpy) ---
        ab = psi[i] >= psi_bndry[i]
        inw = R < wRmin
        he = inw[:, :-1] & inw[:, 1:] & (ab[:, :-1] != ab[:, 1:])
        ve = inw[:-1, :] & inw[1:, :] & (ab[:-1, :] != ab[1:, :])
        n_cross = int(he.sum()) + int(ve.sum())
        n_edges = int((inw[:, :-1] & inw[:, 1:]).sum()) + int((inw[:-1, :] & inw[1:, :]).sum())
        iw[i] = n_cross / n_edges if n_edges else 0.0

    out = dict(other)
    out.update({"psi_total": psi, "mask": mask, "axes": axes,
                "wall_contact": wc, "wall_contact_excess": wce,
                "inwall_sep_frac": iw})
    np.savez(path, **out)
    return "backfilled", n


def main() -> None:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "dn_fno_2608/data_v6")
    n_ok = n_skip = 0
    for chunk in sorted(base.glob("*/[tv]*/chunk_*.npz")):
        status, cnt = backfill_chunk(chunk)
        if status == "backfilled":
            print(f"backfilled {chunk} ({cnt} samples)")
            n_ok += 1
        else:
            n_skip += 1
    print(f"done: {n_ok} backfilled, {n_skip} skipped (already have labels)")


if __name__ == "__main__":
    main()
