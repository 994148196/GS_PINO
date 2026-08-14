"""Latency benchmark (paper Table III) for the DN FNO surrogate.

  - FNO on GPU: batch 1, 500 test samples, synchronized torch.cuda.Event timing
  - FNO on CPU: batch 1, 500 samples, single PyTorch thread, perf_counter
  - FREEGS baseline: re-solves the 500 test samples with the SAME solver config
    as data generation (Picard, gamma=1e-12, isoflux->midplane, maxits=50),
    per-solve perf_counter, sequential
  - reports median / p95 / p95-to-median ratio / speedup (local hardware; the
    paper's absolute numbers are A100 SXM4-40GB and not directly comparable)

Usage:
  python -m gs_pino.latency_dn_fno --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s3/best.pt \
      --out-dir dn_fno_2608/outputs/report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gs_pino.data_dn_fno import DNFnoDataset
from gs_pino.model_dn_fno import build_model

import freegs
from freegs import boundary, control, critical, jtor

# same solver config as data generation (paper: "identical solver configuration")
RMIN, RMAX, ZMIN, ZMAX = 0.1, 2.0, -2.0, 2.0
ISOFLUX_REF = (1.5, 0.0)
GAMMA, RTOL, MAXITS = 1e-12, 1e-3, 50


def freegs_baseline(params: np.ndarray, x_coords: np.ndarray) -> list[float]:
    """Sequential FREEGS solves of the test params; returns per-solve seconds."""
    times = []
    for i in range(len(params)):
        Ip, paxis, fvac = params[i]
        lo, up = tuple(x_coords[i, :2]), tuple(x_coords[i, 2:])
        t0 = time.perf_counter()
        tokamak = freegs.machine.TestTokamak()
        eq = freegs.Equilibrium(tokamak=tokamak, Rmin=RMIN, Rmax=RMAX,
                                Zmin=ZMIN, Zmax=ZMAX, nx=65, ny=65,
                                boundary=boundary.freeBoundaryHagenow)
        profiles = jtor.ConstrainPaxisIp(eq, paxis=float(paxis), Ip=float(Ip), fvac=float(fvac))
        constrain = control.constrain(
            xpoints=[lo, up],
            isoflux=[(*lo, *ISOFLUX_REF), (*up, *ISOFLUX_REF)],
            gamma=GAMMA)
        freegs.solve(eq, profiles, constrain, rtol=RTOL, maxits=MAXITS, show=False)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  freegs baseline: {i+1}/{len(params)}")
    return times


def bench_model(model: torch.nn.Module, ds: DNFnoDataset, device: str,
                n: int) -> list[float]:
    """Batch-1 inference times for n samples (ms)."""
    times = []
    model.eval()
    with torch.no_grad():
        if device == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            x0 = ds[0][0][None].to("cuda")
            for _ in range(20):  # warmup
                _ = model(x0)
            torch.cuda.synchronize()
            for i in range(n):
                x = ds[i][0][None].to("cuda")
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        else:
            torch.set_num_threads(1)
            model = model.to("cpu")
            for _ in range(20):
                _ = model(ds[0][0][None])
            for i in range(n):
                x = ds[i][0][None]
                t0 = time.perf_counter()
                _ = model(x)
                times.append((time.perf_counter() - t0) * 1e3)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--skip-freegs", action="store_true",
                    help="skip the FREEGS baseline (~10 min sequential)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    ds = DNFnoDataset(args.test_data, stats=stats)
    n = min(args.n, len(ds))

    result = {}

    # FNO GPU
    if torch.cuda.is_available():
        model = build_model().cuda()
        model.load_state_dict(ckpt["model_state"])
        gpu_times = bench_model(model, ds, "cuda", n)
        result["fno_gpu_ms"] = {
            "median": float(np.median(gpu_times)), "p95": float(np.percentile(gpu_times, 95))}
        print(f"FNO GPU (batch 1, {n} samples): median {result['fno_gpu_ms']['median']:.3f} ms, "
              f"p95 {result['fno_gpu_ms']['p95']:.3f} ms  (paper A100: 2.765 / 2.803)")
    else:
        print("no CUDA device, skipping GPU bench")

    # FNO CPU
    model = build_model().cpu()
    model.load_state_dict(ckpt["model_state"])
    cpu_times = bench_model(model, ds, "cpu", n)
    result["fno_cpu_ms"] = {
        "median": float(np.median(cpu_times)), "p95": float(np.percentile(cpu_times, 95))}
    print(f"FNO CPU (batch 1, {n} samples): median {result['fno_cpu_ms']['median']:.3f} ms, "
          f"p95 {result['fno_cpu_ms']['p95']:.3f} ms  (paper: 25.587 / 25.810)")

    # FREEGS baseline
    if not args.skip_freegs:
        print(f"\nFREEGS baseline: solving {n} test samples sequentially ...")
        t0 = time.perf_counter()
        fs_times = freegs_baseline(ds.params[:n], ds.x_coords[:n])
        result["freegs_ms"] = {
            "median": float(np.median(fs_times) * 1e3),
            "p95": float(np.percentile(fs_times, 95) * 1e3),
            "mean": float(np.mean(fs_times) * 1e3),
            "std": float(np.std(fs_times) * 1e3),
            "total_s": float(time.perf_counter() - t0)}
        print(f"FREEGS CPU: median {result['freegs_ms']['median']:.1f} ms, "
              f"p95 {result['freegs_ms']['p95']:.1f} ms, std {result['freegs_ms']['std']:.1f} ms "
              f"(paper: 1768 / 2002 / 92)")

    if "fno_gpu_ms" in result and "freegs_ms" in result:
        result["speedup_gpu"] = float(
            result["freegs_ms"]["median"] / result["fno_gpu_ms"]["median"])
    if "freegs_ms" in result:
        result["speedup_cpu"] = float(
            result["freegs_ms"]["median"] / result["fno_cpu_ms"]["median"])
    result["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "paper_gpu": "NVIDIA A100 SXM4-40GB",
        "note": "absolute latencies are hardware-specific; methodology matches the paper",
    }
    print(f"\nspeedup: GPU {result.get('speedup_gpu', float('nan')):.0f}x "
          f"(paper ~640x) | CPU {result.get('speedup_cpu', float('nan')):.0f}x "
          f"(paper ~69x)  [skipped if freegs baseline not run]")

    with open(out_dir / "latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out_dir}/latency.json")


if __name__ == "__main__":
    main()
