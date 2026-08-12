"""Generate one-click reproduce scripts from historical checkpoint args.

Reads every outputs/<run>/best.pt, reconstructs the exact CLI invocation
(train_freegs.py for free-boundary runs, train.py for fixed-boundary runs),
and emits:

    scripts/reproduce_freegs.sh   — free-boundary experiments (functions)
    scripts/reproduce_fixed.sh    — fixed-boundary experiments (functions)

Each experiment is a shell function run_<run>(); run with:

    bash scripts/reproduce_freegs.sh                 # run ALL experiments
    bash scripts/reproduce_freegs.sh plasma_coil     # run a single experiment

Re-run this generator any time new experiments are added:
    python scripts/gen_reproduce.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
SCRIPTS = REPO / "scripts"

# train_freegs.py CLI: checkpoint key -> (flag, formatting)
FREEGS_FLAGS = [
    ("model", "--model", str),
    ("hidden_dim", "--hidden-dim", int),
    ("fourier_freqs", "--fourier-freqs", int),
    ("dropout", "--dropout", float),
    ("use_coil_input", "--use-coil-input", "flag"),
    ("predict_plasma", "--predict-plasma", "flag"),
    ("scale_mse", "--scale-mse", float),
    ("scale_pde", "--scale-pde", float),
    ("scale_axis", "--scale-axis", float),
    ("scale_ip", "--scale-ip", float),
    ("scale_curvature", "--scale-curvature", float),
    ("batch_size", "--batch-size", int),
    ("accum_steps", "--accum-steps", int),
    ("amp", "--amp", "bool"),
    ("lr", "--lr", float),
    ("warmup_epochs", "--warmup-epochs", int),
    ("epochs", "--epochs", int),
    ("min_epochs", "--min-epochs", int),
    ("patience", "--patience", int),
    ("weight_decay", "--weight-decay", float),
    ("clip_grad", "--clip-grad", float),
    ("noise_std", "--noise-std", float),
    ("seed", "--seed", int),
    ("data", "--data", str),
]

# train.py CLI (fixed-boundary): key -> (flag, formatting)
FIXED_FLAGS = [
    ("modes1", "--modes1", int),
    ("modes2", "--modes2", int),
    ("width", "--width", int),
    ("layers", "--layers", int),
    ("pde_weight", "--pde-weight", float),
    ("bc_weight", "--bc-weight", float),
    ("ip_weight", "--ip-weight", float),
    ("axis_weight", "--axis-weight", float),
    ("clip_grad", "--clip-grad", float),
    ("accum_steps", "--accum-steps", int),
    ("amp", "--amp", "bool"),
    ("warmup_epochs", "--warmup-epochs", int),
    ("patience", "--patience", int),
    ("min_epochs", "--min-epochs", int),
    ("seed", "--seed", int),
    ("epochs", "--epochs", int),
    ("batch_size", "--batch-size", int),
    ("lr", "--lr", float),
    ("data", "--data", str),
]


def fmt_value(v: float) -> str:
    """Format numbers compactly (1e-4 stays 1e-4, 1.0 stays 1.0)."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return repr(v)


def build_flags(args: dict, flags: list, module: str) -> list[str]:
    """Return CLI flag list reconstructed from checkpoint args."""
    out = [f"python -m {module}"]
    for key, flag, kind in flags:
        if key not in args:
            continue  # key absent in old checkpoints -> CLI default matches history
        v = args[key]
        if kind == "flag":  # BooleanOptionalAction: only emit when True
            if v:
                out.append(flag)
        elif kind == "bool":
            out.append(flag if v else "--no-" + flag.lstrip("--"))
        else:
            out.append(f"{flag} {fmt_value(v)}")
    return out


def metric_line(run_dir: Path, kind: str) -> str:
    """Pull the headline metric from test_metrics.json, if present."""
    mf = run_dir / "test_metrics.json"
    if not mf.exists():
        return ""
    try:
        m = json.loads(mf.read_text())
    except Exception:
        return ""
    if kind == "freegs" and "plasma_rel_l2" in m:
        p = m["plasma_rel_l2"]["mean"] * 100
        t = m.get("plasma_rel_l2_total", {}).get("mean", 0) * 100
        return f"  test: plasma rel L2 = {p:.2f}% (psi_total: {t:.2f}%)"
    if kind == "fixed" and "rel_l2" in m:
        return f"  test: rel L2 = {m['rel_l2'] * 100:.2f}%"
    return ""


def scan_runs() -> list[tuple[str, Path, str]]:
    """Return [(run_name, run_dir, kind)] sorted by checkpoint mtime."""
    runs = []
    for best in sorted(OUTPUTS.glob("*/best.pt")):
        run_dir = best.parent
        try:
            ck = torch.load(best, map_location="cpu", weights_only=False)
            args = ck.get("args", {})
        except Exception:
            continue
        if "scale_mse" in args or "hidden_dim" in args:
            kind = "freegs"
        elif "pde_weight" in args or "bc_weight" in args:
            kind = "fixed"
        else:
            continue
        runs.append((run_dir.name, run_dir, kind))
    runs.sort(key=lambda t: t[1].stat().st_mtime)
    return runs


def emit_script(kind: str, runs: list[tuple[str, Path, str]]) -> str:
    flags_map = FREEGS_FLAGS if kind == "freegs" else FIXED_FLAGS
    module = "gs_pino.train_freegs" if kind == "freegs" else "gs_pino.train"
    eval_module = "gs_pino.evaluate_freegs" if kind == "freegs" else "gs_pino.evaluate"
    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by scripts/gen_reproduce.py — {kind} experiments.",
        "# Usage: bash scripts/reproduce_freegs.sh [run1 run2 ...]",
        "#        (no args = run ALL experiments; run names from function list below)",
        "",
        "set -e",
        'export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"',
        "",
    ]
    for name, run_dir, rkind in runs:
        if rkind != kind:
            continue
        ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        args = dict(ck.get("args", {}))
        flags = build_flags(args, flags_map, module)
        outdir = str(run_dir).replace("\\", "/")
        # output-dir 不在 checkpoint args 的 CLI 映射里，必须显式追加，
        # 否则会落到 CLI 默认目录覆盖其它实验
        flags.append(f"--output-dir {outdir}")
        metric = metric_line(run_dir, kind)
        extra = ""
        if kind == "freegs" and args.get("predict_plasma"):
            extra = "  predict_plasma"
        if kind == "freegs" and args.get("use_coil_input"):
            extra += " + use_coil_input"
        lines += [
            f"run_{name}() {{",
            f"    # {run_dir}  ({datetime.fromtimestamp(run_dir.stat().st_mtime):%Y-%m-%d}){extra}",
            f"    #{metric}" if metric else "",
            f'    echo "=== {name} ==="',
            "    " + " \\\n    ".join(flags),
            f"    python -m {eval_module} --checkpoint {outdir}/best.pt --data {args.get('data', '')}",
            "}",
            "",
        ]
    lines += [
        "if [ $# -gt 0 ]; then",
        "    for exp in \"$@\"; do",
        "        case \"$exp\" in",
    ]
    for name, run_dir, rkind in runs:
        if rkind == kind:
            lines.append(f"            {name}) run_{name} ;;")
    lines += [
        "            *) echo \"unknown experiment: $exp\"; echo \"known: " + " ".join(n for n, _, k in runs if k == kind) + "\"; exit 1 ;;",
        "        esac",
        "    done",
        "else",
        "    " + "\n    ".join(f"run_{n}" for n, _, k in runs if k == kind),
        "fi",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    SCRIPTS.mkdir(exist_ok=True)
    runs = scan_runs()
    freegs_runs = [r for r in runs if r[2] == "freegs"]
    fixed_runs = [r for r in runs if r[2] == "fixed"]
    for kind, rlist, fname in [
        ("freegs", freegs_runs, "reproduce_freegs.sh"),
        ("fixed", fixed_runs, "reproduce_fixed.sh"),
    ]:
        (SCRIPTS / fname).write_text(emit_script(kind, rlist), encoding="utf-8")
        print(f"{fname}: {len(rlist)} experiments")
        for name, _, _ in rlist:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
