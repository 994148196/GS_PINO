#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m gs_pino.generate_dataset --out data/smoke_gs.npz --n-samples 16 --nr 32 --nz 32
python -m gs_pino.train --data data/smoke_gs.npz --epochs 2 --batch-size 2 --width 8 --modes1 4 --modes2 4 --layers 1 --output-dir outputs/smoke
python -m gs_pino.evaluate --data data/smoke_gs.npz --checkpoint outputs/smoke/best.pt --output-dir outputs/smoke_eval --max-plots 3
