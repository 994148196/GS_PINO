#!/usr/bin/env bash
# End-to-end practical workflow: generate a larger dataset, train a U-FNO model,
# and write metrics plus detailed/aggregate evaluation figures.
set -euo pipefail

# Keep imports local when the package is not installed in editable mode.
export PYTHONPATH="${PYTHONPATH:-}:src"

# Centralized paths/knobs.  These values mirror configs/practical.yaml so the
# shell script can be used directly on a workstation or cluster login node.
DATA_PATH="${DATA_PATH:-data/gs_fixed_boundary_practical.npz}"
RUN_DIR="${RUN_DIR:-outputs/practical_run}"
EVAL_DIR="${EVAL_DIR:-outputs/practical_eval}"
N_SAMPLES="${N_SAMPLES:-4096}"
NR="${NR:-128}"
NZ="${NZ:-128}"
EPOCHS="${EPOCHS:-250}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python -m gs_pino.generate_dataset \
  --out "${DATA_PATH}" \
  --n-samples "${N_SAMPLES}" \
  --nr "${NR}" \
  --nz "${NZ}" \
  --seed 2026

python -m gs_pino.train \
  --data "${DATA_PATH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 5.0e-4 \
  --modes1 24 \
  --modes2 24 \
  --width 64 \
  --layers 5 \
  --pde-weight 0.02 \
  --bc-weight 0.10 \
  --output-dir "${RUN_DIR}"

python -m gs_pino.evaluate \
  --data "${DATA_PATH}" \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output-dir "${EVAL_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-plots 24
