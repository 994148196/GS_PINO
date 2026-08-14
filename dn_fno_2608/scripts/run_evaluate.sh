#!/usr/bin/env bash
# Evaluate best models + latency benchmark for the scaling study.
set -e
cd "$(dirname "$0")/../.."   # repo root

PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
TEST="dn_fno_2608/data/test.npz"
OUT="dn_fno_2608/outputs"
REPORT="dn_fno_2608/outputs/report"

for N in 500 1000 2000 5000; do
  for S in 1 2 3; do
    CKPT="$OUT/fno_n${N}_s${S}/best.pt"
    if [ -f "$CKPT" ]; then
      echo "=== evaluate N=$N seed=$S ==="
      "$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno --test-data "$TEST" --checkpoint "$CKPT" \
        --out-dir "$REPORT/n${N}_s${S}" 2>&1 | tail -30
    else
      echo "skip N=$N seed=$S (no checkpoint)"
    fi
  done
done

echo "=== latency: best model (N=5000 seed=1, if present) ==="
CKPT="$OUT/fno_n5000_s1/best.pt"
if [ -f "$CKPT" ]; then
  "$PY" -u -m gs_pino_dn_fno_2608.latency_dn_fno --test-data "$TEST" --checkpoint "$CKPT" \
    --out-dir "$REPORT" 2>&1 | tail -20
fi
echo "=== evaluation done ==="
