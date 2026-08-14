#!/usr/bin/env bash
# Scaling study (paper Table I): N in {500,1000,2000,5000} x 1 seed = 4 runs
# (user decision 2026-08-13: 4 runs suffice, no multi-seed; paper used 3 seeds,
# deviation noted in the report).
# Nested subsets via fixed permutation (seed 12345), so larger N strictly contains smaller N.
set -e
cd "$(dirname "$0")/../.."   # repo root

PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
TRAIN="dn_fno_2608/data/train.npz"
VAL="dn_fno_2608/data/val.npz"
OUT="dn_fno_2608/outputs"

for N in 500 1000 2000 5000; do
  S=1
  CKPT="$OUT/fno_n${N}_s${S}/best.pt"
  if [ -f "$CKPT" ]; then
    echo "skip N=$N seed=$S (already trained)"
    continue
  fi
  echo "=== N=$N seed=$S ==="
  "$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train "$N" --seed "$S" \
    --out-dir "$OUT/fno_n${N}_s${S}" 2>&1 | tee "dn_fno_2608/logs/train_n${N}_s${S}.log"
done
echo "=== scaling study done ==="
