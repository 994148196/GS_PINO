#!/usr/bin/env bash
# Full DN dataset generation per paper 2608.05555:
#   train 5000 (seed 123), val 500 (seed 456), test 500 (seed 789)
# Chunked & resumable: re-running skips completed chunks.
set -e
cd "$(dirname "$0")/../.."   # repo root

PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
OUT="dn_fno_2608/data"
JOBS=24

echo "=== generating val (500, seed 456) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" | tee dn_fno_2608/logs/gen_val.log

echo "=== generating test (500, seed 789) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test --n-samples 500 --seed 789 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" | tee dn_fno_2608/logs/gen_test.log

echo "=== generating train (5000, seed 123) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" | tee dn_fno_2608/logs/gen_train.log

echo "=== merging chunks ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val   --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test  --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir "$OUT" --merge

echo "=== done ==="
