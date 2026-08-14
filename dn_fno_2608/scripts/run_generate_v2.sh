#!/usr/bin/env bash
# data_v2 dataset generation (profile-shape exponents sampled):
#   alpha_m ~ U[1.0, 2.0], alpha_n ~ U[1.5, 2.5]; everything else identical to
#   the paper baseline (data/): train 5000 (seed 123), val 500 (seed 456),
#   test 500 (seed 789). Chunked & resumable: re-running skips completed chunks.
set -e
cd "$(dirname "$0")/../.."   # repo root

PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
OUT="dn_fno_2608/data_v2"
JOBS=24
# full param resampling on rejection, 20 attempts -> 6000/6000 (v2: ~half the
# sampled (paxis,Ip,fvac,alpha) combos fail the L/Beta0 or DN checks, so plain
# retries with the same params never recover them)
FLAGS="--alpha-sampling --max-retries 20"

echo "=== generating val (500, seed 456) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v2_val.log

echo "=== generating test (500, seed 789) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test --n-samples 500 --seed 789 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v2_test.log

echo "=== generating train (5000, seed 123) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v2_train.log

echo "=== merging chunks ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val   --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test  --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir "$OUT" --merge

echo "=== done ==="
