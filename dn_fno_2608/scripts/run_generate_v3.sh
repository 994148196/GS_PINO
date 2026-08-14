#!/usr/bin/env bash
# data_v3 dataset generation (wider X-point jitter + sampled isoflux anchor):
#   X-points (1.1, +-0.6) +- 0.20 m (vs paper +-0.02 m)
#   anchor R ~ U[1.2, 1.8], Z ~ U[-0.3, 0.3] (vs fixed (1.5, 0.0))
#   alpha_m/alpha_n sampled as in data_v2 (5 params)
#   train 2000 (seed 123), val 500 (seed 456), test 500 (seed 789)
# Chunked & resumable: re-running skips completed chunks.
#
# NOTE: all splits MUST be generated with the same flags. Mixing old-flag chunks
# with new-flag chunks silently produces a merged npz where some samples lack
# the "anchor" field (see data_v2 README for the same accident history).
set -e
cd "$(dirname "$0")/../.."   # repo root

PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
OUT="dn_fno_2608/data_v3"
JOBS=24
FLAGS="--alpha-sampling --xpt-jitter 0.20 --isoflux-sampling --max-retries 20"

echo "=== generating val (500, seed 456) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v3_val.log

echo "=== generating test (500, seed 789) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test --n-samples 500 --seed 789 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v3_test.log

echo "=== generating train (2000, seed 123) ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 2000 --seed 123 \
    --out-dir "$OUT" --chunk-size 500 --n-jobs "$JOBS" $FLAGS | tee dn_fno_2608/logs/gen_v3_train.log

echo "=== merging chunks ==="
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val   --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test  --out-dir "$OUT" --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir "$OUT" --merge

echo "=== done ==="
