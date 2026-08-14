#!/usr/bin/env bash
# exp004/exp005 three-model training chain on data_v3 (N=500 seed 1 each).
#   A  (11ch, X-points only)    -> exp004/model_a11ch_xpt
#   A' (13ch, X-points+anchor)  -> exp004/model_a13ch_xa
#   B  (11ch, coil currents)    -> exp005
set -e
cd "$(dirname "$0")/../.."
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
T="dn_fno_2608/data_v3/train.npz"; V="dn_fno_2608/data_v3/val.npz"
E4="dn_fno_2608/experiments/exp004_xpoints_anchor_v3"
E5="dn_fno_2608/experiments/exp005_coil_input_v3"

echo "=== A: X-points 11ch (degraded control) ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --train-data "$T" --val-data "$V" \
  --n-train 500 --seed 1 --out-dir "$E4/model_a11ch_xpt" 2>&1 | tee dn_fno_2608/logs/exp004_a11ch_train.log | tail -5

echo "=== A': X-points+anchor 13ch (upper bound) ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa --train-data "$T" --val-data "$V" \
  --n-train 500 --seed 1 --out-dir "$E4/model_a13ch_xa" 2>&1 | tee dn_fno_2608/logs/exp004_a13ch_train.log | tail -5

echo "=== B: coil currents 11ch ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils --train-data "$T" --val-data "$V" \
  --n-train 500 --seed 1 --out-dir "$E5" 2>&1 | tee dn_fno_2608/logs/exp005_train.log | tail -5

echo "=== all training done ==="
