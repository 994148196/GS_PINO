#!/usr/bin/env bash
# exp013: data_v6_clean 清洗数据重训（与 exp012 完全同设置，仅数据源换 clean）
# 对照: exp012 (data_v6 原数据, 五配置混合, N=500 seed 1, 800 epochs)
# 用法: bash dn_fno_2608/scripts/run_exp013_train.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
CLEAN="dn_fno_2608/data_v6_clean"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data "$CLEAN/dn/train.npz,$CLEAN/sn/train.npz,$CLEAN/snow_single/train.npz,$CLEAN/snow_double/train.npz,$CLEAN/limiter/train.npz" \
  --val-data "$CLEAN/dn/val.npz,$CLEAN/sn/val.npz,$CLEAN/snow_single/val.npz,$CLEAN/snow_double/val.npz,$CLEAN/limiter/val.npz" \
  --input-mode coils --no-config-channel --n-train 500 --seed 1 \
  --epochs 800 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 \
  --lr-patience 20 --lr-factor 0.5 --min-lr 1e-5 --patience 75 \
  --out-dir dn_fno_2608/experiments/exp013_clean_v6
echo "== exp013 train done =="
