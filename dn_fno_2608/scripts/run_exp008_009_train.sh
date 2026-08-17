#!/usr/bin/env bash
# exp008/exp009 training on data_v5 (N=500 seed 1, 延续惯例).
#   exp008 (mixed): MAST DN+SN 拼接训练, xa + config-input -> 14 ch
#   exp009 (split): DN-only / SN-only 分开训练, xa 13 ch (无 config 通道,
#                   SN-only 的 up 恒 0 通道由 std 防护自动忽略)
set -e
cd "$(dirname "$0")/../.."
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
DN_T="$V5/dn/train.npz"; SN_T="$V5/sn/train.npz"
DN_V="$V5/dn/val.npz";   SN_V="$V5/sn/val.npz"
E8="dn_fno_2608/experiments/exp008_mixed_configs"
E9="dn_fno_2608/experiments/exp009_split_configs"

echo "=== exp008: mixed DN+SN, xa 14ch + config ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa --config-input \
  --train-data "$DN_T,$SN_T" --val-data "$DN_V,$SN_V" \
  --n-train 500 --seed 1 --out-dir "$E8/model_a14ch_xa_mix" \
  2>&1 | tee dn_fno_2608/logs/exp008_train.log | tail -5

echo "=== exp009: DN-only (xa 13ch) ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data "$DN_T" --val-data "$DN_V" \
  --n-train 500 --seed 1 --out-dir "$E9/model_dn_13ch" \
  2>&1 | tee dn_fno_2608/logs/exp009_dn_train.log | tail -5

echo "=== exp009: SN-only (xa 13ch) ==="
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data "$SN_T" --val-data "$SN_V" \
  --n-train 500 --seed 1 --out-dir "$E9/model_sn_13ch" \
  2>&1 | tee dn_fno_2608/logs/exp009_sn_train.log | tail -5

echo "=== all training done ==="
