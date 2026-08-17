#!/usr/bin/env bash
# exp011: coil 18ch 混合训练（DN+SN）评估 + 可视化
# 用法: bash dn_fno_2608/scripts/run_exp011_eval_vis.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/experiments/exp011_coil_input_v5"
CKPT="$EXP/model_b18ch_coils_mix/best.pt"

echo "== eval_all (n=1000 混合整体) =="
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data "$V5/dn/test.npz,$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b18ch_coils_mix/eval_all" 2>&1 | tail -4
echo "== eval_dn (n=500) =="
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b18ch_coils_mix/eval_dn" 2>&1 | tail -4
echo "== eval_sn (n=500) =="
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data "$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b18ch_coils_mix/eval_sn" 2>&1 | tail -4

echo "== figures_dn =="
"$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b18ch_coils_mix/figures_dn" --machine mast \
  --title "exp011 coil 18ch mixed DN+SN (N=500) - DN test" 2>&1 | tail -4
echo "== figures_sn =="
"$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
  --test-data "$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b18ch_coils_mix/figures_sn" --machine mast \
  --title "exp011 coil 18ch mixed DN+SN (N=500) - SN test" 2>&1 | tail -4
echo "== done =="
