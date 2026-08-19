#!/usr/bin/env bash
# KAN M1: N=500 seed 1 四阶段训练（1000 epochs）+ 六桶评估 + 可视化
# 用法: bash dn_fno_2608/kan/scripts/run_kan_m1_train_eval_vis.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp001_kan_v5"
CKPT="$EXP/model_b19ch_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== 1. 训练 N=500 seed 1（四阶段: 监督->半监督->剪枝->微调, 1000 ep）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz,$V5/sn/train.npz" \
    --val-data "$V5/dn/val.npz,$V5/sn/val.npz" \
    --n-train 500 --seed 1 --out-dir "$EXP/model_b19ch_kan_mix" \
    2>&1 | tee "$EXP/train_m1.log" | tail -8
fi

echo "== 2. eval_all (n=1000 混合整体) =="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz,$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b19ch_kan_mix/eval_all" 2>&1 | tail -5
echo "== 3. eval_dn (n=500) =="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b19ch_kan_mix/eval_dn" 2>&1 | tail -5
echo "== 4. eval_sn (n=500) =="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b19ch_kan_mix/eval_sn" 2>&1 | tail -5

echo "== 5. figures_dn =="
"$PY" -u -m gs_pino_kan_2608.visualize_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b19ch_kan_mix/figures_dn" --machine mast \
  --title "exp001 KAN 19ch mixed DN+SN (N=500) - DN test" 2>&1 | tail -3
echo "== 6. figures_sn =="
"$PY" -u -m gs_pino_kan_2608.visualize_kan \
  --test-data "$V5/sn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_b19ch_kan_mix/figures_sn" --machine mast \
  --title "exp001 KAN 19ch mixed DN+SN (N=500) - SN test" 2>&1 | tail -3
echo "== done: 回填 $EXP/README.md §3 对比表 =="
