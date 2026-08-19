#!/usr/bin/env bash
# KAN M0 冒烟：单测（model/data）+ N=100 短训练（四阶段走完）+ 剪枝检查
#              + best.pt round-trip（evaluate + visualize）
# 用法: bash dn_fno_2608/kan/scripts/run_kan_smoke.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
OUT="dn_fno_2608/kan/experiments/exp001_kan_v5/smoke"
mkdir -p "$OUT"

echo "== 0. 单测: model_kan (B 样条 vs scipy / 解析导 vs 中心差分 / 剪枝语义) =="
"$PY" -m gs_pino_kan_2608.model_kan
echo "== 1. 单测: data_kan (stats / J 目标 / 采样 / 全网格) =="
"$PY" -m gs_pino_kan_2608.data_kan

echo "== 2. 训练冒烟 N=100 / 50 轮（强制走完监督->半监督->剪枝->微调四阶段）=="
"$PY" -u -m gs_pino_kan_2608.train_kan \
  --train-data "$V5/dn/train.npz,$V5/sn/train.npz" \
  --val-data "$V5/dn/val.npz,$V5/sn/val.npz" \
  --n-train 100 --epochs 50 --steps-per-epoch 8 --val-every 10 \
  --out-dir "$OUT/model_smoke" 2>&1 | tee "$OUT/train_smoke.log" | tail -12

echo "== 3. round-trip: evaluate_kan（smoke ckpt）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/sn/test.npz" --checkpoint "$OUT/model_smoke/best.pt" \
  --out-dir "$OUT/eval_sn" 2>&1 | tail -6

echo "== 4. round-trip: visualize_kan（smoke ckpt）=="
"$PY" -u -m gs_pino_kan_2608.visualize_kan \
  --test-data "$V5/sn/test.npz" --checkpoint "$OUT/model_smoke/best.pt" \
  --out-dir "$OUT/figures_sn" --machine mast --max-samples 4 \
  --title "KAN smoke (N=100/50ep) - SN" 2>&1 | tail -5

echo "== smoke OK: 产物在 $OUT =="
ls "$OUT/model_smoke" "$OUT/eval_sn" "$OUT/figures_sn"
