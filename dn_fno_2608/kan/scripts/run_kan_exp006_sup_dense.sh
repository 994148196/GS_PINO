#!/usr/bin/env bash
# KAN exp006: 纯监督 + 采样密度 4x（points-per-sample 1024）
# 动机（用户反馈, 2026-08-19）: 监督 psi 误差必须很小（论文 KAN-1 0.631%），
#   20% 就往后做没有意义；exp002/003 监督期停在 15-21%，已停。根因假说:
#   每 epoch 每样本只采 256/4225 ≈ 6% 的空间点（真空区 1000 ep 只覆盖 2 遍），
#   信息供给不足 → 提高每样本点数到 1024。
# 设置: 与 exp004 完全一致（DN-only / 论文架构 hidden10 grid2 / 纯监督
#   pde=0 reg=0 / 1000 ep），唯一差异 = --points-per-sample 1024（256→1024）。
# 对照: exp004（256 点纯监督）→ 两者差 = 采样密度的净增益。
# 若 1024 仍不够 → 下一步 2048 或全网格监督（4225 点），或提容量（grid 6-8）。
# exp001-005 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp006_sup_dense.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp006_sup_dense_kan_v5"
CKPT="$EXP/model_supdense_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp006 纯监督 DN-only N=500 seed 1（论文架构, 1000 ep, pde=0 reg=0, 1024 点/样本）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 --pde-weight 0 --reg-weight 0 \
    --points-per-sample 1024 \
    --out-dir "$EXP/model_supdense_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp006.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_supdense_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
