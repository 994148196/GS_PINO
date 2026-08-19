#!/usr/bin/env bash
# KAN exp007: 只预测 psi（--j-weight 0）——隔离 J 对 psi 学习的干扰
# 动机（用户反馈, 2026-08-19）: 论文 200 epochs 已 0.63%，我们 200 ep 15.6%
#   （exp004），且 1024 点采样无改善（exp006）→ 采样/PDE/reg 均排除。
#   剩余嫌疑: LJψ 的梯度通过共享隐藏层干扰 psi——J 在分离面 mask 边界
#   跳变（光滑 B 样条逼近不连续 → 大系数/大梯度），或 psi 目标本身。
# 设置: 与 exp004 完全一致（DN-only / 论文架构 hidden10 grid2 / 1000 ep /
#   256 点/样本 / pde=0 reg=0），唯一差异 = --j-weight 0（损失只含 Lψ）。
# 对照: exp004（同设置 j-weight 1）→ 两者差 = J 联合训练的净干扰。
# exp001-006 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp007_psi_only.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp007_psi_only_kan_v5"
CKPT="$EXP/model_psionly_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp007 只预测 psi DN-only N=500 seed 1（论文架构, 1000 ep, j-weight 0）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 --pde-weight 0 --reg-weight 0 --j-weight 0 \
    --out-dir "$EXP/model_psionly_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp007.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_psionly_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
