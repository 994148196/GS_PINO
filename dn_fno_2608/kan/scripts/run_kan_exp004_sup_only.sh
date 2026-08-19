#!/usr/bin/env bash
# KAN exp004: 纯监督（KAN-1 only）——验证阶段一上限
# 动机（用户反馈 1）: 第一阶段（纯 Lψ+LJψ）都学不好（exp002/exp003 前 120ep
#   psi 停在 ~0.016-0.026 z-score ≈ 15-21% val），后面 PDE/正则/剪枝都是徒劳。
#   先跑纯监督 1000 ep，回答: 监督拟合本身的天花板在哪？
# 设置: 与 exp003 完全一致（DN-only / 论文架构 hidden10 grid2 / 1000 ep），
#   仅 --pde-weight 0 --reg-weight 0（无半监督/正则/剪枝，纯式 5-6）。
# 对照: exp003（同设置 + PDE/Ip/reg/剪枝）→ 两者差 = 多阶段增益（或负收益）。
# exp001/002/003 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp004_sup_only.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp004_sup_only_kan_v5"
CKPT="$EXP/model_suponly_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp004 纯监督 DN-only N=500 seed 1（论文架构 19ch→10→2, 1000 ep, pde=0 reg=0）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 --pde-weight 0 --reg-weight 0 \
    --out-dir "$EXP/model_suponly_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp004.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_suponly_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
