#!/usr/bin/env bash
# KAN exp005: 修正 LPDE（plasma 语义）后的全阶段实验
# 动机（用户反馈 2, 2026-08-19）: "PDE 残差不包含 coil"——旧 total 语义把
#   线圈奇性的中心差分涂抹留在残差里（真值场自检: 真空区残差 0.406 vs
#   修正后 0.0023），Lψ 与 LPDE 在真空区互相矛盾。
# 设置: 与 exp003 完全一致（DN-only / 论文架构 hidden10 grid2 / 1000 ep /
#   全阶段 PDE+Ip+reg+剪枝），唯一差异 = --pde-coils-model plasma（新默认）。
#   对照: exp003（同设置但 total 语义）→ 两者差 = LPDE 修正的纯增益。
# exp001-004 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp005_pde_plasma.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp005_pde_plasma_kan_v5"
CKPT="$EXP/model_pdeplasma_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp005 DN-only N=500 seed 1（论文架构 19ch→10→2, 1000 ep, LPDE=plasma）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 \
    --out-dir "$EXP/model_pdeplasma_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp005.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_pdeplasma_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
