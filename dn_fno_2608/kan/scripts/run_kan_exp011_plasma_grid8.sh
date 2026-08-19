#!/usr/bin/env bash
# KAN exp011: plasma 目标 × grid 8——两个正交杠杆叠加
# 动机（2026-08-19）: 独立杠杆已验证——
#   exp007→exp008: total 目标 grid2→grid8, 100ep 14.0%→7.2%（样条空间分辨率 4x）
#   exp007→exp010: grid2 total→plasma, 100ep 14.0%→5.1%（线圈场线性先验分解）
#   两者正交（一个改目标场、一个改基函数分辨率）→ 叠加应进一步下降。
# 设置: 与 exp010 完全一致（DN-only / 论文架构 hidden10 / 1000 ep / 256 点/
#   j=0 pde=0 reg=0 / N=500 seed 1 / --target plasma），唯一差异 = --grid-size 8。
# 对照: exp010（plasma grid2, pre-prune 2.30%）→ 两者差 = 分辨率在分解后的净增益。
# exp001-010 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp011_plasma_grid8.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp011_plasma_grid8_kan_v5"
CKPT="$EXP/model_plasmatgt8_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp011 plasma×grid8 DN-only N=500 seed 1（hidden10 grid8, 1000 ep, j=0 pde=0 reg=0）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 8 --pde-weight 0 --reg-weight 0 --j-weight 0 \
    --target plasma \
    --out-dir "$EXP/model_plasmatgt8_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp011.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_plasmatgt8_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
