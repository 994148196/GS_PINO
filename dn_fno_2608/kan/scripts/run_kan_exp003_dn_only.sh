#!/usr/bin/env bash
# KAN exp003: DN-only（单一配置）——论文原版架构，数据复杂度砍半
# 动机: exp001 (19ch 混合, 论文架构) 20.9% 与 exp002 (19ch 混合, 大容量)
# 均未达验收 ≤1.4%。用户方向: 只用 DN 数据先做简单的——检验瓶颈是
# "混合位形复杂度" 还是 "容量"。
# 设置: 与 exp001 完全一致的论文原版（hidden 10 / grid 2 / 1000 ep / λ 120,240,600），
#       仅训练/验证数据改为 dn/train.npz 单文件（config 通道成为常量 −1，
#       data_kan 零改动，_minmax 常量兜底）。
# 对照矩阵: exp001 = 论文架构×混合 | exp002 = 大容量×混合 | exp003 = 论文架构×DN-only
# exp001/exp002 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp003_dn_only.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp003_dn_only_kan_v5"
CKPT="$EXP/model_paperarch_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp003 训练 DN-only N=500 seed 1（论文原版架构 19ch→10→2, 1000 ep）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 \
    --out-dir "$EXP/model_paperarch_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp003.log" | tail -8
fi

echo "== 评估（DN test 120 行；SN 对本模型是 OOD，留给 M3 外推）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_paperarch_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md 对比表 =="
