#!/usr/bin/env bash
# KAN exp008: grid 8（空间分辨率 4x）纯监督只预测 psi——验证样条分辨率假说
# 动机（2026-08-19 重新思考）: 200 ep 15.6%（exp004）、1024 点无改善（exp006）
#   → 采样/信息非瓶颈；等离子体细节非淹没（psi_plasma rms 2.1σ）。新根因:
#   B 样条是参数方向基（R/Z 各 2 区间），空间分辨率 = 每区间 ~1.1m，而
#   65×65 网格每格 3.4cm、线圈奇性半径 0.12m ≈ 0.05 个区间——2 区间二次
#   样条 ≈ 全局三次多项式，表达不了 11 线圈 Biot-Savart 场（论文 EXL-50U
#   仅 5 独立对称线圈、形状简单）。grid 2→8：空间分辨率 4x。
# 设置: 与 exp007 完全一致（DN-only / hidden10 / 纯监督 j=0 pde=0 reg=0 /
#   256 点/样本 / 1000 ep），唯一差异 = --grid-size 8。
# 对照: exp007（grid 2）→ 两者差 = 样条空间分辨率的净增益。
# 仍不足则 grid 16 / 2048 点；exp001-007 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp008_grid8.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp008_grid8_kan_v5"
CKPT="$EXP/model_grid8_psi_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp008 grid8 只预测 psi DN-only N=500 seed 1（hidden10 grid8, 1000 ep, j=0 pde=0 reg=0）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 8 --pde-weight 0 --reg-weight 0 --j-weight 0 \
    --out-dir "$EXP/model_grid8_psi_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp008.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_grid8_psi_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
