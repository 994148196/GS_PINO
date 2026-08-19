#!/usr/bin/env bash
# KAN exp010: 只预测 psi_plasma（--target plasma）——线圈场用线性先验精确加回
# 动机（2026-08-19 数据验证）: MAST |psi_coils|/|psi_total| = 183%，lap(psi_coils)
#   奇点 27.9 vs lap(psi_plasma) 0.89（光滑 31 倍）——grid-2 样条（每区间 1.1m）
#   表达不了 Biot-Savart 线圈场，是 exp002-008 停在 11-15% 的直接根因。
#   psi_total = psi_plasma + psi_coils，greens@I 与 psi_coils 一致到 1e-7
#   （数据线性性已验证）→ 网络只学光滑等离子体场，评估时加回线圈场。
# 设置: 与 exp007 完全一致（DN-only / 论文架构 hidden10 grid2 / 1000 ep /
#   256 点/样本 / j=0 pde=0 reg=0 / N=500 seed 1），唯一差异 = --target plasma。
# 对照: exp007（--target total, grid2）→ 两者差 = 线圈场先验分解的净增益。
# exp001-009 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp010_plasma_target.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp010_plasma_target_kan_v5"
CKPT="$EXP/model_plasmatgt_dnonly_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp010 只预测 psi_plasma DN-only N=500 seed 1（论文架构 grid2, 1000 ep, j=0 pde=0 reg=0）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz" \
    --val-data "$V5/dn/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 --pde-weight 0 --reg-weight 0 --j-weight 0 \
    --target plasma \
    --out-dir "$EXP/model_plasmatgt_dnonly_kan_mix" \
    2>&1 | tee "$EXP/train_exp010.log" | tail -8
fi

echo "== 评估（DN test 120 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V5/dn/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_plasmatgt_dnonly_kan_mix/eval_dn" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
