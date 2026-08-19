#!/usr/bin/env bash
# KAN exp002: 容量增强变体（hidden 24 / grid 4 / 2000 ep）—— M1 后置实验
# 动机: exp001 M1 在 ~19-20% val rel L2 平台（验收 ≤1.4% 未达标），容量受限
# 偏差（已在 exp002 README 记录）: 论文 arch 1 隐含层×10 → 24；B 样条
# 2 区间 → 4 区间；epochs 1000 → 2000（λ 边界/剪枝 epoch 按 epochs/1000
# 同比例缩放 → b120=240/b240=480/b600=1200）。其余超参与 exp001 完全一致。
# exp001 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp002_train.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
EXP="dn_fno_2608/kan/experiments/exp002_kan_v5"
CKPT="$EXP/model_b24g4_kan_mix/best.pt"
mkdir -p "$EXP"

echo "== exp002 训练 N=500 seed 1（hidden 24 / grid 4 / 2000 ep，λ 窗口 2x 缩放）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V5/dn/train.npz,$V5/sn/train.npz" \
    --val-data "$V5/dn/val.npz,$V5/sn/val.npz" \
    --n-train 500 --seed 1 --epochs 2000 --milestones 600,1000,1400,1800 \
    --hidden 24 --grid-size 4 \
    --out-dir "$EXP/model_b24g4_kan_mix" \
    2>&1 | tee "$EXP/train_exp002.log" | tail -8
fi

echo "== 六桶评估 =="
for B in all dn sn; do
  T="$V5/dn/test.npz"; [ "$B" = "dn" ] && T="$V5/dn/test.npz" || true
  [ "$B" = "sn" ] && T="$V5/sn/test.npz" || true
  [ "$B" = "all" ] && T="$V5/dn/test.npz,$V5/sn/test.npz" || true
  "$PY" -u -m gs_pino_kan_2608.evaluate_kan \
    --test-data "$T" --checkpoint "$CKPT" \
    --out-dir "$EXP/model_b24g4_kan_mix/eval_$B" 2>&1 | tail -4
done
echo "== done: 回填 $EXP/README.md 对比表 =="
