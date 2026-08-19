#!/usr/bin/env bash
# KAN exp009: TestTokamak（data_v4）简单数据——验证数据复杂度假说
# 动机（用户反馈, 2026-08-19: "如果用 data 这种简单的数据，会不会结果好一些？"）:
#   MAST 11 线圈 |psi_coils|/|psi_total| = 183-185%（线圈场淹没等离子体场），
#   TestTokamak 仅 4 线圈、占比 62%——线圈场不占主导，B 样条 grid-2 的空间
#   分辨率问题（1.1m/区间 vs 3.4cm 网格）影响大幅减小，更接近论文 EXL-50U
#   （5 独立对称线圈、简单形状）。
# 设置: 与 exp007 完全一致（论文架构 hidden10 grid2 / 1000 ep / 256 点/样本 /
#   j=0 pde=0 reg=0 / N=500 seed 1），唯一差异 = 数据源 data_v4 TestTokamak
#   （4 线圈、无 config 通道，11ch）+ --device-config testtokamak。
# 对照: exp007/exp004（MAST data_v5, 19ch）→ 两者差 = 数据复杂度的净增益。
# exp001-008 产物零触碰。
# 用法: bash dn_fno_2608/kan/scripts/run_kan_exp009_testtokamak.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V4="dn_fno_2608/data_v4"
EXP="dn_fno_2608/kan/experiments/exp009_testtokamak_kan"
CKPT="$EXP/model_psionly_tt_kan/best.pt"
mkdir -p "$EXP"

echo "== exp009 TestTokamak 只预测 psi N=500 seed 1（论文架构 grid2, 1000 ep, j=0 pde=0 reg=0, 11ch）=="
if [ -f "$CKPT" ]; then
  echo "  ckpt 已存在，跳过训练（$CKPT）"
else
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data "$V4/train.npz" \
    --val-data "$V4/val.npz" \
    --n-train 500 --seed 1 --epochs 1000 --milestones 300,500,700,900 \
    --hidden 10 --grid-size 2 --pde-weight 0 --reg-weight 0 --j-weight 0 \
    --device-config testtokamak \
    --out-dir "$EXP/model_psionly_tt_kan" \
    2>&1 | tee "$EXP/train_exp009.log" | tail -8
fi

echo "== 评估（data_v4 test 494 行）=="
"$PY" -u -m gs_pino_kan_2608.evaluate_kan \
  --test-data "$V4/test.npz" --checkpoint "$CKPT" \
  --out-dir "$EXP/model_psionly_tt_kan/eval_test" 2>&1 | tail -4
echo "== done: 回填 $EXP/README.md =="
