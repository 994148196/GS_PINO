#!/usr/bin/env bash
# KAN M3: OOD 数据集生成（复用 freegs generator，registry 覆盖采样范围）
#         + 线性外推 KAN-EXT（论文式 4：最近包络点 + 一阶泰勒）
#         + DN/SN 位型分类验证（分离面 X 点计数，DN 正类）
# 用法: bash dn_fno_2608/kan/scripts/run_kan_m3_ood_gen_extrap.sh
#   KAN_CKPT=<best.pt 路径> 覆盖（默认 exp001_kan_v5/model_b19ch_kan_mix）
#   KAN_SKIP_OOD_GEN=1 跳过步骤 1（OOD 数据集共享，exp002 起可跳）
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
OOD="dn_fno_2608/kan/data_ood"
CKPT="${KAN_CKPT:-dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/best.pt}"
EXT="$(dirname "$CKPT")/ext"
mkdir -p "$EXP"

if [ "${KAN_SKIP_OOD_GEN:-0}" != "1" ]; then
echo "== 1. OOD 生成（probe 接受率 -> 分块生成(可续) -> box 包络过滤）=="
for cfg in dn sn; do
  "$PY" -u -m gs_pino_kan_2608.gen_ood_kan \
    --config $cfg --train-data "$V5/dn/train.npz,$V5/sn/train.npz" \
    --out-dir "$OOD" --hull-mode box 2>&1 | grep -E "probe:|generated:|hull:|in_hull|saved|WARNING" | head -8
done
fi

echo "== 2. 外推 + 分类（式 4 + 分离面 X 点计数；tol 由 in-dist 校准）=="
"$PY" -u -m gs_pino_kan_2608.extrapolate_kan \
  --checkpoint "$CKPT" \
  --train-data "$V5/dn/train.npz,$V5/sn/train.npz" \
  --in-dist-data "$V5/dn/test.npz,$V5/sn/test.npz" \
  --ood-dir "$OOD" --hull-mode box --out-dir "$EXT" 2>&1 | tail -18
echo "== done: $EXT/classify_metrics.json + samples.json + fig_classify.png =="
