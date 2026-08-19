#!/usr/bin/env bash
# KAN M2: 符号回归 KAN-SR（镜像论文表 3）+ verify（误差相关 + 延迟基准）
# 用法: bash dn_fno_2608/kan/scripts/run_kan_m2_symbolic.sh
#   KAN_CKPT=<best.pt 路径> 覆盖（默认 exp001_kan_v5/model_b19ch_kan_mix）
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
CKPT="${KAN_CKPT:-dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/best.pt}"
SR="$(dirname "$CKPT")/sr"
mkdir -p "$SR"

echo "== 符号回归 + 生成 symbolic_gs_kan.py =="
"$PY" -u -m gs_pino_kan_2608.symbolic_kan \
  --checkpoint "$CKPT" --test-data "$V5/sn/test.npz" \
  --out-dir "$SR" --max-samples 10 2>&1 | tail -25
echo "== done: sr_stats.json / expressions.json / symbolic_gs_kan.py / verify.json =="
