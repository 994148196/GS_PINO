#!/usr/bin/env bash
# exp307/308: PI-DeepONet 调优（exp304 0.802%/J 2.61% 基础上的容量方向）
#   exp307 = deeponet_wide：branch/trunk 宽度 256→384（0.60M→1.34M 参数，2.2×）
#           —— 直接打 J/Ip 弱 2× 的容量短板；其余训练参数与 exp102 逐项一致
#   exp308 = deeponet_ff：exp307 加宽 + trunk 输入 NeRF 式多频傅里叶特征
#           （ff_l=6，最高频 32π 在网格 Nyquist 之下）；307→308 增量归因 FF 贡献
# 数据/两阶段方案同 exp102：18ch = R,Z + 5 params + 11 线圈电流；
#   阶段1 监督 psi_plasma+J（val rel L2 < 3% 或 e300 切阶段2），阶段2 自洽
#   GS 残差 + Ip 约束（物理权重 ramp 预热，防阶段2 首 epoch 爆炸）
# 日志/评估产物落各自实验目录；冒烟 30-epoch 落 _smoke_exp305_308/
# 用法: bash dn_fno_2608/scripts/run_exp307_308_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E307="dn_fno_2608/experiments/exp307_pideeponet_wide_pino_twostage_n500"
E308="dn_fno_2608/experiments/exp308_pideeponet_ff_pino_twostage_n500"
MODE="${1:-all}"

train_307() {
  echo "== exp307: deeponet_wide 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E307"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model deeponet_wide \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E307" 2>&1 | tee "$E307/train.log" | tail -6
}

train_308() {
  echo "== exp308: deeponet_ff 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E308"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model deeponet_ff \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E308" 2>&1 | tee "$E308/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟
  local S="dn_fno_2608/experiments/_smoke_exp305_308"
  mkdir -p "$S"
  for m in deeponet_wide deeponet_ff; do
    echo "== smoke $m (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model "$m" \
      --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
      --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
      --stage2-ramp-epochs 30 \
      --out-dir "$S/$m" 2>&1 | tail -4
  done
}

eval_307() {
  echo "== exp307 评估（DN test 500） =="
  mkdir -p "$E307"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E307/best.pt" \
    --out-dir "$E307" --fig-dir "$E307/figures" \
    --machine mast --title "exp307 DeepONet-wide+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E307/eval.log" | tail -6
}

eval_308() {
  echo "== exp308 评估（DN test 500） =="
  mkdir -p "$E308"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E308/best.pt" \
    --out-dir "$E308" --fig-dir "$E308/figures" \
    --machine mast --title "exp308 DeepONet-FF+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E308/eval.log" | tail -6
}

case "$MODE" in
  smoke) smoke ;;
  train) train_307; train_308 ;;
  eval) eval_307; eval_308 ;;
  *) smoke; train_307; train_308; eval_307; eval_308 ;;
esac
echo "== done =="
