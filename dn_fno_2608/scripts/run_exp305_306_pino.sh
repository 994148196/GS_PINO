#!/usr/bin/env bash
# exp305/306: UFNO 调优（exp302 0.729% 基础上的两个正交方向）
#   exp305 = ufno_tuned 架构侧：解码器 conv→FNOBlock 谱化（dec modes 4/8/8/8）
#           + bottleneck 1→2 层 + 65² 层模态 16→20（≈3.60M < FNO 4.21M）
#           —— 其余训练参数与 exp102 逐项一致
#   exp306 = 原 ufno 架构 + 训练侧杠杆（零代码）：w_j 1.0→2.0、w_pde 0.1→0.3、
#           stage1 阈值 3%→4%（e70 切换偏晚，提前让物理项介入）、ramp 30→60
# 数据/两阶段方案同 exp102：18ch = R,Z + 5 params + 11 线圈电流；
#   阶段1 监督 psi_plasma+J（val rel L2 < 阈值 或 e300 切阶段2），阶段2 自洽
#   GS 残差 + Ip 约束（物理权重 ramp 预热，防阶段2 首 epoch 爆炸）
# 日志/评估产物落各自实验目录；冒烟 30-epoch 落 _smoke_exp305_308/
# 用法: bash dn_fno_2608/scripts/run_exp305_306_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E305="dn_fno_2608/experiments/exp305_ufno_tuned_pino_twostage_n500"
E306="dn_fno_2608/experiments/exp306_ufno_weights_pino_twostage_n500"
MODE="${1:-all}"

train_305() {
  echo "== exp305: ufno_tuned 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E305"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model ufno_tuned \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E305" 2>&1 | tee "$E305/train.log" | tail -6
}

train_306() {
  echo "== exp306: ufno 原架构+训练侧权重 (N=500, 800 epochs) =="
  mkdir -p "$E306"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model ufno \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.3 --ip-weight 1.0 \
    --j-weight 2.0 --stage1-threshold 0.04 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 60 \
    --out-dir "$E306" 2>&1 | tee "$E306/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟
  local S="dn_fno_2608/experiments/_smoke_exp305_308"
  mkdir -p "$S"
  for m in ufno_tuned; do
    echo "== smoke $m (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model "$m" \
      --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
      --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
      --stage2-ramp-epochs 30 \
      --out-dir "$S/$m" 2>&1 | tail -4
    # exp306 无新代码，只换 CLI 参数——冒烟只需验证参数被接受
  done
  echo "== smoke ufno (exp306 参数 30 epochs) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model ufno \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 30 --phys-weight 0.3 --ip-weight 1.0 \
    --j-weight 2.0 --stage1-threshold 0.04 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 60 \
    --out-dir "$S/ufno_wts" 2>&1 | tail -4
}

eval_305() {
  echo "== exp305 评估（DN test 500） =="
  mkdir -p "$E305"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E305/best.pt" \
    --out-dir "$E305" --fig-dir "$E305/figures" \
    --machine mast --title "exp305 UFNO-tuned+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E305/eval.log" | tail -6
}

eval_306() {
  echo "== exp306 评估（DN test 500） =="
  mkdir -p "$E306"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E306/best.pt" \
    --out-dir "$E306" --fig-dir "$E306/figures" \
    --machine mast --title "exp306 UFNO+wts twostage (N=500, DN test)" \
    2>&1 | tee "$E306/eval.log" | tail -6
}

case "$MODE" in
  smoke) smoke ;;
  train) train_305; train_306 ;;
  eval) eval_305; eval_306 ;;
  *) smoke; train_305; train_306; eval_305; eval_306 ;;
esac
echo "== done =="
