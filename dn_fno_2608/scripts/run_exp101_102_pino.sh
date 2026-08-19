#!/usr/bin/env bash
# exp101/102: FNO + GS 物理残差约束（data_v5/dn 单一位形，MAST DN）
#   exp101 = 做法1 单阶段：MSE(psi_plasma) + w_pde * ||Δ*ψ_pred + μ0RJ_data||^2
#   exp102 = 做法2 两阶段：阶段1 监督 psi_plasma+J；阶段2 自洽残差 + Ip 约束
# 日志落在各自实验目录（train.log / eval.log）
# 用法: bash dn_fno_2608/scripts/run_exp101_102_pino.sh [train|eval|verify|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E101="dn_fno_2608/experiments/exp101_pino_rhs_n500"
E102="dn_fno_2608/experiments/exp102_pino_twostage_n500"
MODE="${1:-all}"

run_verify() {
  echo "== verify_pde: 真值场残差基线/greens/Ip 恒等式 =="
  "$PY" -u -m gs_pino_fno_phys.verify_pde 2>&1 | tail -8
}

train_101() {
  echo "== exp101: 做法1 单阶段 RHS 残差 (N=500, 800 epochs) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 \
    --out-dir "$E101" 2>&1 | tee "$E101/train.log" | tail -6
}

train_102() {
  echo "== exp102: 做法2 两阶段 (N=500, 800 epochs, stage1<3% 切换, ramp 30) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E102" 2>&1 | tee "$E102/train.log" | tail -6
}

eval_101() {
  echo "== exp101 评估 + 可视化（exp011 风格 fig1/2/3） =="
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E101/best.pt" \
    --out-dir "$E101" --fig-dir "$E101/figures" --machine mast \
    --title "exp101 FNO+GS rhs (N=500, DN test)" \
    2>&1 | tee "$E101/eval.log" | tail -20
}

eval_102() {
  echo "== exp102 评估 + 可视化（exp011 风格 fig1/2/3） =="
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E102/best.pt" \
    --out-dir "$E102" --fig-dir "$E102/figures" --machine mast \
    --title "exp102 FNO+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E102/eval.log" | tail -20
}

case "$MODE" in
  verify) run_verify ;;
  train) train_101; train_102 ;;
  eval) eval_101; eval_102 ;;
  *) run_verify; train_101; train_102; eval_101; eval_102 ;;
esac
echo "== done =="
