#!/usr/bin/env bash
# exp301/302: 非 FNO 骨架 + GS 两阶段物理约束（data_v5/dn，与 exp102 完全同口径）
#   exp301 = U-Net CNN（plain conv，无归一化，base 24/depth 4，≈2.76M 参数）
#   exp302 = UFNO 傅里叶 U-Net（编码器 FNOBlock 多尺度谱卷积 + 普通 conv 解码器，≈2.47M）
# 数据/两阶段方案/训练参数与 exp102 逐项一致：18ch = R,Z + 5 params + 11 线圈电流，
#   阶段1 监督 psi_plasma+J（val rel L2 < 3% 或 e300 切阶段2），阶段2 自洽 GS 残差
#   + Ip 约束（w_pde=0.1, w_ip=1.0, w_j=1.0, ramp 30）；唯一差异 = --model 骨架
# 日志/评估产物落各自实验目录；冒烟 30-epoch 落 _smoke_exp3xx/
# 用法: bash dn_fno_2608/scripts/run_exp301_302_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E301="dn_fno_2608/experiments/exp301_unet_pino_twostage_n500"
E302="dn_fno_2608/experiments/exp302_ufno_pino_twostage_n500"
MODE="${1:-all}"

run_verify() {
  echo "== verify_pde: 真值场残差基线/greens/Ip 恒等式（模型无关，一次即可） =="
  "$PY" -u -m gs_pino_fno_phys.verify_pde 2>&1 | tail -8
}

train_301() {
  echo "== exp301: U-Net 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E301"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model unet \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E301" 2>&1 | tee "$E301/train.log" | tail -6
}

train_302() {
  echo "== exp302: UFNO 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E302"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model ufno \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E302" 2>&1 | tee "$E302/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟，两骨架
  local S="dn_fno_2608/experiments/_smoke_exp3xx"
  mkdir -p "$S"
  for m in unet ufno; do
    echo "== smoke $m (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model "$m" \
      --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
      --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
      --stage2-ramp-epochs 30 \
      --out-dir "$S/$m" 2>&1 | tail -4
  done
}

eval_301() {
  echo "== exp301 评估（DN test 500） =="
  mkdir -p "$E301"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E301/best.pt" \
    --out-dir "$E301" --fig-dir "$E301/figures" \
    --machine mast --title "exp301 UNet+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E301/eval.log" | tail -6
}

eval_302() {
  echo "== exp302 评估（DN test 500） =="
  mkdir -p "$E302"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E302/best.pt" \
    --out-dir "$E302" --fig-dir "$E302/figures" \
    --machine mast --title "exp302 UFNO+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E302/eval.log" | tail -6
}

case "$MODE" in
  verify) run_verify ;;
  smoke) smoke ;;
  train) train_301; train_302 ;;
  eval) eval_301; eval_302 ;;
  *) run_verify; smoke; train_301; train_302; eval_301; eval_302 ;;
esac
echo "== done =="
