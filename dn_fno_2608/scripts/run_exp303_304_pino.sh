#!/usr/bin/env bash
# exp303/304: 非 FNO 骨架 + GS 两阶段物理约束（data_v5/dn，与 exp102 完全同口径）
#   exp303 = FNO-KAN 混合（FNOBlock 的 1x1 conv 换成 per-pixel KANConv1x1，
#            tanh 门控入样条域，grid 4/degree 2，≈4.33M 参数）
#   exp304 = PI-DeepONet（分支 2×MLP(16 标量) + 共享主干 MLP(R,Z) 点积输出，≈0.60M）
# 数据/两阶段方案/训练参数与 exp102 逐项一致：18ch = R,Z + 5 params + 11 线圈电流，
#   阶段1 监督 psi_plasma+J（val rel L2 < 3% 或 e300 切阶段2），阶段2 自洽 GS 残差
#   + Ip 约束（w_pde=0.1, w_ip=1.0, w_j=1.0, ramp 30）；唯一差异 = --model 骨架
# 日志/评估产物落各自实验目录；冒烟 30-epoch 落 _smoke_exp3xx/
# 用法: bash dn_fno_2608/scripts/run_exp303_304_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E303="dn_fno_2608/experiments/exp303_fnokan_pino_twostage_n500"
E304="dn_fno_2608/experiments/exp304_pideeponet_pino_twostage_n500"
MODE="${1:-all}"

run_verify() {
  echo "== verify_pde: 真值场残差基线/greens/Ip 恒等式（模型无关，一次即可） =="
  "$PY" -u -m gs_pino_fno_phys.verify_pde 2>&1 | tail -8
}

train_303() {
  echo "== exp303: FNO-KAN 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E303"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model fnokan \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E303" 2>&1 | tee "$E303/train.log" | tail -6
}

train_304() {
  echo "== exp304: PI-DeepONet 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E304"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model deeponet \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E304" 2>&1 | tee "$E304/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟，两骨架
  local S="dn_fno_2608/experiments/_smoke_exp3xx"
  mkdir -p "$S"
  for m in fnokan deeponet; do
    echo "== smoke $m (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model "$m" \
      --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
      --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
      --stage2-ramp-epochs 30 \
      --out-dir "$S/$m" 2>&1 | tail -4
  done
}

eval_303() {
  echo "== exp303 评估（DN test 500） =="
  mkdir -p "$E303"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E303/best.pt" \
    --out-dir "$E303" --fig-dir "$E303/figures" \
    --machine mast --title "exp303 FNO-KAN+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E303/eval.log" | tail -6
}

eval_304() {
  echo "== exp304 评估（DN test 500） =="
  mkdir -p "$E304"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E304/best.pt" \
    --out-dir "$E304" --fig-dir "$E304/figures" \
    --machine mast --title "exp304 PI-DeepONet+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E304/eval.log" | tail -6
}

case "$MODE" in
  verify) run_verify ;;
  smoke) smoke ;;
  train) train_303; train_304 ;;
  eval) eval_303; eval_304 ;;
  *) run_verify; smoke; train_303; train_304; eval_303; eval_304 ;;
esac
echo "== done =="
