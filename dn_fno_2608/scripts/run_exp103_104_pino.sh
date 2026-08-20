#!/usr/bin/env bash
# exp103/104: FNO + GS 物理残差约束——混合 DN+SN（data_v5，coil 18ch 无 config）
#   exp103 = 做法1 单阶段：MSE(psi_plasma) + w_pde * ||Δ*ψ_pred + μ0RJ_data||^2
#   exp104 = 做法2 两阶段：阶段1 监督 psi_plasma+J；阶段2 自洽残差 + Ip 约束
# 与 exp011 同口径混合：逗号拼接 dn+sn（4000 池嵌套抽 500 = 255 DN + 245 SN），
# stats 全池计算，无 config 通道（模型从 11 线圈电流自推断位形）
# 日志/评估产物落各自实验目录
# 用法: bash dn_fno_2608/scripts/run_exp103_104_pino.sh [train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
TRAIN="$V5/dn/train.npz,$V5/sn/train.npz"
VAL="$V5/dn/val.npz,$V5/sn/val.npz"
TEST="$V5/dn/test.npz,$V5/sn/test.npz"
E103="dn_fno_2608/experiments/exp103_pino_rhs_mix_n500"
E104="dn_fno_2608/experiments/exp104_pino_twostage_mix_n500"
MODE="${1:-all}"

train_103() {
  echo "== exp103: 做法1 单阶段 RHS 残差，混合 DN+SN (N=500, 800 epochs) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 \
    --out-dir "$E103" 2>&1 | tee "$E103/train.log" | tail -6
}

train_104() {
  echo "== exp104: 做法2 两阶段，混合 DN+SN (N=500, 800 epochs, stage1<3% 切换, ramp 30) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E104" 2>&1 | tee "$E104/train.log" | tail -6
}

# 三桶评估：all(拼接1000) / dn(500) / sn(500)，每桶独立 figures
eval_bucket() { # $1=exp  $2=name  $3=test-data  $4=title
  local E="$1" NAME="$2"
  echo "== $E 评估桶 $NAME =="
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$3" --checkpoint "$E/best.pt" \
    --out-dir "$E/eval_$NAME" --fig-dir "$E/figures_$NAME" \
    --machine mast --title "$4" \
    2>&1 | tee "$E/eval_$NAME.log" | tail -6
}

eval_103() {
  eval_bucket "$E103" all  "$TEST" "exp103 FNO+GS rhs mixed (N=500, DN+SN test)"
  eval_bucket "$E103" dn   "$V5/dn/test.npz" "exp103 FNO+GS rhs mixed (N=500, DN test)"
  eval_bucket "$E103" sn   "$V5/sn/test.npz" "exp103 FNO+GS rhs mixed (N=500, SN test)"
}

eval_104() {
  eval_bucket "$E104" all  "$TEST" "exp104 FNO+GS twostage mixed (N=500, DN+SN test)"
  eval_bucket "$E104" dn   "$V5/dn/test.npz" "exp104 FNO+GS twostage mixed (N=500, DN test)"
  eval_bucket "$E104" sn   "$V5/sn/test.npz" "exp104 FNO+GS twostage mixed (N=500, SN test)"
}

case "$MODE" in
  train) train_103; train_104 ;;
  eval) eval_103; eval_104 ;;
  *) train_103; train_104; eval_103; eval_104 ;;
esac
echo "== done =="
