#!/usr/bin/env bash
# exp201/202: FNO + GS 物理残差约束——混合 DN+SN（data_gspack2_v1，coil 18ch 无 config）
#   exp201 = 做法1 单阶段：MSE(psi_plasma) + w_pde * ||Δ*ψ_pred + μ0RJ_data||^2
#   exp202 = 做法2 两阶段：阶段1 监督 psi_plasma+J；阶段2 自洽残差 + Ip 约束
# 与 exp103/104 同口径混合：逗号拼接 dn+sn（池 1000 嵌套抽 500 = 256 DN + 244 SN，
#   无 config 通道，18ch = R,Z + 5 params + 11 线圈电流），stats 全池计算；
# 管线（train/evaluate/data/model/loss）零改动——唯一数据源换为 gspack2_TRAE
# 生成的 data_gspack2_v1（MAST 11 线圈复刻，26 键 schema 与 data_v5 一致）
# 日志/评估产物落各自实验目录
# 用法: bash dn_fno_2608/scripts/run_exp201_202_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
G2="dn_fno_2608/data_gspack2_v1"
TRAIN="$G2/dn/train.npz,$G2/sn/train.npz"
VAL="$G2/dn/val.npz,$G2/sn/val.npz"
TEST="$G2/dn/test.npz,$G2/sn/test.npz"
E201="dn_fno_2608/experiments/exp201_pino_rhs_mix_gspack2_n500"
E202="dn_fno_2608/experiments/exp202_pino_twostage_mix_gspack2_n500"
MODE="${1:-all}"

train_201() {
  echo "== exp201: 做法1 单阶段 RHS 残差，混合 DN+SN (N=500, 800 epochs) =="
  mkdir -p "$E201"   # tee 需要目录先存在
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 \
    --out-dir "$E201" 2>&1 | tee "$E201/train.log" | tail -6
}

train_202() {
  echo "== exp202: 做法2 两阶段，混合 DN+SN (N=500, 800 epochs, stage1<3% 切换, ramp 30) =="
  mkdir -p "$E202"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E202" 2>&1 | tee "$E202/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟，双模式
  local S="dn_fno_2608/experiments/_smoke_g2"
  mkdir -p "$S"
  for pair in "rhs smoke_rhs" "twostage smoke_twostage"; do
    set -- $pair
    echo "== smoke $1 (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode "$1" \
      --train-data "$TRAIN" --val-data "$VAL" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 \
      --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
      --out-dir "$S/$2" 2>&1 | tail -4
  done
}

# 三桶评估：all(拼接1000) / dn(500) / sn(500)，每桶独立 figures
eval_bucket() { # $1=exp  $2=name  $3=test-data  $4=title
  local E="$1" NAME="$2"
  echo "== $E 评估桶 $NAME =="
  mkdir -p "$E"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$3" --checkpoint "$E/best.pt" \
    --out-dir "$E/eval_$NAME" --fig-dir "$E/figures_$NAME" \
    --machine mast --title "$4" \
    2>&1 | tee "$E/eval_$NAME.log" | tail -6
}

eval_201() {
  eval_bucket "$E201" all "$TEST" "exp201 FNO+GS rhs mixed (N=500, gspack2 DN+SN test)"
  eval_bucket "$E201" dn  "$G2/dn/test.npz" "exp201 FNO+GS rhs mixed (N=500, gspack2 DN test)"
  eval_bucket "$E201" sn  "$G2/sn/test.npz" "exp201 FNO+GS rhs mixed (N=500, gspack2 SN test)"
}

eval_202() {
  eval_bucket "$E202" all "$TEST" "exp202 FNO+GS twostage mixed (N=500, gspack2 DN+SN test)"
  eval_bucket "$E202" dn  "$G2/dn/test.npz" "exp202 FNO+GS twostage mixed (N=500, gspack2 DN test)"
  eval_bucket "$E202" sn  "$G2/sn/test.npz" "exp202 FNO+GS twostage mixed (N=500, gspack2 SN test)"
}

case "$MODE" in
  smoke) smoke ;;
  train) train_201; train_202 ;;
  eval) eval_201; eval_202 ;;
  *) train_201; train_202; eval_201; eval_202 ;;
esac
echo "== done =="
