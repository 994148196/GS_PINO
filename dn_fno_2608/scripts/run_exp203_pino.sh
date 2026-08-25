#!/usr/bin/env bash
# exp203: FNO + GS 物理残差（做法1 rhs，单阶段）——data_gspack2_v2 五配置混合
#   = exp105 的数据源替换 + 数据质量改进版本（对照 exp105：v6_clean 2.362%）
#   gspack2_TRAE 生成 MASTU_simple（129²、21ch、val/test 对齐 v6_clean →
#   逐桶 1:1 对比）；SN 生成侧硬门（midplane≥0.05、zaxis≤0.5、gs_true≤15）
#   目标：SN 桶显著优于 exp105 的 7.738%
# 数据：data_gspack2_v2 dn/sn/snow_single/snow_double/limiter 逗号拼接
#   （train 池 5×500=2500 嵌套抽 500，seed 1；val/test 各 5×100/5×200）
# 物理约定与 exp105 相同：网络只预测 psi_plasma（+J），psi_total = psi_plasma
#   + Σ I_k·G_k（greens 加回）；J = R·p' + F·F'/(μ0R)
# 用法: bash dn_fno_2608/scripts/run_exp203_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
G3="dn_fno_2608/data_gspack2_v2"
CFGS="dn sn snow_single snow_double limiter"
E203="dn_fno_2608/experiments/exp203_pino_rhs_g3_n500"
MODE="${1:-all}"

join_csv() { local out=""; for c in $CFGS; do out+="$G3/$c/$1.npz,"; done; echo "${out%,}"; }
TRAIN=$(join_csv train)
VAL=$(join_csv val)
TEST=$(join_csv test)

train() {
  echo "== exp203: rhs, data_gspack2_v2 五配置混合 (N=500, 800 epochs) =="
  mkdir -p "$E203"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 \
    --out-dir "$E203" 2>&1 | tee "$E203/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟（数据加载/训练管线自检）
  echo "== exp203 smoke (30 epochs) =="
  mkdir -p "dn_fno_2608/experiments/_smoke_g3_rhs"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 \
    --out-dir "dn_fno_2608/experiments/_smoke_g3_rhs" 2>&1 | tail -4
}

# 六桶评估：all(1000) + 5 配置各 200；limiter 桶几何指标按设计 NaN（无分离面）
eval_bucket() { # $1=name  $2=test-data  $3=title
  local NAME="$1"
  echo "== exp203 评估桶 $NAME =="
  mkdir -p "$E203"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$2" --checkpoint "$E203/best.pt" \
    --out-dir "$E203/eval_$NAME" --fig-dir "$E203/figures_$NAME" \
    --machine mastu_simple --title "$3" \
    2>&1 | tee "$E203/eval_$NAME.log" | tail -6
}

eval_exp() {
  eval_bucket all "$TEST" "exp203 rhs g3 (N=500, all 5 configs test)"
  for cfg in $CFGS; do
    eval_bucket "$cfg" "$G3/$cfg/test.npz" \
      "exp203 rhs g3 (N=500, $cfg test)"
  done
}

case "$MODE" in
  smoke) smoke ;;
  train) train ;;
  eval) eval_exp ;;
  *) smoke; train; eval_exp ;;
esac
echo "== done =="
