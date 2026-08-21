#!/usr/bin/env bash
# exp105/106: FNO + GS 物理残差约束——data_v6_clean 五配置混合（类似 exp013 口径）
#   exp105 = 做法1 单阶段：MSE(psi_plasma) + w_pde * ||Δ*ψ_pred + μ0RJ_data||^2
#   exp106 = 做法2 两阶段：阶段1 监督 psi_plasma+J；阶段2 自洽残差 + Ip 约束
# 数据：data_v6_clean dn/sn/snow_single/snow_double/limiter 逗号拼接
#   （train 池 5×500=2500 嵌套抽 500，seed 1，21ch = R,Z + 5 params + 14 线圈电流，
#   无 config 通道——同 exp013 口径）
# 物理约定与 exp103/104 相同：网络只预测 psi_plasma（+J），psi_total = psi_plasma
#   + Σ I_k·G_k（greens 加回）；J = R·p' + F·F'/(μ0R)（float64 重构，Ip 重构
#   mean 3-16e-4，greens 恒等式 1e-7 量级已验证）
# 日志/评估产物落各自实验目录
# 用法: bash dn_fno_2608/scripts/run_exp105_106_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
CLEAN="dn_fno_2608/data_v6_clean"
CFGS="dn sn snow_single snow_double limiter"
E105="dn_fno_2608/experiments/exp105_pino_rhs_v6clean_n500"
E106="dn_fno_2608/experiments/exp106_pino_twostage_v6clean_n500"
MODE="${1:-all}"

join_csv() { local out=""; for c in $CFGS; do out+="$CLEAN/$c/$1.npz,"; done; echo "${out%,}"; }
TRAIN=$(join_csv train)
VAL=$(join_csv val)
TEST=$(join_csv test)

train() { # $1=exp  $2=mode  $3=extra args
  local E="$1" M="$2"
  echo "== $E: $M, v6_clean 五配置混合 (N=500, 800 epochs) =="
  mkdir -p "$E"   # tee 需要目录先存在（train_pino 的 mkdir 发生在 python 启动后）
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode "$M" \
    --train-data "$TRAIN" --val-data "$VAL" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 ${3:-} \
    --out-dir "$E" 2>&1 | tee "$E/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟
  for pair in "rhs smoke_rhs" "twostage smoke_twostage"; do
    set -- $pair
    echo "== smoke $1 (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode "$1" \
      --train-data "$TRAIN" --val-data "$VAL" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 \
      --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
      --out-dir "dn_fno_2608/experiments/_smoke_v6_$2" 2>&1 | tail -4
  done
}

# 六桶评估：all(1000) + 5 配置各 200；limiter 桶几何指标按设计 NaN（无分离面）
eval_bucket() { # $1=exp  $2=name  $3=test-data  $4=title
  local E="$1" NAME="$2"
  echo "== $E 评估桶 $NAME =="
  mkdir -p "$E"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$3" --checkpoint "$E/best.pt" \
    --out-dir "$E/eval_$NAME" --fig-dir "$E/figures_$NAME" \
    --machine mastu_simple --title "$4" \
    2>&1 | tee "$E/eval_$NAME.log" | tail -6
}

eval_exp() { # $1=exp  $2=tag
  local E="$1" T="$2"
  eval_bucket "$E" all "$TEST" "$T v6_clean FNO+GS mixed (N=500, all 5 configs test)"
  for cfg in $CFGS; do
    eval_bucket "$E" "$cfg" "$CLEAN/$cfg/test.npz" \
      "$T v6_clean FNO+GS mixed (N=500, $cfg test)"
  done
}

case "$MODE" in
  smoke) smoke ;;
  train) train "$E105" rhs ""; train "$E106" twostage "--ip-weight 1.0 --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30" ;;
  eval) eval_exp "$E105" "exp105 rhs"; eval_exp "$E106" "exp106 twostage" ;;
  *) train "$E105" rhs ""; train "$E106" twostage "--ip-weight 1.0 --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30"; eval_exp "$E105" "exp105 rhs"; eval_exp "$E106" "exp106 twostage" ;;
esac
echo "== done =="
