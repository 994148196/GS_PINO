#!/usr/bin/env bash
# data_gspack2_v2 数据生成驱动（gspack2_TRAE MASTU_simple 五配置，exp203 数据源）
#   首批（gen）：dn/sn/snow_single/snow_double/limiter × train 500@123 /
#     val 100@456 / test 200@789（val/test 对齐 data_v6_clean → exp203 与
#     exp105 逐桶 1:1 对比）
#   采样口径 = data_v6（MASTU_simple 129²，21ch：R,Z + 5 params + 14 单元电流）
#     --alpha-sampling --isoflux-sampling --anchor-midplane
#   SN 质量门（生成侧硬门，探针校准阈值）：
#     --sn-midplane-ratio-min 0.05 --sn-zaxis-ratio-max 0.5 --max-gs-true 15
#   snow_double 雪点偏差门：--max-snow-xpt-dev 0.15
#   snowflake 求解：零电流初始 + 松弛 Picard（blend=0.3，生成器内建）
# 用法: bash dn_fno_2608/scripts/run_generate_g3.sh [probe|gen|topup|all]
#   topup: bash ... topup dn train 1500 123   （= 补到 1500 的训练池）
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
G3="dn_fno_2608/data_gspack2_v2"
NJOBS=16
COMMON="--alpha-sampling --isoflux-sampling --anchor-midplane \
  --xpt-jitter 0.06 --xpt-jitter-z 0.10 \
  --max-isoflux-residual 0.35 --max-xpt-deviation 0.10 \
  --min-anchor-xpt-dist 0.15 --coil-margin 0.05 --min-core-depth 0.005 \
  --save-constraint-diag --n-jobs $NJOBS --chunk-size 500 --max-retries 20 \
  --sn-midplane-ratio-min 0.05 --sn-zaxis-ratio-max 0.5 --max-gs-true 15 \
  --max-snow-xpt-dev 0.15"

# 注意：每配置独立 out-dir（$G3/dn、$G3/sn、…，同 g2/data_v5 布局）——chunk
# 断点续跑按 (config, split, chunk_idx) 定位；共用 out-dir 会撞号被整块跳过
run_split() { # $1=cfg  $2=split  $3=n  $4=seed
  echo "== generate $1 $2 n=$3 seed=$4 =="
  "$PY" -u -m gs_gspack2_dn_fno_2608.generate_dn_g3_dataset \
    --config "$1" --split "$2" --n-samples "$3" --seed "$4" --out-dir "$G3/$1" $COMMON \
    > "$G3/log_${1}_${2}.txt" 2>&1
  "$PY" -u -m gs_gspack2_dn_fno_2608.generate_dn_g3_dataset \
    --merge --out-dir "$G3/$1" --split "$2" \
    >> "$G3/log_${1}_${2}.txt" 2>&1
  tail -4 "$G3/log_${1}_${2}.txt"
}

probe() {
  "$PY" dn_fno_2608/scripts/probe_g3.py --config dn --n 80 --order-check
  "$PY" dn_fno_2608/scripts/probe_g3.py --config sn --n 80
  "$PY" dn_fno_2608/scripts/probe_g3.py --config snow_single --n 80
  "$PY" dn_fno_2608/scripts/probe_g3.py --config snow_double --n 80
  "$PY" dn_fno_2608/scripts/probe_g3.py --config limiter --n 80
}

gen() {
  for cfg in dn sn snow_single snow_double limiter; do
    run_split "$cfg" train 500 123
    run_split "$cfg" val 100 456
    run_split "$cfg" test 200 789
  done
}

case "$1" in
  probe) probe ;;
  gen) gen ;;
  topup) run_split "$2" "$3" "$4" "$5" ;;
  *) probe; gen ;;
esac
echo "== done =="
