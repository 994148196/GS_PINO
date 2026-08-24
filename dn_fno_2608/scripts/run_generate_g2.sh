#!/usr/bin/env bash
# data_gspack2_v1 数据生成驱动（gspack2_TRAE MAST 11 线圈复刻，exp201/202 数据源）
#   首批（gen）：dn/sn × train 500@123 / val 500@456 / test 500@789
#   补数据（topup）：同 seed 更大 --n-samples 重跑 → 只生成新 chunk → merge
#     重建 {split}.npz（行 0..N0-1 比特级不变；训练子集随池大小变化，README §9）
#   采样口径与 data_v5 MAST 一致：--alpha-sampling（5 params → 18 输入通道）
#     --isoflux-sampling --anchor-midplane，X 点 (0.7,±1.1) jitter R 0.06/Z 0.10，
#     锚点 R~U[1.2,1.6]，接受约束全套（isoflux<=0.35 / xpt_dev<=0.10 /
#     anchor_xpt_dist>=0.15 / coil_margin 0.05 / core_depth>=0.005）
# 用法: bash dn_fno_2608/scripts/run_generate_g2.sh [probe|gen|topup|all]
#   topup: bash ... topup dn train 1500 123   （= 补到 1500 的训练池）
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
G2="dn_fno_2608/data_gspack2_v1"
M="mast_g2"
NJOBS=24
COMMON="--machine $M --alpha-sampling --isoflux-sampling --anchor-midplane \
  --xpt-jitter 0.06 --xpt-jitter-z 0.10 \
  --max-isoflux-residual 0.35 --max-xpt-deviation 0.10 \
  --min-anchor-xpt-dist 0.15 --coil-margin 0.05 --min-core-depth 0.005 \
  --save-constraint-diag --n-jobs $NJOBS --chunk-size 500 --max-retries 20"

# 注意：每配置独立 out-dir（$G2/dn、$G2/sn，同 data_v5 布局）——chunk 断点
# 续跑按 (config, split, chunk_idx) 定位；共用 out-dir 会让 sn 的 chunk_000
# 与 dn 的撞号（实测 sn 整块被跳过、merge 出 dn-only 数据）
run_split() { # $1=cfg  $2=split  $3=n  $4=seed
  echo "== generate $M $1 $2 n=$3 seed=$4 =="
  "$PY" -u -m gs_gspack2_dn_fno_2608.generate_g2_dataset \
    --config "$1" --split "$2" --n-samples "$3" --seed "$4" --out-dir "$G2/$1" $COMMON
  "$PY" -u -m gs_gspack2_dn_fno_2608.generate_g2_dataset \
    --merge --out-dir "$G2/$1" --split "$2"
}

probe() {
  "$PY" dn_fno_2608/scripts/probe_g2.py --machine $M --config dn \
    --n 80 --seed 123 --out "$G2/_probe_dn.json"
  "$PY" dn_fno_2608/scripts/probe_g2.py --machine $M --config sn \
    --n 80 --seed 123 --out "$G2/_probe_sn.json"
}

gen() {
  run_split dn train 500 123
  run_split sn train 500 123
  run_split dn val 500 456
  run_split sn val 500 456
  run_split dn test 500 789
  run_split sn test 500 789
}

case "$1" in
  probe) probe ;;
  gen) gen ;;
  topup) run_split "$2" "$3" "$4" "$5" ;;
  *) probe; gen ;;
esac
echo "== done =="
