#!/bin/bash
# data_v6 补足候选池：data_v6_clean 筛选剔除后按新 seed 独立生成健康候选。
#
# 为什么不用"同 seed 更大 --n-samples"扩展（run_generate_v6.sh 注释里的方式）：
# chunk resume 按文件跳过且不推进 rng，新 chunk 的参数序列会与旧段前段重复；
# 因此补足用全新 seed 独立候选池，与原样本参数不重叠（连续分布，重复概率≈0）。
#
# 候选池规模 = 需求健康数 / (1 - 新判据拒率) + margin:
#   sn 拒率 13.5-18.4% -> 需求 137 -> 候选 130/30/45 (train/val/test)
#   dn 拒率 ~2%       -> 需求 13  -> 候选 20/5/10
#   snow_double 拒率 2%-> 需求 11  -> 候选 20/5/10
# （候选池经生成器 max_retries=20 补齐后全部通过原 7 项接受判据）
#
# 用法（仓库根目录，需要 freegs_snow fork）:
#   bash dn_fno_2608/scripts/run_generate_v6_topup.sh
set -e
export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow;D:/D_F/Fusion/AI/PINN/GitHub-tests/GS_PINO/src"
PY="python"
BASE="dn_fno_2608/data_v6_topup"
FLAGS="--machine mastu_simple --alpha-sampling --xpt-jitter 0.06 --xpt-jitter-z 0.10 \
--isoflux-sampling --anchor-midplane --max-isoflux-residual 0.35 \
--max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --coil-margin 0.05 \
--min-core-depth 0.005 --save-constraint-diag --max-retries 20 \
--chunk-size 500 --n-jobs 24"

run_split() {
    local cfg=$1 split=$2 n=$3 seed=$4
    "$PY" -u -m gs_pino_dn_fno_2608.generate_dn_dataset \
        --split "$split" --n-samples "$n" --seed "$seed" \
        --out-dir "$BASE/$cfg" --config "$cfg" $FLAGS 2>&1 | grep -v "Warning"
    "$PY" -u -m gs_pino_dn_fno_2608.generate_dn_dataset \
        --split "$split" --out-dir "$BASE/$cfg" --merge 2>&1 | grep -v "Warning"
}

# sn (train 92 + val 18 + test 27 被筛)
run_split sn train 130 20260821
run_split sn val 30 20260822
run_split sn test 45 20260823
# dn (train 9 + test 4 被筛)
run_split dn train 20 20260831
run_split dn val 5 20260832
run_split dn test 10 20260833
# snow_double (train 7 + test 4 被筛)
run_split snow_double train 20 20260911
run_split snow_double val 5 20260912
run_split snow_double test 10 20260913
echo "topup done"
