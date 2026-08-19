#!/bin/bash
# data_v6 全量生成：MASTU_simple 5 配置（dn/sn/snow_single/snow_double/limiter）
#
# 规模（用户拍板：每配置 500 训练样本；val/test 配套小量）：
#   train 500 / val 100 / test 200 每配置，129^2 网格，14 线圈
#
# 探针校准（probe_v6.py，seed 123，80 样本/配置）：
#   dn 76.2%（X 点中心 R=0.80，壁内结构判据 INWALL_SEP_FRAC_TOL=0.02 已排除病态）
#   sn 55.0% / snow_single 65.0%（失败多为 No O-points / Picard 不收敛，物理失败）
#   snow_double 100% / limiter 100%
#   触壁全部浅触（max excess 19.1% < 20% 深触阈值），保留 + 每样本标注
#   （wall_contact / wall_contact_excess / inwall_sep_frac）
#   14 通道电流分离度全部 >2.0 std、配对 ≥90.75% -> exp012 输入 21ch（无 config 通道）
#
# 扩展方式（用户要求"以后直接加在后面"）：同 seed 重跑更大 --n-samples，
# 已有 chunk_*.npz 自动跳过（resume），新 chunk 追加，merge 重新合成 split.npz。
# 同 seed + 确定性采样 -> 前段样本完全一致，不重复。
#
# 用法（仓库根目录，需要 freegs_snow fork）:
#   bash dn_fno_2608/scripts/run_generate_v6.sh
set -e
# freegs_snow fork（雪点二阶约束）+ 本仓库 src（-m 模块模式需要）
export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow;D:/D_F/Fusion/AI/PINN/GitHub-tests/GS_PINO/src"
PY="python"
BASE="dn_fno_2608/data_v6"
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

for cfg in dn sn snow_single snow_double limiter; do
    run_split "$cfg" val  100 789
    run_split "$cfg" test 200 1011
    run_split "$cfg" train 500 456
done
echo "data_v6 done"
