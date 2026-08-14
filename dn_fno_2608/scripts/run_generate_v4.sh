#!/bin/bash
# data_v4 全量生成：可行区采样 + 物理合理性接受约束 + 约束诊断字段
#
# 与 data_v3 的差异（见 data_v4/README.md）：
#   X 点 R0 1.1 -> 1.2（远离线圈四边形左边缘）、抖动 R 0.10 / Z 0.15
#   锚点 -> 中平面 (R, 0.0)，R~U[1.35,1.65]（外中平面分离面半径）
#   接受约束：磁轴在三角形内 + 三点在四边形内(余量0.05) + 墙内 +
#     isoflux 残差<=0.35 x core + X 点偏差<=0.10 + 锚点-X点>=0.15 +
#     core 深度>=0.005
#   诊断字段：--save-constraint-diag 保存 8 个约束残差/实际临界点字段
#
# 阈值校准：data_v2 是已知健康数据，0.35/0.05 下通过 99.2%（见探针记录）
set -e
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
OUT="dn_fno_2608/data_v4"
FLAGS="--alpha-sampling --xpt-r0 1.2 --xpt-jitter 0.10 --xpt-jitter-z 0.15 \
--isoflux-sampling --anchor-midplane --max-isoflux-residual 0.35 \
--max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --require-wall \
--coil-margin 0.05 --min-core-depth 0.005 --save-constraint-diag \
--max-retries 20 --chunk-size 500 --n-jobs 24"

run_split() {
    local split=$1 n=$2 seed=$3
    "$PY" -u -m gs_pino_dn_fno_2608.generate_dn_dataset \
        --split "$split" --n-samples "$n" --seed "$seed" \
        --out-dir "$OUT" $FLAGS 2>&1 | grep -v "Warning"
    "$PY" -u -m gs_pino_dn_fno_2608.generate_dn_dataset \
        --split "$split" --out-dir "$OUT" --merge 2>&1 | grep -v "Warning"
}

run_split val  500 456
run_split test 500 789
run_split train 2000 123
echo "data_v4 done"
