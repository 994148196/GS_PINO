#!/bin/bash
# data_v5 全量生成：MAST DN/SN 分开落盘（探针校准，原始接受率 100%）
#
# 与 data_v4 的差异（见 data_v5/README.md）：
#   机器 MAST（无墙 -> --require-wall 忽略）、X 点 (0.7,+-1.1)±0.06/0.10、
#   锚点中平面 R∈[1.2,1.6]、参数 paxis(1e3,5e3)/Ip(3e5,8e5)/fvac(0.3,0.8)
#   SN 判据：分离面 X 点（psi >= psi_bndry - 1e-6）恰 1 个（替换 v4 的 xpt>=2）
#   SN 磁轴：X 点与锚点之间的最高 psi O 点（过滤线圈场真空假极值）
#   样本新增 config 字段（0=DN, 1=SN）；DN/SN 分开落盘 data_v5/{dn,sn}/
#
# 探针校准（probe_v5.py，seed 123，80 样本/配置）：
#   MAST+DN: 80/80 (100%)，isoflux_res 2.0e-4，xpt_dev 0.016，core 0.077
#   MAST+SN: 80/80 (100%)，isoflux_res 2.0e-5，xpt_dev 0.0015，core 0.085，
#            n_iter 40.6±（DN 8.7）-> SN 求解 ~1s/solve，3000 样本 ~2 分钟/24 核
set -e
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
BASE="dn_fno_2608/data_v5"
FLAGS="--machine mast --alpha-sampling --xpt-jitter 0.06 --xpt-jitter-z 0.10 \
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

for cfg in dn sn; do
    run_split "$cfg" val  500 456
    run_split "$cfg" test 500 789
    run_split "$cfg" train 2000 123
done
echo "data_v5 done"
