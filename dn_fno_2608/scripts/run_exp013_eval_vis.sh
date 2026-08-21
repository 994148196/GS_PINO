#!/usr/bin/env bash
# exp013: 评估 + 可视化（镜像 run_exp012_eval_vis.sh，数据换 data_v6_clean）
# 6 桶：eval_all（5 文件 test pool，n=1000）+ 5 单配置各 200
# limiter 桶：X 点/分离面几何指标 NaN（无分离面），只报 rel L2/RMSE/GS 残差/O 点
# 用法: bash dn_fno_2608/scripts/run_exp013_eval_vis.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
CLEAN="dn_fno_2608/data_v6_clean"
EXP="dn_fno_2608/experiments/exp013_clean_v6"
CKPT="$EXP/best.pt"

ALL="$CLEAN/dn/test.npz,$CLEAN/sn/test.npz,$CLEAN/snow_single/test.npz,$CLEAN/snow_double/test.npz,$CLEAN/limiter/test.npz"

echo "== eval_all (n=1000 五配置混合) =="
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data "$ALL" --checkpoint "$CKPT" \
  --out-dir "$EXP/eval_all" 2>&1 | tail -4
for cfg in dn sn snow_single snow_double limiter; do
    echo "== eval_$cfg (n=200) =="
    "$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
      --test-data "$CLEAN/$cfg/test.npz" --checkpoint "$CKPT" \
      --out-dir "$EXP/eval_$cfg" 2>&1 | tail -4
done

echo "== figures (5 配置示例) =="
for cfg in dn sn snow_single snow_double limiter; do
    "$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
      --test-data "$CLEAN/$cfg/test.npz" --checkpoint "$CKPT" \
      --out-dir "$EXP/figures_$cfg" --machine mastu_simple \
      --title "exp013 clean v6 coil 21ch mixed (N=500) - $cfg test" 2>&1 | tail -3
done
echo "== done =="
