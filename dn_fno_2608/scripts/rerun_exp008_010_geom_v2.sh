#!/bin/bash
# 2026-08-17: 几何指标 v2 重跑（用户绘图问题修复后）
#   - geometry_metrics: 真值基准 X 点配对（排除 MAST 真空假鞍点）+ 射线法分离面
#   - fig1: 装置结构/线圈/分离面X点/磁轴/separatrix + R,Z 等比例
# 旧 metrics.json / stats_per_sample.json 备份为 *_v1.json（不覆盖历史）。
# 用法: bash dn_fno_2608/scripts/rerun_exp008_010_geom_v2.sh
set -e
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
EX=dn_fno_2608/experiments

echo "== 1. 备份旧产物 (metrics_v1 / stats_per_sample_v1) =="
for m in exp008_mixed_configs/model_a14ch_xa_mix \
         exp009_split_configs/model_dn_13ch \
         exp009_split_configs/model_sn_13ch \
         exp010_mixed_no_config/model_a13ch_xa_mix; do
  for d in "$EX/$m"/eval* "$EX/$m"/figures*; do
    [ -f "$d/metrics.json" ] && cp -n "$d/metrics.json" "$d/metrics_v1.json"
    [ -f "$d/stats_per_sample.json" ] && cp -n "$d/stats_per_sample.json" "$d/stats_per_sample_v1.json"
  done
done

echo "== 2. evaluate 全量 =="
ev() { echo "  eval -> $3"; "$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
       --test-data "$1" --checkpoint "$2" --out-dir "$3" || echo "  FAIL $3"; }
ev dn_fno_2608/data_v5/dn/test.npz "$EX/exp008_mixed_configs/model_a14ch_xa_mix/best.pt" "$EX/exp008_mixed_configs/model_a14ch_xa_mix/eval_dn"
ev dn_fno_2608/data_v5/sn/test.npz "$EX/exp008_mixed_configs/model_a14ch_xa_mix/best.pt" "$EX/exp008_mixed_configs/model_a14ch_xa_mix/eval_sn"
ev dn_fno_2608/data_v5/dn/test.npz "$EX/exp009_split_configs/model_dn_13ch/best.pt"       "$EX/exp009_split_configs/model_dn_13ch/eval"
ev dn_fno_2608/data_v5/sn/test.npz "$EX/exp009_split_configs/model_dn_13ch/best.pt"       "$EX/exp009_split_configs/model_dn_13ch/eval_cross_sn"
ev dn_fno_2608/data_v5/sn/test.npz "$EX/exp009_split_configs/model_sn_13ch/best.pt"       "$EX/exp009_split_configs/model_sn_13ch/eval"
ev dn_fno_2608/data_v5/dn/test.npz "$EX/exp009_split_configs/model_sn_13ch/best.pt"       "$EX/exp009_split_configs/model_sn_13ch/eval_cross_dn"
ev dn_fno_2608/data_v5/dn/test.npz "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/best.pt" "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/eval_dn"
ev dn_fno_2608/data_v5/sn/test.npz "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/best.pt" "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/eval_sn"

echo "== 3. visualize (--machine mast, figures 重算) =="
viz() { echo "  figs -> $3"; "$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
       --test-data "$1" --checkpoint "$2" --out-dir "$3" --machine mast --title "$4" \
       || echo "  FAIL $3"; }
viz dn_fno_2608/data_v5/dn/test.npz "$EX/exp008_mixed_configs/model_a14ch_xa_mix/best.pt" "$EX/exp008_mixed_configs/model_a14ch_xa_mix/figures_dn" "exp008 mixed 14ch (xa+config) - MAST DN"
viz dn_fno_2608/data_v5/sn/test.npz "$EX/exp008_mixed_configs/model_a14ch_xa_mix/best.pt" "$EX/exp008_mixed_configs/model_a14ch_xa_mix/figures_sn" "exp008 mixed 14ch (xa+config) - MAST SN"
viz dn_fno_2608/data_v5/dn/test.npz "$EX/exp009_split_configs/model_dn_13ch/best.pt"       "$EX/exp009_split_configs/model_dn_13ch/figures"        "exp009 dedicated DN 13ch - MAST DN"
viz dn_fno_2608/data_v5/sn/test.npz "$EX/exp009_split_configs/model_sn_13ch/best.pt"       "$EX/exp009_split_configs/model_sn_13ch/figures"        "exp009 dedicated SN 13ch - MAST SN"
viz dn_fno_2608/data_v5/dn/test.npz "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/best.pt" "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/figures_dn" "exp010 mixed 13ch (no config) - MAST DN"
viz dn_fno_2608/data_v5/sn/test.npz "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/best.pt" "$EX/exp010_mixed_no_config/model_a13ch_xa_mix/figures_sn" "exp010 mixed 13ch (no config) - MAST SN"

echo "== done =="
