#!/usr/bin/env bash
# exp314: POD 固定基 + 并联 UNet 残差（exp313 残差通道从小 FNO 换成纯卷积）
#   exp314 = pod_residual_unet：ψ = ψ0(POD 基, 低秩先验) + UNet 残差(学 y − ψ_pod)
#            POD 通道同 exp312/313（SVD 500 训练输出 z 域, p_psi=3/p_j=10 冻结 buffer）
#            残差 = UNet2d2608 瘦身（base 16、depth 4，≈1.23M）——纯局部卷积
#   科学问题（补 exp301 vs exp313 缺环）：exp301 纯卷积完整任务失败（2.202%，
#   局部感受野不足承载磁面全局拓扑）、exp313 谱卷积残差成功（0.883%）——
#   基外细节（X 点尺度结构）是"局部的"（POD 基已供全局形状）还是
#   "在残差通道上也需要全局混合"？UNet 残差若成功 → 局部可表达；若失败
#   → 全局性在任意通道上都必要
# 训练协议同 exp313：--pod-pretrain-epochs 60 分阶段冻结（先 branch 后残差，
#   阶段1 切换挂起到 pretrain 结束）；其余参数与 exp102 逐项一致
# 用法: bash dn_fno_2608/scripts/run_exp314_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E314="dn_fno_2608/experiments/exp314_pod_residual_unet_pino_twostage_n500"
MODE="${1:-all}"

train_314() {
  echo "== exp314: pod_residual_unet 两阶段 (N=500, 800 epochs, pretrain 60) =="
  mkdir -p "$E314"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model pod_residual_unet \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 --pod-pretrain-epochs 60 \
    --out-dir "$E314" 2>&1 | tee "$E314/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟：pod-pretrain 15 → e16 冻结交接 + 阶段切换挂起
  local S="dn_fno_2608/experiments/_smoke_exp314"
  mkdir -p "$S"
  echo "== smoke pod_residual_unet (30 epochs, pretrain 15) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model pod_residual_unet \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 --pod-pretrain-epochs 15 \
    --out-dir "$S" 2>&1 | tail -8
}

eval_314() {
  echo "== exp314 评估（DN test 500） =="
  mkdir -p "$E314"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E314/best.pt" \
    --out-dir "$E314" --fig-dir "$E314/figures" \
    --machine mast --title "exp314 POD基+UNet残差+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E314/eval.log" | tail -6
}

case "$MODE" in
  smoke) smoke ;;
  train) train_314 ;;
  eval) eval_314 ;;
  *) smoke; train_314; eval_314 ;;
esac
echo "== done =="
