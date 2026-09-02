#!/usr/bin/env bash
# exp313: POD 固定基 + 并联小 FNO 残差（"POD 只留几个基函数不够，并联一个网络补充"）
#   exp313 = pod_residual：ψ = ψ0(POD 基, 低秩先验) + 小 FNO 残差(学 y − ψ_pod)
#            POD 通道同 exp312（SVD 500 训练输出 z 域, p_psi=3/p_j=10 冻结 buffer）
#            残差 = FNO2d2608 骨架瘦身（width 32, modes 12, 4 层, ≈0.59M）
#   ——分阶段冻结协议（防通道竞争：POD 与残差同时训会互相抢信号）——
#      epochs 1..N      ：只训 branch（残差冻结从 0 起步，POD 通道先收敛）
#      epoch N+1 起     ：冻结 branch，解冻残差学 y − ψ_pod
#      阶段1 切换挂起到 N 结束（POD 通道常提前达标，挂起保证残差开始学习时
#      物理 ramp 同步从 0 爬升，阶段1/2 边界不割裂残差训练）
# 数据/两阶段方案同 exp102：18ch = R,Z + 5 params + 11 线圈电流；
#   阶段1 监督 psi_plasma+J（val rel L2 < 3% 或 e300 切阶段2），阶段2 自洽
#   GS 残差 + Ip 约束（物理权重 ramp 预热，防阶段2 首 epoch 爆炸）
# 日志/评估产物落实验目录；冒烟 30-epoch 落 _smoke_exp313/（pretrain 15 触发交接）
# 用法: bash dn_fno_2608/scripts/run_exp313_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E313="dn_fno_2608/experiments/exp313_pod_residual_pino_twostage_n500"
MODE="${1:-all}"

train_313() {
  echo "== exp313: pod_residual 两阶段 (N=500, 800 epochs, pretrain 60) =="
  mkdir -p "$E313"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model pod_residual \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 --pod-pretrain-epochs 60 \
    --out-dir "$E313" 2>&1 | tee "$E313/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟：pod-pretrain 15 → e16 冻结交接 + 阶段切换挂起
  local S="dn_fno_2608/experiments/_smoke_exp313"
  mkdir -p "$S"
  echo "== smoke pod_residual (30 epochs, pretrain 15) =="
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model pod_residual \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 --pod-pretrain-epochs 15 \
    --out-dir "$S" 2>&1 | tail -8
}

eval_313() {
  echo "== exp313 评估（DN test 500） =="
  mkdir -p "$E313"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E313/best.pt" \
    --out-dir "$E313" --fig-dir "$E313/figures" \
    --machine mast --title "exp313 POD基+小FNO残差+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E313/eval.log" | tail -6
}

case "$MODE" in
  smoke) smoke ;;
  train) train_313 ;;
  eval) eval_313 ;;
  *) smoke; train_313; eval_313 ;;
esac
echo "== done =="
