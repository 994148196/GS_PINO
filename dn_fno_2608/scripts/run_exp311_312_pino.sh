#!/usr/bin/env bash
# exp311/312: 第二轮架构候选（ARCHS_SURVEY.md §5 调研结论）
#   exp311 = tkno_lite：TKNO（arXiv:2511.19114，ICML 2026，同一 GS 任务胜出者）
#           去 KAN（exp303 已证逐点 KAN 增益 ~0）→ conv stem 保 65² 细节 +
#            patch 嵌入 65²→32²（1024 token 控注意力成本）+ 4×全局自注意力块
#            + stem-skip 解码（≈1.15M 参数）；全局混合由注意力而非谱卷积承载
#   exp312 = pod_deeponet：POD 基替代学习 trunk（Lu 2022 POD-DeepONet）——
#            SVD(500 训练输出 z 域) → top-p 基作固定 buffer，分支 MLP 16 标量
#            → 系数；物理微调只动系数（低秩先验即实验假说）
# 数据/两阶段方案同 exp102：18ch = R,Z + 5 params + 11 线圈电流；
#   阶段1 监督 psi_plasma+J（val rel L2 < 3% 或 e300 切阶段2），阶段2 自洽
#   GS 残差 + Ip 约束（物理权重 ramp 预热，防阶段2 首 epoch 爆炸）
# 日志/评估产物落各自实验目录；冒烟 30-epoch 落 _smoke_exp309_312/
# 用法: bash dn_fno_2608/scripts/run_exp311_312_pino.sh [smoke|train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E311="dn_fno_2608/experiments/exp311_tkno_lite_pino_twostage_n500"
E312="dn_fno_2608/experiments/exp312_pod_deeponet_pino_twostage_n500"
MODE="${1:-all}"

train_311() {
  echo "== exp311: tkno_lite 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E311"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model tkno_lite \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E311" 2>&1 | tee "$E311/train.log" | tail -6
}

train_312() {
  echo "== exp312: pod_deeponet 两阶段 (N=500, 800 epochs) =="
  mkdir -p "$E312"
  "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model pod_deeponet \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
    --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
    --stage2-ramp-epochs 30 \
    --out-dir "$E312" 2>&1 | tee "$E312/train.log" | tail -6
}

smoke() { # 30-epoch 冒烟
  local S="dn_fno_2608/experiments/_smoke_exp309_312"
  mkdir -p "$S"
  for m in tkno_lite pod_deeponet; do
    echo "== smoke $m (30 epochs) =="
    "$PY" -u -m gs_pino_fno_phys.train_pino --mode twostage --model "$m" \
      --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
      --n-train 500 --seed 1 --epochs 30 --phys-weight 0.1 --ip-weight 1.0 \
      --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
      --stage2-ramp-epochs 30 \
      --out-dir "$S/$m" 2>&1 | tail -4
  done
}

eval_311() {
  echo "== exp311 评估（DN test 500） =="
  mkdir -p "$E311"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E311/best.pt" \
    --out-dir "$E311" --fig-dir "$E311/figures" \
    --machine mast --title "exp311 TKNO-lite+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E311/eval.log" | tail -6
}

eval_312() {
  echo "== exp312 评估（DN test 500） =="
  mkdir -p "$E312"
  "$PY" -u -m gs_pino_fno_phys.evaluate_pino \
    --test-data "$DN/test.npz" --checkpoint "$E312/best.pt" \
    --out-dir "$E312" --fig-dir "$E312/figures" \
    --machine mast --title "exp312 POD-DeepONet+GS twostage (N=500, DN test)" \
    2>&1 | tee "$E312/eval.log" | tail -6
}

case "$MODE" in
  smoke) smoke ;;
  train) train_311; train_312 ;;
  eval) eval_311; eval_312 ;;
  *) smoke; train_311; train_312; eval_311; eval_312 ;;
esac
echo "== done =="
