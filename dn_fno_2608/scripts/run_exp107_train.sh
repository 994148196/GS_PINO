#!/usr/bin/env bash
# exp107: pure-data baseline for paper Table 2 — coil 18ch input, DN-only (N=500)
# Pure MSE on total psi (no coil separation, no physics residual); mirrors the
# exp011 setup (train_dn_fno --input-mode coils) minus the mixed SN data, so the
# Table-2 comparison on the DN test set is apples-to-apples with exp101/102.
# Usage: bash dn_fno_2608/scripts/run_exp107_train.sh [train|eval|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
DN="dn_fno_2608/data_v5/dn"
E107="dn_fno_2608/experiments/exp107_pure_data_dn500"
MODE="${1:-all}"
mkdir -p "$E107"

train_107() {
  echo "== exp107: pure-data baseline coil 18ch DN-only (N=500, 800 epochs) =="
  # --no-config-channel: data_v5 npz now carries an all-zero config column;
  # exp011 (18ch) predates it, so drop it to keep the exact exp011 input spec
  "$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
    --train-data "$DN/train.npz" --val-data "$DN/val.npz" \
    --input-mode coils --n-train 500 --seed 1 --no-config-channel \
    --out-dir "$E107" 2>&1 | tee "$E107/train.log" | tail -6
}

eval_107() {
  echo "== exp107 eval (DN test, n=500) =="
  "$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
    --test-data "$DN/test.npz" --checkpoint "$E107/best.pt" \
    --out-dir "$E107/eval_dn" 2>&1 | tee "$E107/eval.log" | tail -8
  echo "== exp107 figures (DN test) =="
  "$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
    --test-data "$DN/test.npz" --checkpoint "$E107/best.pt" \
    --out-dir "$E107/figures_dn" --machine mast \
    --title "exp107 pure-data coil 18ch DN-only (N=500) - DN test" 2>&1 | tail -4
}

case "$MODE" in
  train) train_107 ;;
  eval) eval_107 ;;
  *) train_107; eval_107 ;;
esac
echo "== done =="
