#!/usr/bin/env bash
# exp008/exp009 evaluation: overall + per-config buckets + cross-config.
#   混合模型 (exp008) 对 整体/DN桶/SN桶 分别评估；
#   专职模型 (exp009) 对自身配置评估 + 跨配置交叉（泛化代价）。
set -e
cd "$(dirname "$0")/../.."
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
V5="dn_fno_2608/data_v5"
DN_TEST="$V5/dn/test.npz"; SN_TEST="$V5/sn/test.npz"
E8="dn_fno_2608/experiments/exp008_mixed_configs/model_a14ch_xa_mix"
E9="dn_fno_2608/experiments/exp009_split_configs"

eval_model() {
    local name=$1 ckpt=$2 test=$3 out=$4
    echo "=== $name on $test ==="
    "$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno --test-data "$test" \
      --checkpoint "$ckpt" --out-dir "$out" 2>&1 | tail -4
}

# exp008 mixed model: overall + per-config buckets
eval_model "exp008-mix" "$E8/best.pt" "$DN_TEST,$SN_TEST" "$E8/eval_all"
eval_model "exp008-mix" "$E8/best.pt" "$DN_TEST"          "$E8/eval_dn"
eval_model "exp008-mix" "$E8/best.pt" "$SN_TEST"          "$E8/eval_sn"

# exp009 specialist models: own config + cross-config generalization
eval_model "exp009-dn" "$E9/model_dn_13ch/best.pt" "$DN_TEST" "$E9/model_dn_13ch/eval"
eval_model "exp009-sn" "$E9/model_sn_13ch/best.pt" "$SN_TEST" "$E9/model_sn_13ch/eval"
eval_model "exp009-dn" "$E9/model_dn_13ch/best.pt" "$SN_TEST" "$E9/model_dn_13ch/eval_cross_sn"
eval_model "exp009-sn" "$E9/model_sn_13ch/best.pt" "$DN_TEST" "$E9/model_sn_13ch/eval_cross_dn"

echo "=== evaluation done ==="
