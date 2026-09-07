#!/usr/bin/env bash

set -u
cd "$(dirname "$0")/.." || exit 1

export AIM4D_QUICK=1
export AIM4D_HMM_RESTARTS=10

LOG_DIR="brev"
mkdir -p "$LOG_DIR"
SKIP_GNN="${1:-}"

PASSED=()
FAILED=()

run_one() {
    local name="$1"
    local cmd="$2"
    local timeout_s="$3"
    local out_csv="$4"
    echo
    echo "=== SMOKE-TEST: $name (timeout ${timeout_s}s) ==="
    if timeout "${timeout_s}" bash -c "$cmd" > "$LOG_DIR/smoke_${name}.log" 2>&1; then
        if [ -n "$out_csv" ] && [ ! -f "$out_csv" ]; then
            echo "  FAIL: ran but did not write $out_csv"
            FAILED+=("$name (no output)")
            tail -20 "$LOG_DIR/smoke_${name}.log"
        else
            echo "  PASS"
            PASSED+=("$name")
        fi
    else
        echo "  FAIL (exit $? or timeout)"
        FAILED+=("$name")
        tail -20 "$LOG_DIR/smoke_${name}.log"
    fi
}

run_one "lead_time_auc" \
    "python3 -u robustness/lead_time_auc.py" \
    120 \
    "robustness/lead_time_auc.csv"

run_one "alternate_labels" \
    "python3 -u robustness/alternate_labels.py" \
    180 \
    "robustness/alternate_labels.csv"

run_one "permutation_importance_oos" \
    "python3 -u robustness/permutation_importance_oos.py" \
    600 \
    "robustness/permutation_importance_oos.csv"

run_one "elastic_net_robustness" \
    "python3 -u robustness/elastic_net_robustness.py" \
    600 \
    "robustness/elastic_net_robustness.csv"

run_one "dsp_imputation_robustness" \
    "python3 -u robustness/dsp_imputation_robustness.py" \
    600 \
    "robustness/dsp_imputation_robustness.csv"

run_one "hyperparameter_sensitivity" \
    "python3 -u robustness/hyperparameter_sensitivity.py" \
    1800 \
    "robustness/hyperparameter_sensitivity.csv"

if [ "$SKIP_GNN" != "fast" ]; then
    run_one "gnn_counterfactual" \
        "python3 -u robustness/gnn_counterfactual.py" \
        900 \
        "robustness/gnn_counterfactual.csv"
else
    echo "Skipping gnn_counterfactual (fast mode)"
fi

run_one "sample_pipeline_loeo_smoke" \
    "AIM4D_SMOKE_LIMIT=1 python3 -u robustness/sample_pipeline_loeo.py" \
    1800 \
    ""

echo
echo "=== Restoring canonical pipeline state after Task F smoke run ==="
unset AIM4D_QUICK AIM4D_HMM_RESTARTS AIM4D_EXCLUDE_COUNTRY
for stage in stage1_factors/extract.py stage2_betas/estimate.py \
             stage3_msvar/estimate.py stage4_nscm/estimate.py \
             stage5_ews/estimate.py; do
    echo "  rerunning $stage ..."
    if python3 -u "$stage" > "$LOG_DIR/restore_$(basename $stage .py).log" 2>&1; then
        echo "    OK"
    else
        echo "    FAILED (see $LOG_DIR/restore_$(basename $stage .py).log)"
        FAILED+=("restore_$(basename $stage .py)")
    fi
done

echo
echo "================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================"
echo "PASSED (${#PASSED[@]}): ${PASSED[*]}"
echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    echo "Logs for failed runs:"
    for f in "${FAILED[@]}"; do
        echo "  $LOG_DIR/smoke_${f%% *}.log"
    done
    exit 1
fi
echo
echo "All smoke tests passed. Safe to run full configurations on Brev."
