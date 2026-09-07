#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
LOG="brev/run_all.log"
: > "$LOG"

exec > >(stdbuf -oL tee -a "$LOG") 2>&1

echo "================================================================"
echo "AIM4D full pipeline run started $(date -u +%FT%TZ)"
echo "Host: $(hostname)  CPUs: $(nproc 2>/dev/null || echo '?')  RAM: $(free -h 2>/dev/null | awk '/Mem:/ {print $2}' || echo '?')"
echo "FAST_ONLY=${FAST_ONLY:-0}  SKIP_GDELT=${SKIP_GDELT:-0}"
echo "================================================================"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

for f in data/vdem_v16.csv data/contiguity/DirectContiguity320 data/atop; do
  if [[ ! -e $f ]]; then
    echo "[ERROR] Missing $f. Run brev/upload.sh from your laptop first." >&2
    exit 1
  fi
done

if [[ ! -f data/gdelt_country_year.csv ]] && [[ "${SKIP_GDELT:-0}" != "1" ]]; then
  echo "--- Downloading GDELT (first run only; ~30 min on 16 vCPU) ---"
  python3 -u data/download_gdelt.py
fi

echo
echo "================================================================"
echo "STAGE 1: POET factor extraction"
echo "================================================================"
python3 -u stage1_factors/extract.py

echo
echo "================================================================"
echo "STAGE 2: Kalman + DCC-GARCH betas"
echo "================================================================"
python3 -u stage2_betas/estimate.py

echo
echo "================================================================"
echo "STAGE 3: MS-VAR HMM regime classification"
echo "================================================================"
python3 -u stage3_msvar/estimate.py

echo
echo "================================================================"
echo "STAGE 4: INE-TARNet network contagion"
echo "================================================================"
python3 -u stage4_nscm/estimate.py

echo
echo "================================================================"
echo "STAGE 5: Multi-channel early warning + meta-learner"
echo "================================================================"
python3 -u stage5_ews/estimate.py

echo
echo "================================================================"
echo "ROBUSTNESS: DSP ablation"
echo "================================================================"
python3 -u robustness/dsp_ablation.py

echo
echo "================================================================"
echo "ROBUSTNESS: cluster-bootstrap 95% CIs"
echo "================================================================"
python3 -u robustness/bootstrap_cis.py

echo
echo "================================================================"
echo "ROBUSTNESS: multi-seed Stage 4 contagion sweep (10 seeds)"
echo "================================================================"
python3 -u robustness/contagion_seed_sweep.py

echo
echo "================================================================"
echo "ROBUSTNESS: network weight stability sweep (10 seeds)"
echo "================================================================"
python3 -u robustness/network_seed_sweep.py || echo "  (skipped — script optional)"

if [[ "${FAST_ONLY:-0}" == "1" ]]; then
  echo
  echo "FAST_ONLY=1 set — skipping Task E and Task F"
  echo "Total wall-time: $(( SECONDS / 60 )) min"
  echo "Output: $LOG"
  exit 0
fi

echo
echo "================================================================"
echo "TASK E: real expanding-window CV (4 folds, full refit per fold)"
echo "Expected: 2-3 hr"
echo "================================================================"
python3 -u robustness/expanding_window_cv.py

echo
echo "================================================================"
echo "TASK F: 5-episode full-pipeline LOEO"
echo "Expected: 2-3 hr"
echo "================================================================"
python3 -u robustness/sample_pipeline_loeo.py

echo
echo "================================================================"
echo "DONE. Total wall-time: $(( SECONDS / 60 )) min"
echo "Output: $LOG"
echo "================================================================"
