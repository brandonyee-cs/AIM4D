import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_comparison import build_labels
from external_benchmarks import load_panel, load_indicator_panel
from benchmark_stress_test import holdout, _postmask

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

NEAR_OUTCOME = re.compile(r"regend|regint|regimpgroup|reginfo|_stock$|v2reg")


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, max_depth=3,
                                          learning_rate=0.05, random_state=42)


def run_leakage_audit():
    ind, indicators = load_indicator_panel()
    near = [c for c in indicators if NEAR_OUTCOME.search(c)]
    clean = [c for c in indicators if c not in near]

    a_full = holdout(ind, indicators, gbm)
    a_clean = holdout(ind, clean, gbm)
    print(f"  raw 332 (all)           AUC={a_full[0]:.3f}  AUC-PR={a_full[1]:.3f}  [{len(indicators)} feats]")
    print(f"  raw (near-outcome out)  AUC={a_clean[0]:.3f}  AUC-PR={a_clean[1]:.3f}  [{len(clean)} feats, dropped {len(near)}]")
    print(f"  dropped: {near}")
    print(f"  -> AUC delta {a_clean[0]-a_full[0]:+.3f}, AUC-PR delta {a_clean[1]-a_full[1]:+.3f}")
    return {"check": "leakage_audit", "auc_full": a_full[0], "aucpr_full": a_full[1],
            "auc_clean": a_clean[0], "aucpr_clean": a_clean[1],
            "n_dropped": len(near), "dropped": ";".join(near)}


def build_non_vdem_panel():
    panel = load_panel()
    gd = pd.read_csv(os.path.join(REPO, "data", "gdelt_country_year.csv"))
    gd = gd.rename(columns={"country_code": "country_text_id"})
    gcols = ["protest_count", "conflict_count", "repression_count", "avg_goldstein",
             "avg_tone", "total_events"]
    gcols = [c for c in gcols if c in gd.columns]
    panel = panel.merge(gd[["country_text_id", "year"] + gcols],
                        on=["country_text_id", "year"], how="left")
    mp = pd.read_csv(os.path.join(REPO, "data", "macro_pitf.csv")).rename(columns={"iso3": "country_text_id"})
    mp_cols = ["inflation_yoy", "food_prod_index", "ext_debt_gni", "work_age_share"]
    mp_cols = [c for c in mp_cols if c in mp.columns]
    panel = panel.merge(mp[["country_text_id", "year"] + mp_cols],
                        on=["country_text_id", "year"], how="left")

    feats = (["gdp_pc", "gdp_growth", "log_gdp_pc", "infant_mortality",
              "n_backsliding_neighbors"] + gcols + mp_cols)
    feats = [c for c in feats if c in panel.columns]
    return panel, feats


def run_non_vdem_ablation():
    panel, feats = build_non_vdem_panel()
    a = holdout(panel, feats, gbm)
    print(f"  non-V-Dem only (GDELT+macro)  AUC={a[0]:.3f}  AUC-PR={a[1]:.3f}  "
          f"[{len(feats)} feats, n_pos={a[2]}]")
    print(f"  features: {feats}")
    print(f"  [ref: raw 332 V-Dem AUC~0.94/AUC-PR~0.66; AIM4D 0.931/0.649]")
    return {"check": "non_vdem_ablation", "auc": a[0], "aucpr": a[1],
            "n_feats": len(feats), "n_pos": a[2]}


def main():
    print("=" * 70)
    print("BENCHMARK FINISHERS (strict 2019 hold-out)")
    print("=" * 70)
    print("\n[1] Leakage audit: drop near-outcome V-Dem indicators")
    r1 = run_leakage_audit()
    print("\n[2] Non-V-Dem-predictor ablation (circularity check)")
    r2 = run_non_vdem_ablation()
    pd.DataFrame([r1, r2]).to_csv(os.path.join(OUTPUT_DIR, "benchmark_finishers_results.csv"), index=False)
    print(f"\nSaved to robustness/benchmark_finishers_results.csv")


if __name__ == "__main__":
    main()
