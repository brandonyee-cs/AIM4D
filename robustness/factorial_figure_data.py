import os

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
DIMS = [("risk_set", "all", "at-risk"), ("label", "window", "future-only"),
        ("origin", "fixed-2019", "rolling"), ("closure", "none", "enforced")]


def marginals(path):
    d = pd.read_csv(os.path.join(OUT, path))
    rows = []
    for dim, permissive, strict in DIMS:
        others = [c for c, _, _ in DIMS if c != dim] + ["learner"]
        a = d[d[dim] == strict].set_index(others)["auc"]
        b = d[d[dim] == permissive].set_index(others)["auc"]
        diff = (a - b).dropna()
        rows.append({"dimension": dim, "n_pairs": len(diff), "mean": round(float(diff.mean()), 4),
                     "lo": round(float(diff.min()), 4), "hi": round(float(diff.max()), 4),
                     "same_sign": int(bool((diff < 0).all() or (diff > 0).all()))})
    return pd.DataFrame(rows)


def main():
    out = []
    for tag, path in [("ledger", "design_factorial.csv"), ("ert", "design_factorial_ert.csv")]:
        m = marginals(path); m.insert(0, "outcome", tag); out.append(m)
    pl = pd.read_csv(os.path.join(OUT, "closure_placebo.csv"))
    obs = float(pl[pl.kind == "observed"].mean_contrast.iloc[0])
    perm = pl[pl.kind == "placebo"].mean_contrast
    out.append(pd.DataFrame([{"outcome": "placebo", "dimension": "closure_permuted",
                              "n_pairs": len(perm), "mean": round(float(perm.mean()), 4),
                              "lo": round(float(perm.min()), 4), "hi": round(float(perm.max()), 4),
                              "same_sign": int((perm < obs).all())},
                             {"outcome": "placebo", "dimension": "closure_observed", "n_pairs": 1,
                              "mean": round(obs, 4), "lo": round(obs, 4), "hi": round(obs, 4),
                              "same_sign": 1}]))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(os.path.join(OUT, "factorial_figure_data.csv"), index=False)
    print(df.to_string(index=False))
    print("\nplacebo draws more negative than observed: "
          f"{int((perm < obs).sum())} of {len(perm)}")


if __name__ == "__main__":
    main()
