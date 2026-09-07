import numpy as np
from scipy.optimize import minimize


def gm_moments(u, Wu, WWu, trWtW, n):
    g = np.array([u @ u, Wu @ Wu, u @ Wu], dtype=float) / n
    G = np.array([
        [2 * (u @ Wu),        -(Wu @ Wu),    float(n)],
        [2 * (Wu @ WWu),      -(WWu @ WWu),  float(trWtW)],
        [(u @ WWu) + (Wu @ Wu), -(Wu @ WWu), 0.0],
    ], dtype=float) / n
    return G, g


def gm_lambda(u, wlag, trWtW, n=None, bounds=(-0.99, 0.99), strict=True):
    u = np.asarray(u, dtype=float)
    n = len(u) if n is None else n
    Wu = wlag(u)
    WWu = wlag(Wu)
    G, g = gm_moments(u, Wu, WWu, trWtW, n)
    scale = float(u @ u) / n
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    Gs, gs = G / scale, g / scale

    def obj(theta):
        lam, s2 = theta
        r = gs - Gs @ np.array([lam, lam * lam, s2])
        return float(r @ r)

    best, bestval, converged = np.nan, np.inf, False
    for lam0 in (-0.5, 0.0, 0.5):
        r = minimize(obj, x0=[lam0, 1.0], bounds=[bounds, (1e-12, None)],
                     method="L-BFGS-B", options={"ftol": 1e-14, "gtol": 1e-12})
        if r.fun < bestval:
            best, bestval, converged = float(r.x[0]), float(r.fun), bool(r.success)
    if strict and not converged:
        return np.nan
    return best


def _dense_wlag(W):
    return lambda v: W @ v


def monte_carlo(n=200, lam_true=0.6, reps=200, seed=0, kind="cycle"):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    if kind == "cycle":
        for i in range(n):
            W[i, (i + 1) % n] = 1.0
            W[i, (i - 1) % n] = 1.0
    else:
        for i in range(n):
            for j in rng.choice([k for k in range(n) if k != i], 4, replace=False):
                W[i, j] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    A = np.linalg.inv(np.eye(n) - lam_true * W)
    trWtW = np.trace(W.T @ W)
    est = np.array([gm_lambda(A @ rng.normal(size=n), _dense_wlag(W), trWtW, n)
                    for _ in range(reps)])
    ok = np.isfinite(est)
    return float(np.mean(est[ok])), float(np.std(est[ok])), float(ok.mean())


def scale_invariance_check(n=200, lam=0.55, seed=7):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1.0
        W[i, (i - 1) % n] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    u = np.linalg.inv(np.eye(n) - lam * W) @ rng.normal(size=n)
    tr = np.trace(W.T @ W)
    vals = [gm_lambda(u * s, _dense_wlag(W), tr, n) for s in (0.01, 1.0, 100.0)]
    return max(vals) - min(vals), vals


if __name__ == "__main__":
    import sys
    print("Monte Carlo check on the estimator (mean estimate over 200 replications)\n")
    ok = True
    for kind in ("cycle", "random"):
        for lam in (-0.4, 0.0, 0.3, 0.6, 0.8):
            m, sd, conv = monte_carlo(lam_true=lam, kind=kind)
            bias = m - lam
            good = abs(bias) < 0.05 and conv > 0.5
            ok &= good
            print(f"  {kind:<7} true {lam:+.2f}   estimate {m:+.4f}  sd {sd:.4f}  "
                  f"bias {bias:+.4f}  converged {conv:.0%}  {'ok ' if good else 'FAIL'}")
    spread, vals = scale_invariance_check()
    print(f"\nscale invariance: lambda over units 0.01, 1, 100 = "
          f"{', '.join(f'{v:.6f}' for v in vals)}  spread {spread:.2e}")
    ok &= spread < 1e-3
    if ok:
        print("estimator recovers the known parameter and is invariant to units")
    else:
        print("ESTIMATOR FAILS ITS OWN CHECK")
        sys.exit(1)
