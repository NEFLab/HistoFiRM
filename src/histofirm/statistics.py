import numpy as np
from scipy.stats import rankdata
from .common import require


def corr(x, y, rank=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    if rank:
        x = rankdata(x)
        y = rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def matrix_correlations(a, b):
    require(a.shape == b.shape, "TARGET_SHAPE_MISMATCH")
    out = []
    for j in range(a.shape[1]):
        ok = np.isfinite(a[:, j]) & np.isfinite(b[:, j])
        x = a[ok, j]
        y = b[ok, j]
        den = np.sum((x - x.mean()) ** 2) if len(x) else 0
        out.append(
            {
                "n": int(ok.sum()),
                "pearson": corr(x, y),
                "spearman": corr(x, y, True),
                "mae": float(np.mean(abs(x - y))) if len(x) else np.nan,
                "r2": float(1 - np.sum((x - y) ** 2) / den) if den > 1e-12 else np.nan,
            }
        )
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ii = np.flatnonzero(np.isfinite(p))
    v = p[ii]
    if len(v):
        o = np.argsort(v)
        q = np.minimum.accumulate((v[o] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1]
        out[ii[o]] = np.minimum(q, 1)
    return out


def bootstrap_mean(values, B=1000, seed=20260926):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "ci95": None}
    if len(a) == 1:
        return {"n": 1, "mean": float(a[0]), "ci95": None, "status": "ONE_DONOR_DESCRIPTIVE_ONLY"}
    rng = np.random.default_rng(seed)
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)])
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "ci95": np.quantile(b, [0.025, 0.975]).tolist(),
        "patient_resampling": True,
    }


def permutation_mean_p(v, B=1999, seed=20260926):
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(a.mean())
    s = 0
    for _ in range(B):
        s += abs((a * rng.choice([-1, 1], len(a))).mean()) >= obs
    return (s + 1) / (B + 1)
