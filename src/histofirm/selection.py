"""Selection is cancer-balanced on fixed truth support; final reporting uses inherited metrics."""

from __future__ import annotations
import numpy as np


def selection(y, p, cancers, train_support, min_n=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    cs = np.asarray(cancers)
    rows = []
    for c in sorted(set(cs.tolist())):
        ix = cs == c
        Y = y[ix]
        P = p[ix]
        finite = np.isfinite(Y)
        n = finite.sum(0)
        mu = np.divide(np.where(finite, Y, 0).sum(0), n, out=np.zeros(Y.shape[1]), where=n > 0)
        dy = np.where(finite, Y - mu, 0)
        vy = (dy * dy).sum(0)
        eligible = np.asarray(train_support[c], bool) & (n >= min_n) & (vy > 1e-16)
        if not eligible.any():
            raise ValueError("NO_SELECTION_SUPPORT " + str(c))
        if np.any(finite[:, eligible] & ~np.isfinite(P[:, eligible])):
            raise ValueError("NONFINITE_SELECTION_PREDICTION")
        mp = np.where(finite, P, 0).sum(0) / np.maximum(n, 1)
        dp = np.where(finite, P - mp, 0)
        vp = (dp * dp).sum(0)
        denom = np.sqrt(vy * vp)
        r = np.divide((dy * dp).sum(0), denom, out=np.zeros_like(vy), where=denom > 1e-16)
        mse = np.where(finite, (np.nan_to_num(Y) - P) ** 2, 0).sum(0) / np.maximum(n, 1)
        # Constant predictions receive zero during selection, not silently dropped.
        rows.append(
            {
                "cancer": str(c),
                "n": int(ix.sum()),
                "targets": int(eligible.sum()),
                "pearson": float(np.clip(r[eligible], -1, 1).mean()),
                "mse": float(mse[eligible].mean()),
                "constant_predictions": int((eligible & (vp <= 1e-16)).sum()),
            }
        )
    return {
        "pearson": float(np.mean([r["pearson"] for r in rows])),
        "mse": float(np.mean([r["mse"] for r in rows])),
        "per_cancer": rows,
        "constant_prediction_selection_policy": "ZERO_ON_FIXED_VARIABLE_TRUTH_SUPPORT",
    }
