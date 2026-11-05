"""Cross-fitted composition-conditioned associations on matched support."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import t as student_t
from .common import require
from .statistics import corr, bh


@dataclass
class Preprocessor:
    median: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    keep: np.ndarray

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, dtype=np.float64)
        require(x.ndim == 2 and x.shape[0] > 0 and not np.isinf(x).any(), "Bad preprocessor input")
        med = np.array(
            [
                np.nanmedian(x[:, i]) if np.isfinite(x[:, i]).any() else 0.0
                for i in range(x.shape[1])
            ]
        )
        a = np.where(np.isfinite(x), x, med)
        mu = a.mean(0)
        sd = a.std(0)
        require(
            np.isfinite(mu).all() and np.isfinite(sd).all(),
            "Nonfinite training preprocessing statistics",
        )
        keep = sd > 1e-12
        sd[~keep] = 1.0
        # A zero-column no-intercept Cox is a valid null-risk model when every
        # allowed training covariate is constant; do not invent a varying column.
        return cls(med, mu, sd, keep)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        require(
            x.ndim == 2 and x.shape[1] == len(self.mean) and not np.isinf(x).any(),
            "Covariate axis mismatch",
        )
        value = ((np.where(np.isfinite(x), x, self.median) - self.mean) / self.sd)[:, self.keep]
        require(np.isfinite(value).all(), "Nonfinite transformed covariates")
        return value


def residualize(y, p, x, fold, alpha=10.0):
    ry = np.full_like(y, np.nan, dtype=float)
    rp = ry.copy()
    base = ry.copy()
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        if len(ix) < 15:
            continue
        for aa, bb in KFold(3, shuffle=True, random_state=20260926 + int(k)).split(ix):
            tr = ix[aa]
            te = ix[bb]
            pp = Preprocessor.fit(x[tr])
            xt = pp.transform(x[tr])
            xv = pp.transform(x[te])
            for j in range(y.shape[1]):
                finite = np.isfinite(y[tr, j]) & np.isfinite(p[tr, j])
                if finite.sum() < max(8, xt.shape[1] + 2):
                    continue
                if xt.shape[1] == 0:
                    my = np.full(len(te), np.mean(y[tr[finite], j]))
                    mp = np.full(len(te), np.mean(p[tr[finite], j]))
                else:
                    my = Ridge(alpha=alpha).fit(xt[finite], y[tr[finite], j]).predict(xv)
                    mp = Ridge(alpha=alpha).fit(xt[finite], p[tr[finite], j]).predict(xv)
                ry[te, j] = y[te, j] - my
                rp[te, j] = p[te, j] - mp
                base[te, j] = my
    return ry, rp, base


def evaluate(y, p, x, fold, target_ids):
    """Run separately within each cancer; folds identify the upstream image model."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    x = np.asarray(x, float)
    fold = np.asarray(fold)
    if (
        y.shape != p.shape
        or len(y) != len(x)
        or len(y) != len(fold)
        or y.shape[1] != len(target_ids)
    ):
        raise ValueError("Association axes differ.")
    ry, rp, base = residualize(y, p, x, fold)
    records = []
    for j, target in enumerate(target_ids):
        ok = np.isfinite(ry[:, j]) & np.isfinite(rp[:, j])
        n = int(ok.sum())
        r = corr(ry[:, j], rp[:, j])
        raw = corr(y[ok, j], p[ok, j])
        composition = corr(y[ok, j], base[ok, j])
        low = high = pv = np.nan
        if n > 3 and np.isfinite(r):
            z = np.arctanh(np.clip(r, -0.999999999, 0.999999999))
            low, high = np.tanh(z + np.array([-1, 1]) * 1.96 / np.sqrt(n - 3))
            pv = 2 * student_t.sf(abs(r) * np.sqrt((n - 2) / max(1 - r * r, 1e-16)), n - 2)
        records.append(
            dict(
                target=target,
                n=n,
                raw_r=raw,
                composition_only_r=composition,
                adjusted_r=r,
                delta_r_after_adjustment=r - raw,
                ci_low=low,
                ci_high=high,
                p_approx=pv,
            )
        )
    result = pd.DataFrame(records)
    result["q_within_cancer"] = bh(result.p_approx)
    return result
