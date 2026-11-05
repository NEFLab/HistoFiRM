"""Fixed cancer/target metric rows, with explicit full-target and frozen-panel coverage."""

from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.stats import rankdata

KEYS = ["pearson", "spearman", "mae", "rmse", "nrmse", "ccc", "r2", "bias"]


def one(y, p, min_n=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or min_n < 2:
        raise ValueError("METRIC_VECTOR_AXIS")
    m = np.isfinite(y)
    n = int(m.sum())
    out = {k: None for k in KEYS}
    out.update(n_total=len(y), n_truth=n, n_prediction=int((m & np.isfinite(p)).sum()))
    if n < min_n:
        return {**out, "status": "LOW_N"}
    if not np.isfinite(p[m]).all():
        return {**out, "status": "MISSING_PREDICTION"}
    a = y[m]
    b = p[m]
    da = a - a.mean()
    db = b - b.mean()
    sy = da @ da
    sp = db @ db
    mse = float(np.mean((a - b) ** 2))
    out.update(mae=float(np.mean(abs(a - b))), rmse=mse**0.5, bias=float(np.mean(b - a)))
    out["r2"] = None if sy == 0 else float(1 - mse * n / sy)
    out["nrmse"] = None if sy == 0 else float((mse / (sy / (n - 1))) ** 0.5)
    cccden = sy / n + sp / n + (a.mean() - b.mean()) ** 2
    out["ccc"] = None if cccden == 0 else float(2 * (da @ db) / n / cccden)
    if sy == 0:
        return {**out, "status": "CONSTANT_TRUTH"}
    if sp == 0:
        return {**out, "status": "CONSTANT_PREDICTION"}
    ra = rankdata(a)
    rb = rankdata(b)
    return {
        **out,
        "pearson": float(np.clip((da @ db) / np.sqrt(sy * sp), -1, 1)),
        "spearman": float(np.corrcoef(ra, rb)[0, 1]),
        "status": "ESTIMABLE",
    }


def validate_panel(panel, cancers, targets):
    if len(set(targets)) != len(targets):
        raise ValueError("DUPLICATE_TARGET")
    if panel is None:
        return
    for cancer in set(cancers):
        if cancer not in panel:
            raise ValueError("PANEL_CANCER_MISSING")
        ts = panel[cancer]
        if len(set(ts)) != len(ts) or not set(ts).issubset(set(targets)):
            raise ValueError("PANEL_TARGET_IDENTITY")


def table(y, p, cancers, targets, panel=None, min_n=10):
    y = np.asarray(y)
    p = np.asarray(p)
    c = np.asarray(cancers)
    if y.shape != p.shape or y.shape != (len(c), len(targets)):
        raise ValueError("METRIC_AXIS")
    validate_panel(panel, c.tolist(), targets)
    rows = []
    for cancer in sorted(set(c)):
        a = c == cancer
        for j, t in enumerate(targets):
            rows.append(
                {
                    "cancer": str(cancer),
                    "target_id": str(t),
                    "panel_membership": panel is not None and t in panel[cancer],
                    **one(y[a, j], p[a, j], min_n),
                }
            )
    return rows


def coverage(rows):
    return {
        "fixed_cells": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "estimable_by_metric": {
            k: int(sum(r.get(k) is not None and np.isfinite(r[k]) for r in rows)) for k in KEYS
        },
        "n_truth_cells": sum(r["n_truth"] for r in rows),
        "n_predicted_cells": sum(r["n_prediction"] for r in rows),
    }


def macro(rows, key="pearson", allow_conditional=False):
    """Equal cancer means on exactly the supplied support; no post hoc cell deletion."""
    missing = int(sum(r.get(key) is None or not np.isfinite(r[key]) for r in rows))
    if missing and not allow_conditional:
        return {
            "value": None,
            "status": "INCOMPLETE_FIXED_SUPPORT",
            "missing_cells": missing,
            "fixed_cells": len(rows),
        }
    values = []
    for c in sorted({r["cancer"] for r in rows}):
        v = [
            r[key]
            for r in rows
            if r["cancer"] == c and r.get(key) is not None and np.isfinite(r[key])
        ]
        if not v:
            return {
                "value": None,
                "status": "NO_ESTIMABLE_CANCER",
                "missing_cells": missing,
                "fixed_cells": len(rows),
            }
        values.append(float(np.mean(v)))
    return {
        "value": float(np.mean(values)) if values else None,
        "status": ("CONDITIONAL" if missing else "COMPLETE") if values else "EMPTY_SUPPORT",
        "missing_cells": missing,
        "fixed_cells": len(rows),
    }
