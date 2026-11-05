from __future__ import annotations
import numpy as np
from .selection import selection


def target_scaling(y, cancers, names):
    y = np.asarray(y, float)
    cs = np.asarray(cancers)
    T = y.shape[1]
    center = np.zeros((len(names), T))
    scale = np.ones_like(center)
    counts = np.zeros_like(center, dtype=np.int64)
    for k, c in enumerate(names):
        a = y[cs == c]
        finite = np.isfinite(a)
        n = finite.sum(0)
        counts[k] = n
        center[k] = np.divide(np.where(finite, a, 0).sum(0), n, out=np.zeros(T), where=n > 0)
        q = np.where(finite, a - center[k], 0)
        var = np.divide((q * q).sum(0), np.maximum(n - 1, 1), out=np.zeros(T), where=n > 1)
        sd = np.sqrt(np.maximum(var, 0))
        scale[k] = np.where(sd > 1e-10, sd, 1)
    supported = counts >= 2
    variable = supported & (scale != 1)
    # SD may legitimately equal 1; detect constants independently.
    for k, c in enumerate(names):
        a = y[cs == c]
        finite = np.isfinite(a)
        v = np.where(finite, a - center[k], 0)
        variable[k] = supported[k] & ((v * v).sum(0) > 1e-20)
    return center, scale, supported, variable, counts


def ridge_grid(X, Y, alphas):
    """SVD via the smaller Gram matrix, grouped by exact per-target missingness."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    D = X.shape[1]
    T = Y.shape[1]
    results = {float(a): (np.zeros((D, T)), np.zeros(T)) for a in alphas}
    groups = {}
    for j in range(T):
        groups.setdefault(np.packbits(np.isfinite(Y[:, j])).tobytes(), []).append(j)
    for js in groups.values():
        mask = np.isfinite(Y[:, js[0]])
        if mask.sum() < 2:
            continue
        x = X[mask]
        y = Y[mask][:, js]
        xm = x.mean(0)
        ym = y.mean(0)
        x = x - xm
        y = y - ym
        if len(x) <= D:
            eig, U = np.linalg.eigh(x @ x.T)
            eig = np.maximum(eig, 0)
            left = x.T @ U
            right = U.T @ y
        else:
            eig, U = np.linalg.eigh(x.T @ x)
            eig = np.maximum(eig, 0)
            left = U
            right = U.T @ (x.T @ y)
        for alpha, (B, b) in results.items():
            coef = (left / (eig + alpha)[None, :]) @ right
            B[:, js] = coef
            b[js] = ym - xm @ coef
    return results


def prepare_base(train, valid, cache, alpha_grid, forced_alphas=None, ln=False):
    """All population transforms are fitted only from `train`. Valid only selects alphas."""
    names = train.cancer_names
    T = len(train.target_ids)
    D = train.manifest["input_dim"]
    index = {s: i for i, s in enumerate(cache["patient_ids"])}
    it = np.array([index[s] for s in train.ids])
    iv = None if valid is None else np.array([index[s] for s in valid.ids])
    h = np.asarray(cache["ln_mean" if ln else "mean"][it], float)
    x_center = h.mean(0)
    if ln:
        x_scale = h.std(0, ddof=1)
    else:
        x_scale = np.sqrt(
            np.maximum(np.asarray(cache["second"][it], float).mean(0) - x_center * x_center, 0)
        )
    x_scale = np.where(x_scale > 1e-6, x_scale, 1.0)
    X = (h - x_center) / x_scale
    Xv = (
        None
        if valid is None
        else (np.asarray(cache["ln_mean" if ln else "mean"][iv], float) - x_center) / x_scale
    )
    ym, ys, sup, var, counts = target_scaling(train.labels, train.cancers, names)
    ci = np.array([names.index(c) for c in train.cancers])
    Y = (train.labels - ym[ci]) / ys[ci]
    if valid is not None:
        cv = np.array([names.index(c) for c in valid.cancers])
        Yv = (valid.labels - ym[cv]) / ys[cv]
    else:
        Yv = None
        cv = None
    B = np.zeros((len(names), D, T))
    b = np.zeros((len(names), T))
    chosen = {}
    curves = []
    pv = None if valid is None else np.zeros_like(valid.labels, dtype=float)
    for k, c in enumerate(names):
        mask = ci == k
        grid = alpha_grid if forced_alphas is None else [forced_alphas[c]]
        candidates = ridge_grid(X[mask], Y[mask], grid)
        best = None
        for alpha, (coef, intercept) in candidates.items():
            if valid is not None:
                vv = cv == k
                pp = Xv[vv] @ coef + intercept
                sc = selection(Yv[vv], pp, [c] * int(vv.sum()), {c: var[k]}, min_n=10)
                criterion = (sc["pearson"], -sc["mse"], -float(alpha))
                curves.append(
                    {"cancer": c, "alpha": alpha, "pearson": sc["pearson"], "mse": sc["mse"]}
                )
            else:
                criterion = (0, 0, -float(alpha))
            if best is None or criterion > best[0]:
                best = (criterion, alpha, coef, intercept)
        _, alpha, B[k], b[k] = best
        chosen[c] = alpha
        if valid is not None:
            pv[cv == k] = Xv[cv == k] @ B[k] + b[k]
    arrays = {
        "x_center": x_center,
        "x_scale": x_scale,
        "y_center": ym,
        "y_scale": ys,
        "base_coef": B,
        "base_bias": b,
        "supported": sup,
        "variable": var,
        "train_counts": counts,
    }
    report = {
        "alphas": chosen,
        "alpha_selection_trace": curves,
        "input_transform": "MEAN_OF_PATCH_LAYERNORM" if ln else "FULL_DIM_FIXED_AFFINE",
        "training_n": len(train.ids),
        "validation_n": None if valid is None else len(valid.ids),
        "validation": None
        if valid is None
        else selection(Yv, pv, valid.cancers, {c: var[k] for k, c in enumerate(names)}),
    }
    return arrays, report
