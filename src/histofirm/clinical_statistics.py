"""Finite post-fit analyses for one frozen, post-hoc-selected risk score.
No predictor fitting, no score selection, no cut-point optimization.
"""

from __future__ import annotations
import math
import numpy as np
from scipy import optimize, special, stats


class Hold(ValueError):
    pass


class IdentityError(ValueError):
    pass


def p_adjust(values, method="BH", family=None):
    v = np.asarray([np.nan if x is None else x for x in values], float)
    m = len(v) if family is None else int(family)
    if m < len(v):
        raise ValueError("Family cannot be smaller than planned tests")
    p = np.r_[np.where(np.isfinite(v), np.clip(v, 0, 1), 1.0), np.ones(m - len(v))]
    order = np.argsort(p, kind="mergesort")
    q = np.empty(m)
    if method == "BH":
        z = p[order] * m / np.arange(1, m + 1)
        z = np.minimum.accumulate(z[::-1])[::-1]
    elif method == "Holm":
        z = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    else:
        raise ValueError("Unknown adjustment")
    q[order] = np.minimum(z, 1)
    return [float(q[i]) if np.isfinite(v[i]) else None for i in range(len(v))]


def stable_seed(base, key):
    import hashlib

    return int.from_bytes(hashlib.sha256((str(base) + "|" + key).encode()).digest()[:8], "big")


def colbasis(x, tol=1e-10):
    x = np.asarray(x, float)
    if x.shape[1] == 0:
        return np.empty((len(x), 0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    return u[:, :rank]


class C6Transform:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, c):
        c = np.asarray(c, float)
        if c.shape[1] != 6 or np.isinf(c).any():
            raise Hold("C6_SHAPE_OR_INFINITY_HOLD")
        self.missing = np.mean(~np.isfinite(c), axis=0)
        if np.any(self.missing > self.cfg["maximum_training_missing_fraction"]):
            raise Hold("C6_TRAIN_MISSING_EXCEEDS_30_PERCENT_NO_COLUMN_HUNT")
        self.median = np.nanmedian(c, axis=0)
        a = np.where(np.isfinite(c), c, self.median)
        self.mean = a.mean(0)
        self.sd = a.std(0, ddof=0)
        self.keep = self.sd > 1e-12
        if not np.any(self.keep):
            raise Hold("C6_ALL_CONSTANT_ADJUSTMENT_UNAVAILABLE")
        z = (a[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        rank = int(np.sum(s > self.cfg["rank_tolerance"] * max(1.0, s[0])))
        if rank == 0:
            raise Hold("C6_NO_TRAINING_COLUMN_SPACE")
        self.basis = vt[:rank].T
        self.rank = rank
        return self

    def transform(self, c):
        c = np.asarray(c, float)
        if c.shape[1] != 6 or np.isinf(c).any():
            raise Hold("C6_EXTERNAL_INFINITY_OR_SHAPE")
        a = np.where(np.isfinite(c), c, self.median)
        z = (a[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        return z @ self.basis

    def metadata(self, c):
        c = np.asarray(c, float)
        a = np.where(np.isfinite(c), c, self.median)
        z = (a[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        residual = z - z @ self.basis @ self.basis.T
        return {
            "n": len(c),
            "training_median": self.median,
            "training_mean": self.mean,
            "training_sd": self.sd,
            "training_retained_columns": np.where(self.keep)[0],
            "training_columnspace_rank": self.rank,
            "missing_counts": np.sum(~np.isfinite(c), axis=0),
            "outside_training_columnspace_n": int(np.sum(np.linalg.norm(residual, axis=1) > 1e-7)),
            "training_constant_column_changed_n": [
                int(np.sum(np.abs(a[:, j] - self.mean[j]) > 1e-12)) for j in np.where(~self.keep)[0]
            ],
            "qualification": "ARCHIVED_C6_NUMERIC_SPAN_NOT_NAMED_CLINICAL_INDEPENDENCE",
        }


def cohort_nuisance(c):
    c = np.asarray(c, float)
    return colbasis(c - c.mean(0, keepdims=True))


def cox_derivatives(beta, X, t, e, need_hessian=True):
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    if int(e.sum()) == 0:
        raise Hold("COX_NO_EVENTS")
    eta = X @ beta
    if not np.isfinite(eta).all():
        raise FloatingPointError("NONFINITE_COX_PREDICTOR")
    event_times, ds = np.unique(t[e == 1], return_counts=True)
    mask = t[None, :] >= event_times[:, None]
    a = np.where(mask, eta[None, :], -np.inf)
    lden = special.logsumexp(a, axis=1)
    q = np.exp(a - lden[:, None])
    mu = q @ X
    loss = (np.dot(ds, lden) - np.dot(e, eta)) / e.sum()
    gradient = (ds @ mu - e @ X) / e.sum()
    if not need_hessian:
        return float(loss), gradient
    second = X.T @ ((ds @ q)[:, None] * X) - mu.T @ (ds[:, None] * mu)
    return float(loss), gradient, second / e.sum()


def cox_fit(X, t, e, cfg, callback=None):
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    n, p = X.shape
    events = int(e.sum())
    if n < cfg["min_n"] or events < max(
        cfg["min_events"], cfg["min_events_per_parameter"] * max(p, 1)
    ):
        raise Hold("COX_N_EVENT_SUPPORT_HOLD")
    if not np.isfinite(X).all():
        raise Hold("NONFINITE_COX_DESIGN")
    if p == 0:
        loss, g, H = cox_derivatives(np.zeros(0), X, t, e)
        return {
            "beta": np.zeros(0),
            "cov": np.empty((0, 0)),
            "information": H,
            "loss": loss,
            "gradient_inf": 0.0,
            "iterations": 0,
            "optimizer_calls": 0,
            "state": "NO_NUISANCE_COEFFICIENTS",
            "n": n,
            "events": events,
        }
    if np.linalg.matrix_rank(X, tol=cfg["rank_tolerance"]) < p:
        raise Hold("COX_DESIGN_RANK_HOLD")
    trajectory = []
    start = np.zeros(p)

    def fun(b):
        return cox_derivatives(b, X, t, e, False)

    def cb(b):
        loss, g = fun(b)
        r = {
            "iteration": len(trajectory) + 1,
            "loss": loss,
            "gradient_inf": float(np.max(np.abs(g))),
        }
        trajectory.append(r)
        if callback:
            callback(r, b.copy())

    fit = optimize.minimize(
        fun,
        start,
        jac=True,
        method="L-BFGS-B",
        callback=cb,
        options={"maxiter": cfg["max_iterations"], "gtol": 1e-8, "ftol": 0.0, "maxls": 50},
    )
    loss, g, H = cox_derivatives(fit.x, X, t, e)
    info = H * events
    evals = np.linalg.eigvalsh(info)
    gin = float(np.max(np.abs(g)))
    condition = float(np.linalg.cond(info))
    diagnostic = {
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "gradient_inf": gin,
        "information_min_eigenvalue": float(evals.min()),
        "information_condition": condition,
        "iterations": int(fit.nit),
    }
    if (
        not np.isfinite(loss)
        or gin > cfg["gradient_tolerance"]
        or evals.min() < 1e-8
        or condition > 1e10
    ):
        raise Hold("COX_NUMERIC_OR_SEPARATION_HOLD::" + str(diagnostic))
    cov = np.linalg.inv(info)
    if not np.isfinite(cov).all():
        raise Hold("NONFINITE_COX_COVARIANCE")
    return {
        **diagnostic,
        "beta": fit.x,
        "cov": cov,
        "information": info,
        "loss": loss,
        "optimizer_calls": 1,
        "n": n,
        "events": events,
        "state": "POSTFIT_ASSOCIATION_NOT_DEPLOYMENT_MODEL",
        "trajectory": trajectory,
    }


def cox_effect(fit, null_loss):
    b = float(fit["beta"][0])
    se = float(np.sqrt(fit["cov"][0, 0]))
    z = b / se
    limits = np.array([b - 1.959963984540054 * se, b + 1.959963984540054 * se])
    if np.max(np.abs(limits)) > 700:
        raise Hold("COX_HR_OVERFLOW_HOLD")
    return {
        "logHR": b,
        "SE": se,
        "HR": math.exp(b),
        "CI95": np.exp(limits).tolist(),
        "Wald_p": float(2 * stats.norm.sf(abs(z))),
        "LR_p": float(stats.chi2.sf(max(0.0, 2 * fit["events"] * (null_loss - fit["loss"])), 1)),
        "n": fit["n"],
        "events": fit["events"],
        "post_selection": True,
        "qualification": "POSTFIT_ASSOCIATION_NOT_DEPLOYMENT_MODEL",
    }


def ph_time_score(beta, X, t, e, index=0):
    """Efficient score test for z_i * standardized log(event time), Breslow ties.
    Nuisance constant-Cox coefficients are profiled via a Schur complement.
    This is NOT an exact reimplementation of R's default KM-transform cox.zph.
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    times, d = np.unique(t[e == 1], return_counts=True)
    if len(times) < 8 or e.sum() < 30:
        raise Hold("PH_EVENT_TIME_SUPPORT_HOLD")
    l = np.log(times)
    m = np.average(l, weights=d)
    sd = np.sqrt(np.average((l - m) ** 2, weights=d))
    if sd < 1e-12:
        raise Hold("PH_CONSTANT_TRANSFORMED_TIME")
    h = (l - m) / sd
    eta = X @ beta
    p = X.shape[1]
    Ibb = np.zeros((p, p))
    Ibg = np.zeros(p)
    Igg = 0.0
    U = 0.0
    Ub = np.zeros(p)
    rows = []
    for u, du, hu in zip(times, d, h):
        at = t >= u
        dead = (t == u) & (e == 1)
        xx = X[at]
        ww = special.softmax(eta[at])
        mu = ww @ xx
        cv = xx.T @ (ww[:, None] * xx) - np.outer(mu, mu)
        residual = X[dead].sum(0) - du * mu
        U += hu * residual[index]
        Ub += residual
        Ibb += du * cv
        Ibg += du * hu * cv[:, index]
        Igg += du * hu * hu * cv[index, index]
        rows.append(
            {
                "archived_event_time": float(u),
                "event_count": int(du),
                "log_time_standardized": float(hu),
                "schoenfeld_sum_tested": float(residual[index]),
            }
        )
    if np.linalg.cond(Ibb) > 1e10:
        raise Hold("PH_INFORMATION_HOLD")
    effU = U - Ibg @ np.linalg.solve(Ibb, Ub)
    effI = Igg - Ibg @ np.linalg.solve(Ibb, Ibg)
    if not np.isfinite(effI) or effI <= 1e-10:
        raise Hold("PH_EFFICIENT_VARIANCE_HOLD")
    statistic = effU * effU / effI
    # Aggregate only: event-by-event rows remain server-private.
    edges = np.quantile(h, np.linspace(0, 1, 6))
    bins = np.searchsorted(edges[1:-1], h, side="right")
    aggregate = []
    for j in range(5):
        rr = [r for k, r in enumerate(rows) if bins[k] == j]
        if rr:
            aggregate.append(
                {
                    "time_bin": j,
                    "event_count": sum(x["event_count"] for x in rr),
                    "mean_transformed_time": float(
                        np.mean([x["log_time_standardized"] for x in rr])
                    ),
                    "schoenfeld_sum": sum(x["schoenfeld_sum_tested"] for x in rr),
                }
            )
    return {
        "test": "BRESLOW_EFFICIENT_SCORE_TEST_LOGTIME_INTERACTION",
        "chi2": float(statistic),
        "df": 1,
        "p": float(stats.chi2.sf(statistic, 1)),
        "efficient_U": float(effU),
        "efficient_information": float(effI),
        "no_rejection_does_not_prove_PH": True,
        "private_event_rows": rows,
        "aggregate_bins": aggregate,
    }


def same_column_space(a, b, tol=1e-8):
    a = colbasis(a - np.mean(a, axis=0))
    b = colbasis(b - np.mean(b, axis=0))
    return a.shape[1] == b.shape[1] and np.linalg.norm(a - b @ (b.T @ a)) < tol


def hc3_group(y, group, C, cfg):
    """Coefficient-specific HC3 after VERIFIED zero-information nuisance saturation.
    No outcomes or p values enter sample selection. Original and informational
    coefficients, residuals and df must agree. Original n is always retained.
    """
    y = np.asarray(y, float)
    g = np.asarray(group, float)
    C = np.asarray(C, float)
    N = colbasis(np.column_stack([np.ones(len(y)), C]))
    r = g - N @ (N.T @ g)
    ss = float(r @ r)
    if ss <= 1e-10:
        raise Hold("GROUP_IN_C6_COLUMN_SPACE")
    a = r / ss
    Q = colbasis(np.column_stack([N, g]))
    df = len(y) - Q.shape[1]
    beta = float(a @ y)
    res = y - Q @ (Q.T @ y)
    hn = np.sum(N * N, axis=1)
    sat = hn >= 1 - 1e-10
    if np.any(sat):
        if np.max(np.abs(a[sat])) > 1e-8:
            raise Hold("SATURATED_ROW_NONZERO_GROUP_WEIGHT")
        keep = ~sat
        N2 = colbasis(np.column_stack([np.ones(keep.sum()), C[keep]]))
        r2 = g[keep] - N2 @ (N2.T @ g[keep])
        ss2 = float(r2 @ r2)
        if ss2 <= 1e-10:
            raise Hold("REDUCED_GROUP_INFORMATION_ZERO")
        a2 = r2 / ss2
        Q2 = colbasis(np.column_stack([N2, g[keep]]))
        res2 = y[keep] - Q2 @ (Q2.T @ y[keep])
        beta2 = float(a2 @ y[keep])
        df2 = int(keep.sum()) - Q2.shape[1]
        if (
            abs(beta2 - beta) > 1e-8 * max(1, abs(beta))
            or df2 != df
            or not np.allclose(res[keep], res2, atol=1e-8, rtol=1e-8)
        ):
            raise Hold("SATURATION_ALGEBRA_CHECK_FAILED")
        h = np.sum(Q2 * Q2, axis=1)
        aw = a2
        rw = res2
    else:
        keep = np.ones(len(y), bool)
        h = np.sum(Q * Q, axis=1)
        aw = a
        rw = res
    if min(np.sum(g[keep] == 0), np.sum(g[keep] == 1)) < cfg["minimum_group_n"]:
        raise Hold("INFORMATIONAL_GROUP_N_HOLD")
    if df < 10 or np.max(h) >= cfg["maximum_information_leverage"]:
        raise Hold("HC3_INFORMATIONAL_LEVERAGE_HOLD")
    variance = float(np.sum((aw * rw / (1 - h)) ** 2))
    if variance <= 1e-20:
        if abs(beta) < 1e-10:
            return {
                "beta": 0.0,
                "SE": 0.0,
                "CI95": [0.0, 0.0],
                "p": 1.0,
                "df": df,
                "saturated_n": int(sat.sum()),
                "information_n": int(keep.sum()),
                "state": "CONSTANT_NO_GROUP_DIFFERENCE",
            }
        raise Hold("ZERO_RESIDUAL_VARIANCE_NONZERO_EFFECT")
    se = math.sqrt(variance)
    critical = stats.t.ppf(0.975, df)
    return {
        "beta": beta,
        "SE": se,
        "CI95": [beta - critical * se, beta + critical * se],
        "p": float(2 * stats.t.sf(abs(beta / se), df)),
        "df": df,
        "saturated_n": int(sat.sum()),
        "information_n": int(keep.sum()),
        "maximum_information_leverage": float(np.max(h)),
        "state": "COEFFICIENT_SPECIFIC_HC3_SENSITIVITY",
    }


def reference_scale(a):
    a = np.asarray(a, float)
    finite = np.isfinite(a)
    n = finite.sum(0)
    mu = np.divide(np.where(finite, a, 0).sum(0), n, out=np.full(a.shape[1], np.nan), where=n > 0)
    dif = np.where(finite, a - mu, 0)
    sd = np.sqrt(np.divide((dif * dif).sum(0), n, out=np.full(a.shape[1], np.nan), where=n > 0))
    return mu, sd, n


def axis_analysis(matrix, group, targets, active, scale_sd, C, cfg):
    Y = np.asarray(matrix, float)
    g = np.asarray(group, bool)
    rows = []
    for j, target in enumerate(targets):
        y = Y[:, j]
        finite = np.isfinite(y)
        hi = finite & g
        lo = finite & ~g
        nh, nl = int(hi.sum()), int(lo.sum())
        row = {
            "target_index": j,
            "target_id": str(target),
            "planned_family_n": 1564,
            "n_original": len(y),
            "n_finite": int(finite.sum()),
            "n_high": nh,
            "n_low": nl,
            "nonfinite_n": int((~finite).sum()),
            "active": bool(active[j]),
            "logFC": None,
            "logFC_state": "NOT_DEFINED_FOR_SIGNED_NON_RATIO_FUNCTION_SCORES",
            "MW_p": None,
            "adjusted_p": None,
            "adjusted_beta": None,
            "adjusted_CI95": None,
            "source_scale_SD": float(scale_sd[j]) if np.isfinite(scale_sd[j]) else None,
        }
        if nh and nl:
            row.update(
                {
                    "median_high": float(np.median(y[hi])),
                    "median_low": float(np.median(y[lo])),
                    "median_difference_raw": float(np.median(y[hi]) - np.median(y[lo])),
                    "mean_high": float(y[hi].mean()),
                    "mean_low": float(y[lo].mean()),
                    "mean_difference_raw": float(y[hi].mean() - y[lo].mean()),
                }
            )
        if not active[j]:
            row["state"] = "ORIGINAL_INACTIVE_TARGET_HOLD"
            rows.append(row)
            continue
        if (
            min(nh, nl) < cfg["minimum_group_n"]
            or nh < cfg["minimum_finite_fraction"] * g.sum()
            or nl < cfg["minimum_finite_fraction"] * (~g).sum()
        ):
            row["state"] = "FUNCTION_FINITE_GROUP_SUPPORT_HOLD"
            rows.append(row)
            continue
        if not np.isfinite(scale_sd[j]) or scale_sd[j] <= 1e-12:
            row["state"] = "TCGA_REFERENCE_SCALE_HOLD"
            rows.append(row)
            continue
        row["standardized_mean_difference"] = row["mean_difference_raw"] / scale_sd[j]
        row["standardized_median_difference"] = row["median_difference_raw"] / scale_sd[j]
        if np.max(y[finite]) - np.min(y[finite]) <= 1e-12:
            U = nh * nl / 2
            p = 1.0
        else:
            test = stats.mannwhitneyu(
                y[hi], y[lo], alternative="two-sided", method="asymptotic", use_continuity=True
            )
            U, p = float(test.statistic), float(test.pvalue)
        row["MW_U_high"] = U
        row["MW_p"] = p
        row["rank_biserial_high_minus_low"] = 2 * U / (nh * nl) - 1
        row["state"] = "UNADJUSTED_FUNCTION_DIFFERENCE_ESTIMABLE"
        if C is not None:
            try:
                result = hc3_group(y[finite] / scale_sd[j], g[finite], C[finite], cfg)
                row.update(
                    {
                        "adjusted_state": result["state"],
                        "adjusted_beta": result["beta"],
                        "adjusted_SE": result["SE"],
                        "adjusted_CI95": result["CI95"],
                        "adjusted_p": result["p"],
                        "adjusted_df": result["df"],
                        "saturated_n": result["saturated_n"],
                        "information_n": result["information_n"],
                        "information_max_leverage": result.get("maximum_information_leverage"),
                    }
                )
            except Hold as ex:
                row["adjusted_state"] = str(ex)
        else:
            row["adjusted_state"] = "C6_DESIGN_DEPENDENCY_HOLD"
        rows.append(row)
    for field in ("MW", "adjusted"):
        q = p_adjust([r.get(field + "_p") for r in rows], "BH", 1564)
        for r, v in zip(rows, q):
            r[field + "_BH1564"] = v
    return rows


def resampled_mean_effect(matrix, group, sd, draws, seed):
    Y = np.asarray(matrix, float)
    g = np.asarray(group, bool)
    sd = np.asarray(sd, float)
    rng = np.random.default_rng(seed)
    out = np.full((draws, Y.shape[1]), np.nan)

    def block_average(idx):
        a = Y[idx]
        finite = np.isfinite(a)
        filled = np.where(finite, a, 0.0)
        counts = np.stack(
            [
                np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
                for _ in range(draws)
            ]
        ).astype(float)
        den = counts @ finite.astype(float)
        num = counts @ filled
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    high = block_average(np.where(g)[0])
    low = block_average(np.where(~g)[0])
    valid = np.isfinite(sd) & (sd > 1e-12)
    out[:, valid] = (high[:, valid] - low[:, valid]) / sd[valid]
    return out


def effect_concordance(a, b, zero=1e-8):
    a = np.asarray([np.nan if v is None else v for v in a], float)
    b = np.asarray([np.nan if v is None else v for v in b], float)
    good = np.isfinite(a) & np.isfinite(b)
    x, y = a[good], b[good]
    pos = (np.abs(x) > zero) & (np.abs(y) > zero)
    result = {
        "universe_n": len(a),
        "paired_finite_n": len(x),
        "missing_or_held_n": int((~good).sum()),
        "zero_in_either_n": int((~pos).sum()),
        "paired_nonzero_n": int(pos.sum()),
        "same_positive_n": int(np.sum((x > zero) & (y > zero))),
        "same_negative_n": int(np.sum((x < -zero) & (y < -zero))),
        "opposite_n": int(np.sum(pos & (np.sign(x) != np.sign(y)))),
        "agreement_fraction": float(np.mean(np.sign(x[pos]) == np.sign(y[pos])))
        if pos.any()
        else None,
        "Pearson": None,
        "Spearman": None,
        "p": None,
        "qualification": "DESCRIPTIVE_CORRELATED_TARGETS_NOT_BERNOULLI_INFERENCE",
    }
    if len(x) >= 10 and min(np.std(x), np.std(y)) > 1e-12:
        result["Pearson"] = float(np.corrcoef(x, y)[0, 1])
        result["Spearman"] = float(stats.spearmanr(x, y).statistic)
    return result


def paired_C_bootstrap(t, e, a, b, draws, seed):
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ii, jj = np.where((t[:, None] < t[None, :]) & (e[:, None] == 1))
    if len(ii) < 50:
        raise Hold("C_COMPARABLE_PAIR_SUPPORT_HOLD")

    def credit(r):
        d = r[ii] - r[jj]
        return (d > 0).astype(float) + 0.5 * (d == 0)

    ca, cb = credit(a), credit(b)
    difference = ca - cb
    rng = np.random.default_rng(seed)
    out = []
    strata = [np.where(e == q)[0] for q in (0, 1)]
    for i in range(draws):
        idx = np.concatenate([rng.choice(q, len(q), replace=True) for q in strata if len(q)])
        count = np.bincount(idx, minlength=len(t))
        w = count[ii] * count[jj]
        out.append(float(w @ difference / w.sum()) if w.sum() else np.nan)
    return np.asarray(out), float(difference.mean())
