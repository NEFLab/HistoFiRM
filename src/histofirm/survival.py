from __future__ import annotations

import math
import hashlib
import numpy as np
from scipy import linalg, optimize, special, stats

from .clinical_statistics import Hold


def stable_seed(base, tag):
    b = hashlib.sha256((str(base) + "|" + str(tag)).encode("utf-8")).digest()
    return int.from_bytes(b[:8], "big")


def comparable(t, e, folds=None):
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    ok = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    if folds is not None:
        f = np.asarray(folds)
        ok &= f[:, None] == f[None, :]
    return np.where(ok)


def concordance(t, e, risk, folds=None, counts=None):
    risk = np.asarray(risk, float)
    if risk.ndim == 1:
        risk = risk[:, None]
    if not np.all(np.isfinite(risk)):
        raise Hold("NONFINITE_PREDICTED_SCORE")
    i, j = comparable(t, e, folds)
    if len(i) == 0:
        return np.full(risk.shape[1], np.nan), 0.0
    w = np.ones(len(i)) if counts is None else counts[i] * counts[j]
    denominator = float(w.sum())
    if denominator <= 0:
        return np.full(risk.shape[1], np.nan), 0.0
    d = risk[i] - risk[j]
    credit = (d > 0).astype(float) + 0.5 * (d == 0)
    return (credit * w[:, None]).sum(axis=0) / denominator, denominator


def cox_loss(eta, t, e, hessian=False):
    eta = np.asarray(eta, float)
    t, e = np.asarray(t, float), np.asarray(e, int)
    events = int(e.sum())
    if events == 0 or not np.all(np.isfinite(eta)):
        raise Hold("COX_NO_EVENTS_OR_NONFINITE_RISK")
    times, counts = np.unique(t[e == 1], return_counts=True)
    mask = t[None, :] >= times[:, None]
    z = np.where(mask, eta[None, :], -np.inf)
    logden = special.logsumexp(z, axis=1)
    q = np.exp(z - logden[:, None])
    loss = (np.dot(counts, logden) - np.dot(e, eta)) / events
    mean = counts @ q
    grad = (mean - e) / events
    if not hessian:
        return float(loss), grad
    H = (np.diag(mean) - q.T @ (counts[:, None] * q)) / events
    return float(loss), grad, H


def rank_loss(eta, t, e):
    i, j = comparable(t, e)
    if len(i) == 0:
        raise Hold("RANK_NO_COMPARABLE_PAIRS")
    d = eta[i] - eta[j]
    loss = np.logaddexp(0.0, -d).mean()
    a = -special.expit(-d) / len(i)
    g = np.bincount(i, weights=a, minlength=len(eta))
    g -= np.bincount(j, weights=a, minlength=len(eta))
    return float(loss), g


def split_ids(ids, seed, k, events=None):
    ids = np.asarray(ids).astype(str)
    if len(set(ids)) != len(ids):
        raise Hold("DUPLICATE_ID_IN_SPLIT")
    groups = [np.arange(len(ids))]
    if events is not None:
        e = np.asarray(events, int)
        groups = [np.where(e == v)[0] for v in (0, 1)]
    out = np.full(len(ids), -1, int)
    for group in groups:
        order = sorted(
            group, key=lambda i: hashlib.sha256((str(seed) + "|" + ids[i]).encode("utf-8")).digest()
        )
        for rank, i in enumerate(order):
            out[i] = rank % k
    if np.any(out < 0):
        raise Hold("INCOMPLETE_SPLIT")
    return out


class Standard:
    def __init__(self, fraction=0.95):
        self.fraction = fraction

    def fit(self, x, allowed=None):
        x = np.asarray(x, float)
        allowed = np.ones(x.shape[1], bool) if allowed is None else np.asarray(allowed, bool)
        finite = np.isfinite(x)
        mask = allowed & (finite.mean(axis=0) >= self.fraction)
        cols = np.where(mask)[0]
        if len(cols) == 0:
            self.cols = np.array([], int)
            self.median = self.mean = self.sd = np.array([], float)
            return self
        a = x[:, cols]
        med = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
        a = np.where(np.isfinite(a), a, med)
        mu, sd = a.mean(axis=0), a.std(axis=0, ddof=0)
        keep = np.isfinite(sd) & (sd > 1e-12)
        self.cols, self.median = cols[keep], med[keep]
        self.mean, self.sd = mu[keep], sd[keep]
        return self

    def transform(self, x):
        a = np.asarray(x, float)[:, self.cols]
        a = np.where(np.isfinite(a), a, self.median)
        return (a - self.mean) / self.sd


class PCA:
    def __init__(self, components, fraction=0.95):
        self.components = components
        self.std = Standard(fraction)
        self.state = "NOT_FITTED"

    def fit(self, x, allowed=None):
        self.std.fit(x, allowed)
        z = self.std.transform(x)
        if min(z.shape, default=0) < self.components:
            self.state = "RANK_SUPPORT_HOLD"
            self.V = np.zeros((z.shape[1], 0))
            return self
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        tol = np.finfo(float).eps * max(z.shape) * s[0]
        if int(np.sum(s > tol)) < self.components:
            self.state = "RANK_SUPPORT_HOLD"
            self.V = np.zeros((z.shape[1], 0))
            return self
        v = vt[: self.components].copy()
        for row in v:
            pos = np.argmax(np.abs(row))
            if row[pos] < 0:
                row *= -1
        self.V = v.T
        self.state = "READY"
        return self

    def transform(self, x):
        return self.std.transform(x) @ self.V


class Prep:
    def __init__(self, cfg):
        self.cfg = cfg
        self.c = Standard(cfg["quality_fraction"])
        self.h = PCA(cfg["h_dim"], cfg["quality_fraction"])
        self.f = PCA(cfg["f_dim"], cfg["quality_fraction"])

    def fit(self, C, H, P, active):
        self.c.fit(C)
        self.h.fit(H)
        self.f.fit(P, active)
        return self

    def transform(self, C, H, P):
        return (self.c.transform(C), self.h.transform(H), self.f.transform(P))

    def require(self, family):
        if "H" in family and self.h.state != "READY":
            raise Hold("H_PCA_RANK_SUPPORT_HOLD")
        if "F" in family and self.f.state != "READY":
            raise Hold("F_PCA_RANK_SUPPORT_HOLD")


class Net:
    def __init__(self, family, dims, hidden=16):
        self.family, self.dims, self.hidden = family, tuple(dims), hidden
        self.layout = []
        self.n = 0

        def add(name, shape, penalized=True):
            start = self.n
            self.n += int(np.prod(shape))
            self.layout.append((name, tuple(shape), start, self.n, penalized))

        if "C" in family:
            add("C", (dims[0],))
        for name, dim in (("H", dims[1]), ("F", dims[2])):
            if name in family:
                add(name + "_linear", (dim,))
                add(name + "_W", (dim, hidden))
                add(name + "_b", (hidden,), False)
                add(name + "_v", (hidden,))
        self.penalty = np.zeros(self.n)
        for _, _, a, b, yes in self.layout:
            self.penalty[a:b] = float(yes)

    def unpack(self, theta):
        return {name: theta[a:b].reshape(shape) for name, shape, a, b, _ in self.layout}

    def initialize(self, seed):
        w = np.zeros(self.n, float)
        p = self.unpack(w)
        for branch, dim in (("H", self.dims[1]), ("F", self.dims[2])):
            if branch not in self.family:
                continue
            rng = np.random.default_rng(stable_seed(seed, branch))
            p[branch + "_W"][:] = rng.uniform(
                -math.sqrt(6 / (dim + self.hidden)),
                math.sqrt(6 / (dim + self.hidden)),
                size=(dim, self.hidden),
            )
            p[branch + "_v"][:] = rng.uniform(
                -math.sqrt(6 / (self.hidden + 1)),
                math.sqrt(6 / (self.hidden + 1)),
                size=self.hidden,
            )
        return w

    def predict(self, theta, X):
        C, H, F = X
        p = self.unpack(theta)
        n = len(C)
        parts = {"C": np.zeros(n), "H": np.zeros(n), "F": np.zeros(n)}
        if "C" in self.family:
            parts["C"] = C @ p["C"]
        fref = np.zeros(n)
        for branch, x in (("H", H), ("F", F)):
            if branch in self.family:
                a = np.tanh(x @ p[branch + "_W"] + p[branch + "_b"])
                parts[branch] = x @ p[branch + "_linear"] + a @ p[branch + "_v"]
                if branch == "F":
                    fref[:] = np.tanh(p["F_b"]) @ p["F_v"]
        eta = parts["C"] + parts["H"] + parts["F"]
        return eta, parts["F"] - fref

    def objective(self, theta, X, t, e, lam, rank_weight=0.0):
        C, H, F = X
        p = self.unpack(theta)
        eta, _ = self.predict(theta, X)
        loss, g = cox_loss(eta, t, e)
        if rank_weight:
            lr, gr = rank_loss(eta, t, e)
            loss += rank_weight * lr
            g += rank_weight * gr
        grads = {}
        if "C" in self.family:
            grads["C"] = C.T @ g
        for branch, x in (("H", H), ("F", F)):
            if branch not in self.family:
                continue
            a = np.tanh(x @ p[branch + "_W"] + p[branch + "_b"])
            d = (g[:, None] * p[branch + "_v"]) * (1 - a * a)
            grads[branch + "_linear"] = x.T @ g
            grads[branch + "_W"] = x.T @ d
            grads[branch + "_b"] = d.sum(axis=0)
            grads[branch + "_v"] = a.T @ g
        out = np.zeros_like(theta)
        for name, shape, start, end, _ in self.layout:
            out[start:end] = grads[name].ravel()
        loss += 0.5 * lam * np.dot(self.penalty * theta, theta)
        out += lam * self.penalty * theta
        return float(loss), out


def baseline(eta, t, e):
    eta = np.asarray(eta, float)
    if not np.all(np.isfinite(eta)):
        raise Hold("NONFINITE_RISK_FOR_BASELINE")
    if int(np.sum(e)) == 0:
        raise Hold("BASELINE_NO_EVENTS")
    center = float(eta.mean())
    times, d = np.unique(np.asarray(t)[np.asarray(e) == 1], return_counts=True)
    inc = np.array(
        [
            math.log(int(n)) - special.logsumexp(eta[np.asarray(t) >= u] - center)
            for u, n in zip(times, d)
        ]
    )
    logcum = np.logaddexp.accumulate(inc)
    return {"times": times, "logcum": logcum, "center": center}


def log_survival(base, eta, times):
    times = np.asarray(times, float)
    j = np.searchsorted(base["times"], times, side="right") - 1
    h = np.full(len(times), -np.inf)
    good = j >= 0
    h[good] = base["logcum"][j[good]]
    loghaz = np.asarray(eta)[:, None] - base["center"] + h[None, :]
    with np.errstate(over="ignore", under="ignore"):
        return -np.exp(loghaz)


def probabilities(logS):
    return -np.expm1(np.asarray(logS))


def ensemble_logS(values):
    a = np.stack(values, axis=0)
    if a.shape[0] != 20:
        raise Hold("ENSEMBLE_REQUIRES_ALL_20_SEEDS")
    return special.logsumexp(a, axis=0) - math.log(20)


def km(t, event):
    t, event = np.asarray(t, float), np.asarray(event, int)
    rows = [
        {
            "time": 0.0,
            "survival": 1.0,
            "lower": 1.0,
            "upper": 1.0,
            "at_risk": len(t),
            "events": 0,
            "censored": 0,
        }
    ]
    s, g = 1.0, 0.0
    for u in np.unique(t):
        n = int(np.sum(t >= u))
        d = int(np.sum((t == u) & (event == 1)))
        c = int(np.sum((t == u) & (event == 0)))
        s *= 1 - d / n
        if d and n > d:
            g += d / (n * (n - d))
        if s <= 0:
            lo = hi = 0.0
        elif s >= 1:
            lo = hi = 1.0
        else:
            se = math.sqrt(g) / abs(math.log(s))
            z = math.log(-math.log(s))
            lo = math.exp(-math.exp(z + 1.95996398454 * se))
            hi = math.exp(-math.exp(z - 1.95996398454 * se))
        rows.append(
            {
                "time": float(u),
                "survival": s,
                "lower": lo,
                "upper": hi,
                "at_risk": n,
                "events": d,
                "censored": c,
            }
        )
    return rows


def km_at(curve, t, left=False):
    tt = np.array([r["time"] for r in curve])
    ss = np.array([r["survival"] for r in curve])
    idx = np.searchsorted(tt, t, side="left" if left else "right") - 1
    return np.where(idx >= 0, ss[np.maximum(idx, 0)], 1.0)


def lr_stat(t, e, group):
    t, e, group = np.asarray(t), np.asarray(e), np.asarray(group, bool)
    if min(int(group.sum()), int((~group).sum())) < 2:
        return {"state": "GROUP_SUPPORT_HOLD", "T": np.nan, "p": np.nan}
    U, V = 0.0, 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u
        death = (t == u) & (e == 1)
        n, n1 = int(at.sum()), int(np.sum(at & group))
        d, d1 = int(death.sum()), int(np.sum(death & group))
        U += d1 - d * n1 / n
        if n > 1:
            V += n1 * (n - n1) * d * (n - d) / (n * n * (n - 1))
    if V <= 1e-14:
        return {"state": "LOGRANK_VARIANCE_HOLD", "T": np.nan, "p": np.nan}
    T = U * U / V
    return {"state": "ESTIMABLE", "T": T, "U": U, "V": V, "p": float(stats.chi2.sf(T, 1))}


def cut_scan(t, e, score, cfg):
    t, e, score = np.asarray(t), np.asarray(e), np.asarray(score, float)
    n = len(t)
    if not np.all(np.isfinite(score)):
        raise Hold("NONFINITE_CUTPOINT_SCORE")
    order = np.argsort(score, kind="mergesort")
    x = score[order]
    lower = max(cfg["minimum_group_n"], int(math.ceil(cfg["minimum_fraction"] * n)))
    positions = np.arange(1, n)
    keep = (positions >= lower) & ((n - positions) >= lower)
    keep &= x[:-1] < x[1:]
    ec = np.cumsum(e[order])
    keep &= ec[:-1] >= cfg["minimum_group_events"]
    keep &= (ec[-1] - ec[:-1]) >= cfg["minimum_group_events"]
    pos = positions[keep]
    if len(pos) == 0:
        return {"state": "NO_ADMISSIBLE_CUT", "rows": [], "best": None}
    times = np.unique(t[e == 1])
    R = (t[None, :] >= times[:, None]).astype(float)
    D = ((t[None, :] == times[:, None]) & (e[None, :] == 1)).astype(float)
    Y, dd = R.sum(axis=1), D.sum(axis=1)
    yh = Y[:, None] - np.cumsum(R[:, order], axis=1)[:, pos - 1]
    dh = dd[:, None] - np.cumsum(D[:, order], axis=1)[:, pos - 1]
    U = (dh - dd[:, None] * yh / Y[:, None]).sum(axis=0)
    den = np.maximum(Y - 1, 1)
    V = (
        yh * (Y[:, None] - yh) * dd[:, None] * (Y - dd)[:, None] / (Y[:, None] ** 2 * den[:, None])
    ).sum(axis=0)
    rows = []
    for z, k in enumerate(pos):
        if V[z] <= 1e-14:
            continue
        cut = float(x[k - 1] + (x[k] - x[k - 1]) / 2)
        if not x[k - 1] <= cut < x[k]:
            cut = float(x[k - 1])
        T = float(U[z] * U[z] / V[z])
        rows.append(
            {
                "position": int(k),
                "cut": cut,
                "n_low": int(k),
                "n_high": int(n - k),
                "events_low": int(ec[k - 1]),
                "events_high": int(ec[-1] - ec[k - 1]),
                "T": T,
                "ordinary_p": float(stats.chi2.sf(T, 1)),
                "direction_U": float(U[z]),
            }
        )
    if not rows:
        return {"state": "NO_ESTIMABLE_CUT", "rows": [], "best": None}
    maximum = max(r["T"] for r in rows)
    ties = [r for r in rows if abs(r["T"] - maximum) <= 1e-12]
    best = min(ties, key=lambda r: (abs(r["position"] - n / 2), r["cut"]))
    return {"state": "ESTIMABLE", "rows": rows, "best": best}


def cut_crossfit(t, e, score, ids, seed, cfg):
    fold = split_ids(ids, seed, 5)
    groups = np.zeros(len(t), bool)
    cuts = []
    for f in range(5):
        tr, va = fold != f, fold == f
        scan = cut_scan(np.asarray(t)[tr], np.asarray(e)[tr], np.asarray(score)[tr], cfg)
        if scan["best"] is None:
            return {"state": "CROSSFIT_CUT_HOLD", "T": np.nan, "groups": None}
        cut = scan["best"]["cut"]
        groups[va] = np.asarray(score)[va] > cut
        cuts.append({"fold": f, "cut": cut, "train_n": int(tr.sum()), "test_n": int(va.sum())})
    result = lr_stat(t, e, groups)
    return {**result, "groups": groups, "folds": fold, "cuts": cuts}


def holm(values):
    p = np.array([1.0 if v is None or not np.isfinite(v) else float(v) for v in values])
    order = np.argsort(p, kind="mergesort")
    q = np.empty(len(p))
    last = 0.0
    for rank, j in enumerate(order):
        last = max(last, min(1.0, (len(p) - rank) * p[j]))
        q[j] = last
    return [None if v is None or not np.isfinite(v) else float(q[i]) for i, v in enumerate(values)]


def bootstrap_indices(rng, event, folds=None):
    event = np.asarray(event)
    folds = np.zeros(len(event), int) if folds is None else np.asarray(folds)
    pieces = []
    for f in np.unique(folds):
        for v in (0, 1):
            q = np.where((folds == f) & (event == v))[0]
            if len(q):
                pieces.append(rng.choice(q, size=len(q), replace=True))
    return np.concatenate(pieces)


def interval(values, alpha=0.05, minimum_fraction=0.8):
    a = np.asarray(values, float)
    finite = a[np.isfinite(a)]
    if len(finite) < math.ceil(minimum_fraction * len(a)):
        return None
    return np.quantile(finite, [alpha / 2, 1 - alpha / 2]).tolist()


def time_metric(t, e, risk, survival, horizon, censor_curve, cfg):
    t, e = np.asarray(t), np.asarray(e)
    risk, survival = np.asarray(risk), np.asarray(survival)
    case = (t <= horizon) & (e == 1)
    control = t > horizon
    nc, nn = int(case.sum()), int(control.sum())
    if nc < cfg["time_minimum_cases"] or nn < cfg["time_minimum_controls"]:
        return {"state": "TIME_SUPPORT_HOLD", "cases": nc, "controls": nn}
    gc = km_at(censor_curve, t[case], left=True)
    gh = float(km_at(censor_curve, horizon))
    if gh < cfg["censoring_minimum"] or np.any(gc < cfg["censoring_minimum"]):
        return {"state": "CENSORING_SUPPORT_HOLD", "cases": nc, "controls": nn}
    wc = 1 / gc
    wn = np.full(nn, 1 / gh)
    d = risk[case, None] - risk[None, control]
    credit = (d > 0).astype(float) + 0.5 * (d == 0)
    auc = float(np.sum(credit * wc[:, None] * wn[None, :]) / (wc.sum() * wn.sum()))
    brier = float(
        (np.sum(survival[case] ** 2 / gc) + np.sum((1 - survival[control]) ** 2 / gh)) / len(t)
    )
    labels = np.r_[np.ones(nc), np.zeros(nn)]
    scores = np.r_[risk[case], risk[control]]
    weights = np.r_[wc, wn]
    order = np.argsort(-scores, kind="mergesort")
    labels, scores, weights = labels[order], scores[order], weights[order]
    hit = np.cumsum(weights * labels)
    false = np.cumsum(weights * (1 - labels))
    end = np.r_[scores[:-1] != scores[1:], True]
    roc = [{"fpr": 0.0, "tpr": 0.0}]
    roc += [
        {"fpr": float(fp / wn.sum()), "tpr": float(tp / wc.sum())}
        for fp, tp in zip(false[end], hit[end])
    ]
    return {
        "state": "ESTIMABLE",
        "auc": auc,
        "brier": brier,
        "cases": nc,
        "controls": nn,
        "G_h": gh,
        "roc": roc,
    }


def calibration(t, e, probability, train_probability, horizon):
    edges = np.unique(np.quantile(train_probability, [0, 0.2, 0.4, 0.6, 0.8, 1]))
    if len(edges) < 2:
        return []
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.searchsorted(edges[1:-1], probability, side="right")
    out = []
    for b in range(len(edges) - 1):
        q = bins == b
        if int(q.sum()) < 10:
            out.append({"bin": b, "n": int(q.sum()), "state": "BIN_SUPPORT_HOLD"})
            continue
        curve = km(np.asarray(t)[q], np.asarray(e)[q])
        s = float(km_at(curve, horizon))
        out.append(
            {
                "bin": b,
                "n": int(q.sum()),
                "state": "DESCRIPTIVE",
                "predicted": float(np.mean(probability[q])),
                "observed_KM_event_probability": 1 - s,
            }
        )
    return out


def association(t, e, score, nuisance, train_sd):
    t, e, score = np.asarray(t), np.asarray(e), np.asarray(score, float)
    C = np.asarray(nuisance, float)
    if not np.isfinite(train_sd) or train_sd <= 1e-12:
        return {"state": "CONSTANT_OR_NOT_APPLICABLE", "optimizer_calls": 0}
    x = (score - score.mean()) / train_sd
    C = C - C.mean(axis=0)
    if C.shape[1]:
        Q, R, piv = linalg.qr(C, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        rank = int(np.sum(diag > (diag.max(initial=0) * 1e-10)))
        C = C[:, piv[:rank]]
    if C.shape[1]:
        residual = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        if np.linalg.norm(residual) <= 1e-9 * max(1.0, np.linalg.norm(x)):
            return {"state": "SCORE_IN_NUISANCE_SPACE", "optimizer_calls": 0}
    X = np.column_stack([x, C])
    calls = 0

    def fit(A):
        nonlocal calls
        if A.shape[1] == 0:
            loss, _ = cox_loss(np.zeros(len(t)), t, e)
            return np.zeros(0), loss, np.zeros((0, 0)), 0.0, True
        calls += 1

        def fun(w):
            loss, g = cox_loss(A @ w, t, e)
            return loss, A.T @ g

        r = optimize.minimize(
            fun,
            np.zeros(A.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 0.0, "gtol": 1e-8},
        )
        loss, g, H = cox_loss(A @ r.x, t, e, True)
        info = A.T @ H @ A * int(e.sum())
        eig = np.linalg.eigvalsh(info)
        grad = float(np.max(np.abs(A.T @ g)))
        ok = (
            grad <= 1e-5
            and np.all(np.isfinite(info))
            and eig.min(initial=np.inf) > 1e-8
            and np.linalg.cond(info) < 1e10
        )
        return r.x, loss, info, grad, ok

    try:
        w, lf, info, grad, ok = fit(X)
        _, ln, _, gn, okn = fit(C)
        if not ok or not okn:
            return {
                "state": "ASSOCIATION_NUMERIC_OR_SEPARATION_HOLD",
                "optimizer_calls": calls,
                "gradient": grad,
                "null_gradient": gn,
            }
        se = float(np.sqrt(np.linalg.inv(info)[0, 0]))
        beta = float(w[0])
        ci = np.exp([beta - 1.95996398454 * se, beta + 1.95996398454 * se])
        return {
            "state": "POSTFIT_ASSOCIATION_NOT_DEPLOYMENT_MODEL",
            "n": len(t),
            "events": int(e.sum()),
            "logHR": beta,
            "SE": se,
            "HR": float(np.exp(beta)),
            "CI95": ci.tolist(),
            "wald_p": float(2 * stats.norm.sf(abs(beta / se))),
            "LR_p": float(stats.chi2.sf(max(0, 2 * int(e.sum()) * (ln - lf)), 1)),
            "optimizer_calls": calls,
            "gradient": grad,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return {"state": "ASSOCIATION_NUMERIC_HOLD", "optimizer_calls": calls}


def fidelity(p, r):
    p, r = np.asarray(p, float), np.asarray(r, float)
    good = np.isfinite(p) & np.isfinite(r)
    p, r = p[good], r[good]
    if len(p) < 30 or min(p.std(), r.std()) <= 1e-12:
        return {"state": "MOLECULAR_SUPPORT_HOLD", "n": len(p)}
    sp, sr = p.std(), r.std()
    corr = float(np.corrcoef(p, r)[0, 1])
    mse = float(np.mean((p - r) ** 2))
    cmse = float(np.mean(((p - p.mean()) - (r - r.mean())) ** 2))
    return {
        "state": "DESCRIPTIVE_P_R_NOT_R_R",
        "n": len(p),
        "Pearson": corr,
        "Spearman": float(stats.spearmanr(p, r).statistic),
        "RMSE": math.sqrt(mse),
        "centered_NRMSE": math.sqrt(cmse) / sr,
        "sd_P": sp,
        "sd_R": sr,
        "SD_ratio": sp / sr,
        "mean_difference": float(p.mean() - r.mean()),
        "CCC": float(2 * corr * sp * sr / (sp * sp + sr * sr + (p.mean() - r.mean()) ** 2)),
        "identity_error": abs(cmse - (sp * sp + sr * sr - 2 * corr * sp * sr)),
        "new_p_value": None,
    }
