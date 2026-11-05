"""Assay-supported spatial rank-score proxies and per-section evaluation."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def score(a, genes, sets, min_fraction=0.8, min_genes=5):
    lookup = {x: i for i, x in enumerate(genes)}
    G = len(genes)
    indices = []
    cov = []
    for x in sets:
        mem = set(x["genes"])
        ix = np.array([lookup[g] for g in sorted(mem) if g in lookup], int)
        frac = len(ix) / len(mem) if mem else 0.0
        weighted = str(x["target_id"]).startswith("PROGENY::")
        ok = len(ix) >= min_genes and frac >= min_fraction and len(ix) < G and not weighted
        reason = (
            "SUPPORTED_PROXY"
            if ok
            else "WEIGHTED_FOOTPRINT_NOT_REGISTERED"
            if weighted
            else "MISSING_ORIGINAL_GENESET"
            if not mem
            else "ASSAY_COVERAGE_INSUFFICIENT"
        )
        cov.append(
            {
                "target": x["target_id"],
                "declared_genes": len(mem),
                "assayed_genes": len(ix),
                "coverage": frac,
                "state": reason,
            }
        )
        indices.append(ix if ok else None)
    out = np.full((a.shape[1], len(sets)), np.nan)
    supported = np.array([j for j, ix in enumerate(indices) if ix is not None], int)
    if not len(supported):
        return out, cov
    rr = []
    cc = []
    sizes = []
    for k, j in enumerate(supported):
        rr.extend([k] * len(indices[j]))
        cc.extend(indices[j])
        sizes.append(len(indices[j]))
    membership = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(supported), G))
    sizes = np.asarray(sizes, float)
    for st in range(0, a.shape[1], 64):
        v = a[:, st : st + 64].toarray().T
        r = rankdata(v, method="average", axis=1)
        hit = (membership @ (r**1.25).T) / (membership @ (r**0.25).T)
        miss = (r.sum(axis=1)[None, :] - membership @ r.T) / (G - sizes[:, None])
        scores = ((hit - miss) / G).T
        scores[v.sum(axis=1) <= 0] = np.nan
        out[np.ix_(np.arange(st, st + len(v)), supported)] = scores
    return out, cov


def rho(y, p, rank=False):
    ok = np.isfinite(y) & np.isfinite(p)
    y = np.asarray(y)[ok]
    p = np.asarray(p)[ok]
    if len(y) < 10 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    if rank:
        y = rankdata(y)
        p = rankdata(p)
    return float(np.corrcoef(y, p)[0, 1])


def evaluate(truth, prediction, target_ids):
    if truth.shape != prediction.shape or truth.shape[1] != len(target_ids):
        raise ValueError("Spatial axes differ.")
    return pd.DataFrame(
        [
            {
                "target": t,
                "pearson": rho(truth[:, j], prediction[:, j]),
                "spearman": rho(truth[:, j], prediction[:, j], True),
                "units": int((np.isfinite(truth[:, j]) & np.isfinite(prediction[:, j])).sum()),
            }
            for j, t in enumerate(target_ids)
        ]
    )


def hierarchical_summary(section_metrics):
    """Equal sections within donor, donors within study, then studies within cancer.

    Missing target estimates remain missing; do not pool spots as patients.
    """
    metrics = ["pearson", "spearman"]
    donor = (
        section_metrics.groupby(["cancer", "study", "donor", "target", "family"], dropna=False)[
            metrics
        ]
        .mean()
        .reset_index()
    )
    study = (
        donor.groupby(["cancer", "study", "target", "family"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    cancer = (
        study.groupby(["cancer", "target", "family"], dropna=False)[metrics].mean().reset_index()
    )
    return donor, study, cancer


def first_order_prediction(model, features, cancer, device="cpu", chunk_size=512):
    """Frozen first-order readout in training-standardized target units."""
    import torch

    output = []
    with torch.inference_mode():
        for i in range(0, len(features), chunk_size):
            x = torch.as_tensor(
                features[i : i + chunk_size], dtype=model.x_center.dtype, device=device
            )
            output.append(model.functional_field(x, cancer).cpu().numpy())
    result = np.concatenate(output)
    active = (model.variable[cancer] & model.supported[cancer]).cpu().numpy()
    result[:, ~active] = np.nan
    return result
