"""Patient-level summaries of within-patient ROI associations."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .common import require
from .statistics import corr, bootstrap_mean, permutation_mean_p, bh


def mask_metrics(mask, mapping):
    a = np.asarray(mask)
    require(a.ndim == 3 and a.shape[2] >= 3, "EXPECTED_THREE_CHANNEL_PANOP_MASK")
    require(mapping["region_channel"] != mapping["nucleus_channel"], "PANOP_CHANNELS_NOT_DISTINCT")
    for prefix in ["region", "nuclear_pixel"]:
        den = mapping[prefix + "_denominator_codes"]
        cl = mapping[prefix + "_classes"]
        require(bool(den) and bool(cl), "MISSING_CERTIFIED_PANOP_CLASS_CODES")
        require(
            all(bool(v) and set(v) <= set(den) for v in cl.values()),
            "PANOP_CLASS_CODES_OUTSIDE_DECLARED_DENOMINATOR",
        )
    out = {}
    reg = a[:, :, mapping["region_channel"]]
    nuc = a[:, :, mapping["nucleus_channel"]]
    for typ, v in [("region", reg), ("nuclear_pixel", nuc)]:
        classes = mapping[typ + "_classes"]
        denominator = np.isin(v, mapping[typ + "_denominator_codes"])
        den = int(denominator.sum())
        out[typ + "_denominator_pixels"] = den
        for name, codes in classes.items():
            out[typ + "_" + name] = (
                float(np.sum(np.isin(v, codes) & denominator) / den) if den else np.nan
            )
    # Boundary channel is never treated as a labelled instance-ID map.
    out["nucleus_instances_counted"] = False
    return out


def associations(table, response, target_ids, pairs, min_roi, bootstrap=1000):
    """Rows must have matching patient/ROI IDs and an aligned response matrix.

    Annotation denominators are supplied explicitly. Nuclear pixel proportions
    are not cell counts, and an ROI score is not an intra-ROI prediction map.
    """
    table = table.reset_index(drop=True)
    if len(table) != len(response) or table.roi_id.duplicated().any():
        raise ValueError("ROI axis mismatch or duplicate ROI.")
    records = []
    for pair in pairs:
        j = list(target_ids).index(pair["target_id"])
        label = pair["annotation_column"]
        raw = []
        adjusted = []
        for _, g in table.groupby("patient_id"):
            ii = g.index.to_numpy()
            v = g[label].to_numpy(float)
            ok = np.isfinite(response[ii, j]) & np.isfinite(v)
            if ok.sum() >= min_roi:
                raw.append(corr(response[ii[ok], j], v[ok], True))
            if pair.get("adjust_for"):
                z = g[pair["adjust_for"]].to_numpy(float)
                valid = ok & np.isfinite(z).all(1)
                if valid.sum() >= max(10, z.shape[1] + 4):
                    X = np.c_[np.ones(valid.sum()), z[valid]]
                    a = rankdata(response[ii[valid], j])
                    b = rankdata(v[valid])
                    ar = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
                    br = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
                    adjusted.append(corr(ar, br))
        for kind, values in [("unadjusted", raw)] + (
            [("conditional", adjusted)] if pair.get("adjust_for") else []
        ):
            records.append(
                {
                    "target": pair["target_id"],
                    "annotation": label,
                    "estimand": kind,
                    **bootstrap_mean(values, bootstrap),
                    "p_patient_sign_flip": permutation_mean_p(values),
                }
            )
    result = pd.DataFrame(records)
    result["q"] = bh(result.p_patient_sign_flip)
    return result
