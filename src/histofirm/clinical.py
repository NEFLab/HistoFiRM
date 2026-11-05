"""Matched-subset survival association with numeric stage and residual disease."""

import numpy as np
import pandas as pd
from scipy import stats
from .clinical_statistics import cox_fit, ph_time_score, p_adjust


def residual_indicator(values, cohort):
    mappings = {
        "TCGA": {
            "No Macroscopic disease": 0.0,
            "1-10 mm": 1.0,
            "11-20 mm": 1.0,
            ">20 mm": 1.0,
            "Not Available": np.nan,
            "": np.nan,
        },
        "MSKCC": {"TRUE": 0.0, "FALSE": 1.0, "": np.nan},
    }
    mapping = mappings[cohort]
    normalized = ["" if pd.isna(v) else str(v).strip() for v in values]
    if cohort == "MSKCC":
        normalized = [v.upper() for v in normalized]
    if set(normalized) - set(mapping):
        raise ValueError("Unrecognized residual-disease coding.")
    return np.asarray([mapping[v] for v in normalized], float)


def censor_at(time, event, months=60.0):
    time = np.asarray(time, float)
    event = np.asarray(event)
    if not np.isfinite(time).all() or (time <= 0).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("Invalid survival endpoint.")
    return np.minimum(time, months), (event.astype(bool) & (time <= months)).astype(int)


def matched_cox(frame, age_mean, age_sd, risk_sd, threshold, config):
    """Requires harmonized time/event and residual_present; no endpoint inference.

    Scaling constants come from the full TCGA training cohort, before restricting
    to residual-information availability. Both risk forms and both adjustment
    models use exactly the same residual-available patients.
    """
    f = frame.loc[np.isfinite(frame.residual_present)].copy()
    t = f.time.to_numpy(float)
    e = f.event.to_numpy(int)
    if not np.isfinite(t).all() or (t <= 0).any() or not np.isin(e, [0, 1]).all():
        raise ValueError("Invalid endpoint.")
    if age_sd <= 0 or risk_sd <= 0:
        raise ValueError("Invalid training reference scale.")
    age = (f.age.to_numpy(float) - age_mean) / age_sd
    stage = f.stage.to_numpy(float)
    missing = ~np.isfinite(stage)
    stage = np.where(missing, 3.0, stage)
    if not np.isin(stage, [1, 2, 3, 4]).all():
        raise ValueError("Stage must be numeric I-IV or missing.")
    results = []
    for kind in ("group", "continuous"):
        risk = (
            (f.risk.to_numpy(float) > threshold).astype(float)
            if kind == "group"
            else f.risk.to_numpy(float) / risk_sd
        )
        columns = [risk, age, stage]
        terms = ["CHF", "age_per_TCGA_SD", "stage_per_level"]
        if missing.any():
            columns.append(missing.astype(float))
            terms.append("stage_missing")
        for adjustment in ("base", "residual"):
            cols = columns + (
                [f.residual_present.to_numpy(float)] if adjustment == "residual" else []
            )
            names = terms + (["residual_present"] if adjustment == "residual" else [])
            X = np.column_stack(cols)
            X -= X.mean(0)
            fit = cox_fit(X, t, e, config)
            rows = []
            for j, term in enumerate(names):
                b = fit["beta"][j]
                se = np.sqrt(fit["cov"][j, j])
                low, high = np.exp(b + np.array([-1, 1]) * 1.959963984540054 * se)
                rows.append(
                    dict(
                        term=term,
                        HR=np.exp(b),
                        ci_low=low,
                        ci_high=high,
                        p_nominal=2 * stats.norm.sf(abs(b / se)),
                    )
                )
            ph = ph_time_score(fit["beta"], X, t, e, index=0)
            ph.pop("private_event_rows", None)
            results.append(
                {
                    "risk_form": kind,
                    "adjustment": adjustment,
                    "n": len(f),
                    "events": int(e.sum()),
                    "effects": rows,
                    "risk_PH": ph,
                    "fit": fit,
                }
            )
    return results


def correct_risk_family(cohort_results):
    """Two cohorts x two CHF forms: four tests per adjustment family."""
    for adjustment in ("base", "residual"):
        members = [
            r
            for results in cohort_results.values()
            for r in results
            if r["adjustment"] == adjustment
        ]
        if len(members) != 4:
            raise ValueError("Expected all four prespecified risk tests.")
        qs = p_adjust([r["effects"][0]["p_nominal"] for r in members], "Holm", 4)
        for r, q in zip(members, qs):
            r["CHF_p_Holm4"] = float(q)
    return cohort_results
