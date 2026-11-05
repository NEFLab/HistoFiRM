import numpy as np
import pandas as pd
from scipy import sparse
from histofirm.scoring import log_expression, score_progeny
from histofirm.composition import evaluate
from histofirm.spatial import score, hierarchical_summary
from histofirm.clinical import censor_at, residual_indicator
from histofirm.clinical_statistics import cox_derivatives
from histofirm.pathology import mask_metrics


def test_linear_duplicate_collapse_before_log():
    x = pd.DataFrame([[3.0, 7.0], [1.0, 1.0]], index=["G", "G"], columns=["A", "B"])
    np.testing.assert_allclose(log_expression(x).values, np.log2([[5.0, 9.0]]))


def test_progeny_uses_signed_weights_and_absolute_denominator():
    expr = pd.DataFrame([[1.0, 3.0], [5.0, 1.0]], index=["A", "B"], columns=["S1", "S2"])
    weights = pd.DataFrame({"pathway": ["P", "P"], "gene": ["A", "B"], "weight": [2.0, -1.0]})
    score_, audit = score_progeny(expr, weights)
    np.testing.assert_allclose(score_.values[:, 0], [-1.0, 5 / 3])
    assert audit.absolute_weight_coverage.iloc[0] == 1


def test_composition_support_is_shared():
    r = np.random.default_rng(4)
    x = r.normal(size=(90, 2))
    y = r.normal(size=(90, 2)) + x[:, [0]]
    p = y + r.normal(size=y.shape)
    y[0, 0] = np.nan
    ans = evaluate(y, p, x, np.repeat([0, 1, 2], 30), ["A", "B"])
    assert ans.n.tolist() == [89, 90]
    np.testing.assert_allclose(ans.adjusted_r - ans.raw_r, ans.delta_r_after_adjustment)


def test_spatial_weighted_targets_stay_unavailable():
    a = sparse.csc_matrix(np.arange(1, 31).reshape(10, 3))
    sets = [
        {"target_id": "HALLMARK::A", "genes": [f"G{i}" for i in range(5)]},
        {"target_id": "PROGENY::B", "genes": [f"G{i}" for i in range(5)]},
    ]
    values, _ = score(a, [f"G{i}" for i in range(10)], sets)
    assert np.isfinite(values[:, 0]).all() and np.isnan(values[:, 1]).all()


def test_hierarchical_summary_does_not_overweight_multiregion_case():
    frame = pd.DataFrame(
        {
            "cancer": ["C"] * 5,
            "study": ["S"] * 5,
            "donor": ["A"] * 4 + ["B"],
            "target": ["T"] * 5,
            "family": ["F"] * 5,
            "pearson": [0, 0, 0, 0, 1],
            "spearman": [0, 0, 0, 0, 1],
        }
    )
    _, _, answer = hierarchical_summary(frame)
    assert answer.pearson.iloc[0] == 0.5


def test_administrative_censoring_preserves_early_last_followup():
    time, event = censor_at([24, 80, 60], [0, 1, 1])
    np.testing.assert_array_equal(time, [24, 60, 60])
    np.testing.assert_array_equal(event, [0, 0, 1])
    np.testing.assert_allclose(
        residual_indicator(["TRUE", "FALSE", ""], "MSKCC"), [0, 1, np.nan], equal_nan=True
    )


def test_cox_gradient_and_hessian_are_consistent():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    t = rng.integers(1, 20, 50).astype(float)
    e = rng.random(50) < 0.7
    beta = np.array([0.1, -0.3, 0.2])
    loss, g, H = cox_derivatives(beta, X, t, e)
    eps = 1e-5
    eye = np.eye(3) * eps
    fd = np.column_stack(
        [
            (
                cox_derivatives(beta + a, X, t, e, False)[1]
                - cox_derivatives(beta - a, X, t, e, False)[1]
            )
            / (2 * eps)
            for a in eye
        ]
    )
    np.testing.assert_allclose(H, fd, rtol=1e-7, atol=1e-8)


def test_annotation_metrics_use_declared_pixel_denominator():
    a = np.zeros((2, 2, 3), dtype=int)
    a[:, :, 0] = [[1, 1], [2, 0]]
    a[:, :, 1] = [[3, 0], [4, 4]]
    mapping = {
        "region_channel": 0,
        "nucleus_channel": 1,
        "region_denominator_codes": [1, 2],
        "region_classes": {"tumor": [1], "stroma": [2]},
        "nuclear_pixel_denominator_codes": [3, 4],
        "nuclear_pixel_classes": {"lymphocyte": [3]},
    }
    result = mask_metrics(a, mapping)
    assert result["region_tumor"] == 2 / 3 and result["nuclear_pixel_lymphocyte"] == 1 / 3


def test_fixed_axis_scoring_retains_missing_targets():
    import json
    from pathlib import Path
    from histofirm.scoring import score_fixed_axis

    config = json.loads((Path(__file__).resolve().parents[1] / "configs/scoring.json").read_text())
    rng = np.random.default_rng(1)
    expr = pd.DataFrame(
        rng.gamma(2, 2, (50, 8)),
        index=[f"G{i}" for i in range(50)],
        columns=[f"S{i}" for i in range(8)],
    )
    result = score_fixed_axis(
        expr,
        {"HALLMARK::A": [f"G{i}" for i in range(10)]},
        None,
        ["HALLMARK::A", "HALLMARK::MISSING"],
        config,
    )
    assert (
        result.shape == (8, 2)
        and np.isfinite(result.iloc[:, 0]).all()
        and result.iloc[:, 1].isna().all()
    )


def test_reporting_keeps_missing_predictions_and_constant_status():
    from histofirm.evaluation import one, table, macro

    y = np.arange(12, dtype=float)
    p = y.copy()
    p[0] = np.nan
    assert one(y, p)["status"] == "MISSING_PREDICTION"
    assert one(y, np.ones(12))["status"] == "CONSTANT_PREDICTION"
    rows = table(y[:, None], p[:, None], ["C"] * 12, ["T"])
    assert macro(rows)["value"] is None
    assert one(y, -y)["pearson"] == -1
