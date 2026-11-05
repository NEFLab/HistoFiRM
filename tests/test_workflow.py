import importlib.util
from pathlib import Path
import numpy as np
from histofirm.common import load
from histofirm.data import Patients
from histofirm.prepare import raw_cache
from histofirm.training import fit
from histofirm.prediction import predict, merge_predictions
import pytest


def demo(tmp_path):
    fn = Path(__file__).resolve().parents[1] / "examples/make_demo.py"
    spec = importlib.util.spec_from_file_location("demo_data", fn)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_demo(tmp_path / "input with spaces")


def test_select_refit_predict_round_trip(tmp_path):
    root = demo(tmp_path)
    train = Patients(root / "fit.json", True)
    valid = Patients(root / "validation.json", True)
    outer = Patients(root / "outer.json", True)
    cache = raw_cache([train, valid, outer], tmp_path / "moments.npz", 7)
    config = load(root / "config.json")
    receipt = fit(train, valid, cache, config, tmp_path / "select", 0)
    fit(outer, None, cache, config, tmp_path / "refit", 0, selected=receipt)
    result = predict(
        tmp_path / "refit/model.pt",
        Patients(root / "test.json"),
        tmp_path / "prediction.npz",
        chunk_size=5,
    )
    assert result.shape == (20, 3) and np.isfinite(result).all()
    with pytest.raises(FileExistsError):
        fit(train, valid, cache, config, tmp_path / "select", 0)
    with pytest.raises(ValueError, match="overlap"):
        fit(train, train, cache, config, tmp_path / "invalid", 0)


def test_external_fold_mean_does_not_silently_drop_missing_values(tmp_path):
    files = []
    for k in range(5):
        p = np.array([[float(k), float(k)]])
        if k == 3:
            p[0, 1] = np.nan
        fn = tmp_path / f"fold{k}.npz"
        np.savez(
            fn, fold=k, patient_ids=["S"], cancers=["C"], target_ids=["T1", "T2"], prediction=p
        )
        files.append(fn)
    result = merge_predictions(files, tmp_path / "merged.npz", external=True)
    assert result["prediction"][0, 0] == 2 and np.isnan(result["prediction"][0, 1])
