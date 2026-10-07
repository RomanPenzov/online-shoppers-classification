"""Проверь полный жизненный цикл на локальной синтетической таблице."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from test_pipelines import synthetic_data

from online_shoppers import training
from online_shoppers.config import FEATURES
from online_shoppers.experiment import run_experiment
from online_shoppers.inference import predict_file
from online_shoppers.models import make_pipeline
from online_shoppers.reproducibility import verify_runs


def test_repeated_experiment_and_saved_inference(tmp_path: Path, monkeypatch) -> None:
    """Повтори split, CV, OOF, сериализацию и инференс без реальных данных."""
    monkeypatch.setattr(
        training,
        "build_candidates",
        lambda: {
            "Dummy": (make_pipeline(DummyClassifier(strategy="prior")), {}),
            "LogisticRegression": (
                make_pipeline(LogisticRegression(max_iter=5000), mode="scaled"),
                {},
            ),
        },
    )
    source = synthetic_data(100)
    path = tmp_path / "input.csv"
    source.to_csv(path, index=False)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    first = run_experiment(path, tmp_path / "run1", expected_sha256=checksum)
    second = run_experiment(path, tmp_path / "run2", expected_sha256=checksum)
    assert first["split_sha256"] == second["split_sha256"]
    assert first["deterministic_signature"] == second["deterministic_signature"]
    verify_runs(tmp_path / "run1", tmp_path / "run2")
    assert first["threshold"] == second["threshold"]
    assert first["oof_repeat_max_abs_diff"] == 0
    split = json.loads((tmp_path / "run1/split_manifest.json").read_text())
    assert set(split["train_ids"]).isdisjoint(split["test_ids"])
    assert sorted(split["train_ids"] + split["test_ids"]) == list(range(len(source)))
    hits = np.zeros(len(split["train_ids"]), dtype=int)
    for fold in split["folds"]:
        assert set(fold["fit"]).isdisjoint(fold["valid"])
        hits[fold["valid"]] += 1
    np.testing.assert_array_equal(hits, 1)
    for name in ["cv_comparison.png", "selected_test.png", "model.joblib"]:
        assert (tmp_path / "run1" / name).stat().st_size > 0
    model = tmp_path / "run1/model.joblib"
    safe = tmp_path / "safe.csv"
    source[FEATURES].to_csv(safe, index=False)
    predict_file(model, safe, tmp_path / "safe_predictions.csv")
    poisoned = tmp_path / "poisoned.csv"
    source.assign(PageValues=1e10, Revenue=1 - source.Revenue).to_csv(poisoned, index=False)
    predict_file(model, poisoned, tmp_path / "poisoned_predictions.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "safe_predictions.csv"),
        pd.read_csv(tmp_path / "poisoned_predictions.csv"),
    )
    with pytest.raises(FileExistsError):
        run_experiment(path, tmp_path / "run1", expected_sha256=checksum)
    with pytest.raises(ValueError, match="CSV"):
        run_experiment(path, tmp_path / "wrong_checksum")
