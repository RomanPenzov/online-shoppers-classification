"""Оркестрация эксперимента без зависимости от Jupyter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .config import EXPECTED_SHA256, FEATURES, N_SPLITS, PACKAGES, SEED, TARGET, TEST_SIZE
from .data import load_data, validate_features
from .evaluation import choose_threshold, collect_oof, evaluate_scores, positive_scores
from .plots import plot_results
from .training import compare_models


def run_experiment(
    data_path: Path,
    output: Path,
    *,
    download: bool = False,
    expected_sha256: str = EXPECTED_SHA256,
) -> dict:
    """Выполни фиксированный протокол выбора и единственной test-оценки.

    Args:
        data_path: Исходный CSV UCI; отсутствующий файл скачивается только явно.
        output: Новый или пустой каталог артефактов; результаты не перезаписываются.
        download: Разрешить загрузку официального CSV.
        expected_sha256: Ожидаемые байты CSV; иной хеш означает другой протокол.

    Returns:
        Зафиксированная до открытия test запись выбора модели.

    Raises:
        FileExistsError: Если выходной каталог не пуст.
        ValueError: Если данные не соответствуют контракту или контрольной сумме.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Выберите новый каталог результатов: {output}")
    output.mkdir(parents=True, exist_ok=True)
    DATA_PATH, ARTIFACT_DIR = data_path, output
    VERSIONS = {name: importlib.metadata.version(name) for name in PACKAGES}
    VERSIONS["python"] = platform.python_version()
    VERSIONS["platform"] = platform.platform()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        raw = load_data(DATA_PATH, download=download)
        data_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
        if data_sha256 != expected_sha256:
            raise ValueError("CSV отличается от частей I–II; проверь источник и новый протокол")
        data = raw.drop_duplicates().reset_index(drop=True)
        X = validate_features(data)
        y = data[TARGET].astype(int)
        if X.duplicated().any():
            raise ValueError("Повторяются разрешённые X; нужен отдельный групповой протокол")
        train_ids, test_ids = train_test_split(
            np.arange(len(data)),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=SEED,
        )
        X_train, y_train = X.iloc[train_ids].copy(), y.iloc[train_ids].copy()
        # test_ids сохраняем, но test-прогнозы и метрики пока не вычисляем.
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(cv.split(X_train, y_train))
        assert set(train_ids).isdisjoint(test_ids)
        hits = np.zeros(len(train_ids), dtype=int)
        for fit_idx, valid_idx in folds:
            assert not np.intersect1d(fit_idx, valid_idx).size
            hits[valid_idx] += 1
        assert np.all(hits == 1)
        split_manifest = {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "raw_sha256": data_sha256,
            "deduplication": "full rows; keep first; reset index",
            "train_ids": train_ids.tolist(),
            "test_ids": test_ids.tolist(),
            "folds": [{"fit": fit.tolist(), "valid": valid.tolist()} for fit, valid in folds],
            "features": FEATURES,
        }
        split_text = json.dumps(split_manifest, sort_keys=True)
        split_sha256 = hashlib.sha256(split_text.encode()).hexdigest()
        (ARTIFACT_DIR / "split_manifest.json").write_text(split_text, encoding="utf-8")
        print(
            f"Raw: {len(raw)}; deduplicated: {len(data)}; "
            f"train: {len(train_ids)}; test: {len(test_ids)}"
        )
        print(
            "Train purchases:",
            int(y_train.sum()),
            "; train prevalence:",
            round(y_train.mean(), 4),
        )
        print("CSV SHA-256:", data_sha256)
        print("Split SHA-256:", split_sha256)
        comparison, tuned_pipelines = compare_models(X_train, y_train, folds, ARTIFACT_DIR)
        # Dummy — контрольная линия, а не кандидат для финального бизнес-решения.
        selected_name = comparison.drop(index="Dummy").index[0]
        selected_pipeline = tuned_pipelines[selected_name]
        print(comparison.to_string())
        print("Выбрана по CV AP до открытия test:", selected_name)
        repeat_train, repeat_test = train_test_split(
            np.arange(len(data)),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=SEED,
        )
        np.testing.assert_array_equal(train_ids, repeat_train)
        np.testing.assert_array_equal(test_ids, repeat_test)
        repeat_folds = list(
            StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
                X_train, y_train
            )
        )
        for (fit, valid), (fit_again, valid_again) in zip(folds, repeat_folds, strict=True):
            np.testing.assert_array_equal(fit, fit_again)
            np.testing.assert_array_equal(valid, valid_again)
        oof_scores = collect_oof(selected_pipeline, X_train, y_train, folds)
        repeat_oof = collect_oof(selected_pipeline, X_train, y_train, repeat_folds)
        np.testing.assert_allclose(oof_scores, repeat_oof, rtol=1e-10, atol=1e-12)
        threshold = choose_threshold(y_train, oof_scores)
        assert threshold == choose_threshold(y_train, repeat_oof)
        with threadpool_limits(limits=1):
            final_model = clone(selected_pipeline).fit(X_train, y_train)
            repeated_model = clone(selected_pipeline).fit(X_train, y_train)
            reference_scores = positive_scores(final_model, X_train)
            repeated_scores = positive_scores(repeated_model, X_train)
        np.testing.assert_allclose(reference_scores, repeated_scores, rtol=1e-10, atol=1e-12)
        pd.DataFrame(
            {"row_id": train_ids, "target": y_train.to_numpy(), "score": oof_scores}
        ).to_csv(
            ARTIFACT_DIR / "selected_oof.csv",
            index=False,
        )
        selection_record = {
            "model": selected_name,
            "parameters": json.loads(comparison.loc[selected_name, "params"]),
            "selection_metric": "mean CV average_precision",
            "CV_AP": float(comparison.loc[selected_name, "CV_AP_mean"]),
            "threshold": threshold,
            "threshold_rule": "maximum train OOF F1; largest threshold on tie",
            "score_kind": "decision_function"
            if isinstance(final_model.named_steps["model"], SVC)
            else "predict_proba",
            "versions": VERSIONS,
            "raw_sha256": data_sha256,
            "split_sha256": split_sha256,
            "oof_repeat_max_abs_diff": float(np.max(np.abs(oof_scores - repeat_oof))),
            "refit_repeat_max_abs_diff": float(np.max(np.abs(reference_scores - repeated_scores))),
            "real_time_limitation": (
                "CSV has no point-in-time snapshots; "
                "availability of retained features is conditional"
            ),
            "prior_test_exposure": "same test was reported in parts I and II",
        }
        deterministic_columns = [column for column in comparison if not column.endswith("seconds")]
        signature_payload = (
            comparison[deterministic_columns].to_csv(float_format="%.10g") + split_sha256
        )
        selection_record["deterministic_signature"] = hashlib.sha256(
            signature_payload.encode()
        ).hexdigest()
        # Этот файл фиксируется ДО вычисления test-метрик.
        (ARTIFACT_DIR / "selection_before_test.json").write_text(
            json.dumps(selection_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            pd.Series(
                evaluate_scores(y_train, oof_scores, threshold), name="Train OOF / development"
            )
        )
        print("Reproducibility checks passed; selected:", selected_name, "; threshold:", threshold)
        joblib.dump(
            {"pipeline": final_model, "threshold": threshold, "selection": selection_record},
            ARTIFACT_DIR / "model.joblib",
        )
        X_test, y_test = X.iloc[test_ids].copy(), y.iloc[test_ids].copy()
        with threadpool_limits(limits=1):
            test_scores = positive_scores(final_model, X_test)
        test_metrics = evaluate_scores(y_test, test_scores, threshold)
        test_table = pd.DataFrame([{"model": selected_name, **test_metrics}]).set_index("model")
        test_table.to_csv(ARTIFACT_DIR / "selected_test_metrics.csv")
        pd.DataFrame(
            {
                "row_id": test_ids,
                "target": y_test.to_numpy(),
                "score": test_scores,
                "prediction": (test_scores >= threshold).astype(int),
            }
        ).to_csv(
            ARTIFACT_DIR / "selected_test_predictions.csv",
            index=False,
        )
        print(test_table.to_string())
        plot_results(comparison, y_test, test_scores, threshold, selected_name, ARTIFACT_DIR)
        return selection_record
