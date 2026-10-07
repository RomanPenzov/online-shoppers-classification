"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .models import build_candidates


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Сравни все кандидаты по одним folds и сохрани полный журнал поиска.

    Args:
        features: Только train-признаки.
        target: Только train-метки.
        folds: Общие фиксированные CV-разбиения.
        artifact_dir: Каталог для таблиц поиска, исключаемый из Git.

    Returns:
        Таблица лучших конфигураций и соответствующие необученные pipeline.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows, selected = [], {}
    scoring = {
        "AP": "average_precision",
        "ROC_AUC": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "F1": "f1",
    }
    for name, (pipeline, grid) in build_candidates().items():
        started = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=scoring,
            refit="AP",
            cv=folds,
            n_jobs=1,
            error_score="raise",
        )
        with threadpool_limits(limits=1):
            search.fit(features, target)
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(artifact_dir / f"search_{name}.csv", index=False)
        best = results.iloc[search.best_index_]
        row = {
            "model": name,
            "params": json.dumps(search.best_params_, sort_keys=True),
            "search_seconds": time.perf_counter() - started,
            "fit_seconds": float(best["mean_fit_time"]),
            "score_seconds": float(best["mean_score_time"]),
        }
        for metric in scoring:
            row[f"CV_{metric}_mean"] = float(best[f"mean_test_{metric}"])
            row[f"CV_{metric}_std"] = float(best[f"std_test_{metric}"])
        rows.append(row)
        selected[name] = clone(pipeline).set_params(**search.best_params_)
        print(
            f"{name}: CV AP={row['CV_AP_mean']:.4f}; {row['search_seconds']:.1f} s",
            flush=True,
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(
            ["CV_AP_mean", "model"],
            ascending=[False, True],
            kind="stable",
        )
        .set_index("model")
    )
    table.to_csv(artifact_dir / "cv_comparison.csv")
    return table, selected
