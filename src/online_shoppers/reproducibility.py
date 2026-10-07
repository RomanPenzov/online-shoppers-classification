"""Сравнение двух независимых полных запусков, исключая время выполнения."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_runs(first: Path, second: Path) -> None:
    """Проверь данные, split, выбор и численные результаты двух запусков.

    Args:
        first: Каталог первого завершённого эксперимента.
        second: Каталог второго завершённого эксперимента.

    Raises:
        AssertionError: Если протокол, выбор или метрики отличаются сверх допуска.
        OSError: Если обязательные артефакты отсутствуют.
    """
    records = [
        json.loads((path / "selection_before_test.json").read_text(encoding="utf-8"))
        for path in (first, second)
    ]
    for key in ("raw_sha256", "split_sha256", "model", "parameters", "score_kind"):
        if records[0][key] != records[1][key]:
            raise AssertionError(f"Различается {key}")
    np.testing.assert_allclose(
        records[0]["threshold"], records[1]["threshold"], rtol=1e-10, atol=1e-12
    )
    for filename in (
        "cv_comparison.csv",
        "selected_oof.csv",
        "selected_test_metrics.csv",
        "selected_test_predictions.csv",
    ):
        left, right = [pd.read_csv(path / filename) for path in (first, second)]
        columns = [column for column in left if not column.endswith("seconds")]
        pd.testing.assert_frame_equal(
            left[columns], right[columns], check_exact=False, rtol=1e-10, atol=1e-12
        )
