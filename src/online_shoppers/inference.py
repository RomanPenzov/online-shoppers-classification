"""Применение сохранённого pipeline и train-only порога."""

from pathlib import Path

import joblib
import pandas as pd

from .evaluation import positive_scores


def predict_file(model_path: Path, data_path: Path, output: Path) -> None:
    """Запиши scores и решения для новых сессий.

    Args:
        model_path: Доверенный joblib, созданный этим проектом.
        data_path: CSV с разрешёнными признаками, без обязательного Revenue/PageValues.
        output: Новый выходной CSV; существующий файл не перезаписывается.

    Raises:
        FileExistsError: Если выходной файл уже существует.
        ValueError: Если входная схема не соответствует feature policy.
    """
    if output.exists():
        raise FileExistsError(output)
    bundle = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    scores = positive_scores(bundle["pipeline"], frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"score": scores, "prediction": (scores >= bundle["threshold"]).astype(int)}
    ).to_csv(output, index=False)
