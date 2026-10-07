"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import FEATURES, NUMERIC, RAW_COLUMNS, TARGET, UCI_URL


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверь и выбери разрешённые признаки без обучения на статистиках.

    Args:
        frame: Таблица с разрешёнными колонками; лишние колонки игнорируются.

    Returns:
        Независимая таблица с фиксированным порядком признаков и Weekend 0/1.

    Raises:
        ValueError: Если схема, категории или числовые значения некорректны.
    """
    if not frame.columns.is_unique or not set(FEATURES) <= set(frame.columns):
        raise ValueError("Отсутствуют признаки или повторяются имена колонок")
    if frame.empty:
        raise ValueError("Пустая таблица")
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC:
        values = pd.to_numeric(result[column], errors="raise")
        if np.isinf(values.to_numpy(dtype=float)).any() or (values < 0).any():
            raise ValueError(f"Недопустимые значения: {column}")
        result[column] = values.astype(float)
    for column in ["BounceRates", "ExitRates", "SpecialDay"]:
        if (result[column] > 1).any():
            raise ValueError(f"Ожидается значение от 0 до 1: {column}")
    weekend = result["Weekend"]
    if not weekend.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("Weekend должен быть 0/1 или bool")
    result["Weekend"] = weekend.astype(float)
    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        values = pd.to_numeric(result[column], errors="raise")
        if (
            np.isinf(values.to_numpy(dtype=float)).any()
            or (values.dropna() < 0).any()
            or (values.dropna() % 1 != 0).any()
        ):
            raise ValueError(f"Ожидается неотрицательный целый код: {column}")
        result[column] = values.astype(float)
    for column in ["Month", "VisitorType"]:
        if not result[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Ожидаются строки: {column}")
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def validate_raw(frame: pd.DataFrame) -> None:
    """Проверь строгую схему исходного UCI-файла.

    Args:
        frame: Исходная таблица до удаления полных повторов.

    Raises:
        ValueError: Если схема неверна или target содержит пропуски/небинарные значения.
    """
    if set(frame.columns) != set(RAW_COLUMNS) or not frame.columns.is_unique:
        raise ValueError("Ожидается исходная схема UCI: 18 уникальных колонок")
    validate_features(frame)
    target = frame[TARGET]
    if target.isna().any() or not target.isin([False, True, 0, 1]).all():
        raise ValueError("Revenue должен быть бинарным и без пропусков")
    page = pd.to_numeric(frame["PageValues"], errors="raise")
    if np.isinf(page.to_numpy(dtype=float)).any() or (page < 0).any():
        raise ValueError("Некорректный PageValues в исходном CSV")


def load_data(path: Path, *, download: bool = False) -> pd.DataFrame:
    """Загрузи CSV локально или скачай официальный архив при явном разрешении.

    Args:
        path: Путь к CSV; при скачивании сюда записывается только нужный файл.
        download: Разрешить скачивание UCI, если локальный файл отсутствует.

    Returns:
        Проверенная исходная таблица, включая полные повторы.

    Raises:
        FileNotFoundError: Если файла нет и скачивание выключено.
        ValueError: Если CSV не соответствует контракту.
        OSError: Если чтение, сеть или запись недоступны.
    """
    if not path.is_file():
        if not download:
            raise FileNotFoundError(path)
        with urlopen(UCI_URL, timeout=60) as response:
            archive_bytes = response.read()
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read("online_shoppers_intention.csv")
        # Не извлекаем произвольные пути из архива.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes)
    frame = pd.read_csv(path)
    validate_raw(frame)
    return frame
