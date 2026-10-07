"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def positive_scores(fitted: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Получи score положительного класса без требования вероятностной калибровки.

    Args:
        fitted: Обученный pipeline для классов 0 и 1.
        features: Сессии для предсказания.

    Returns:
        Одномерный массив вероятностей или SVM decision scores.

    Raises:
        ValueError: Если классы или вычисленные scores некорректны.
    """
    if not np.array_equal(fitted.classes_, [0, 1]):
        raise ValueError("Ожидаются классы [0, 1]")
    if isinstance(fitted.named_steps["model"], SVC):
        score = fitted.decision_function(features)
    else:
        score = fitted.predict_proba(features)[:, 1]
    score = np.asarray(score, dtype=float)
    if score.shape != (len(features),) or not np.isfinite(score).all():
        raise ValueError("Некорректные scores")
    return score


def evaluate_scores(target: pd.Series, score: np.ndarray, threshold: float) -> dict:
    """Оцени ранжирование и решения при заранее фиксированном пороге.

    Args:
        target: Истинные метки 0/1, оба класса обязательны для ROC AUC.
        score: Scores положительного класса.
        threshold: Порог решения; значения выше или равные означают класс 1.

    Returns:
        Метрики, элементы confusion matrix и доля положительных решений.
    """
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "AP": float(average_precision_score(target, score)),
        "ROC_AUC": float(roc_auc_score(target, score)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "F1": float(f1_score(target, predicted, zero_division=0)),
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "positive_rate": float(predicted.mean()),
    }


def choose_threshold(target: pd.Series, score: np.ndarray) -> float:
    """Найди порог максимального F1 на train OOF с фиксированным правилом ничьей.

    Args:
        target: Метки обучающей выборки.
        score: OOF-scores в том же порядке.

    Returns:
        Наибольший порог среди равных максимумов F1.
    """
    precision, recall, thresholds = precision_recall_curve(target, score)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def collect_oof(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Получай OOF-scores при независимом fit preprocessing на каждом фолде.

    Args:
        pipeline: Зафиксированный необученный pipeline.
        features: Только train-часть данных.
        target: Train-метки в порядке features.
        folds: Позиционные индексы fit/validation внутри train.

    Returns:
        OOF-scores, по одному на каждую строку train.

    Raises:
        ValueError: Если folds пересекаются или не покрывают train ровно один раз.
    """
    scores = np.full(len(target), np.nan)
    hits = np.zeros(len(target), dtype=int)
    for fit_idx, valid_idx in folds:
        if np.intersect1d(fit_idx, valid_idx).size:
            raise ValueError("Пересечение fit и validation")
        with threadpool_limits(limits=1):
            fitted = clone(pipeline).fit(features.iloc[fit_idx], target.iloc[fit_idx])
            scores[valid_idx] = positive_scores(fitted, features.iloc[valid_idx])
        hits[valid_idx] += 1
    if not np.all(hits == 1) or not np.isfinite(scores).all():
        raise ValueError("Каждая train-строка должна иметь один OOF-score")
    return scores
