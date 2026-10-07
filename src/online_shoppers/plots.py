"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from .evaluation import evaluate_scores


def plot_results(
    comparison: pd.DataFrame,
    target: pd.Series,
    score: np.ndarray,
    threshold: float,
    name: str,
    artifact_dir: Path,
) -> None:
    """Сохрани сравнение CV и итоговые кривые единственной выбранной модели.

    Args:
        comparison: Итоговая CV-таблица всех моделей.
        target: Метки финального test.
        score: Уже рассчитанные test-scores.
        threshold: Порог, ранее выбранный на train OOF.
        name: Имя выбранной модели.
        artifact_dir: Каталог PNG-артефактов.
    """
    ordered = comparison.sort_values("CV_AP_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered.index, ordered["CV_AP_mean"], xerr=ordered["CV_AP_std"])
    ax.set(xlabel="Mean CV Average Precision ± fold std", title="Train CV comparison")
    fig.tight_layout()
    fig.savefig(artifact_dir / "cv_comparison.png", bbox_inches="tight")
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(target, score)
    fpr, tpr, _ = roc_curve(target, score)
    metrics = evaluate_scores(target, score, threshold)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].step(recall, precision, where="post")
    axes[0].axhline(target.mean(), linestyle="--", color="gray")
    axes[0].scatter(metrics["recall"], metrics["precision"], color="red")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Test AP={metrics['AP']:.3f}")
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(xlabel="FPR", ylabel="TPR", title=f"Test ROC AUC={metrics['ROC_AUC']:.3f}")
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    axes[2].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[2].text(j, i, str(value), ha="center", va="center", color="darkred")
    axes[2].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion matrix",
    )
    fig.suptitle(f"{name}; threshold selected on train OOF")
    fig.tight_layout()
    fig.savefig(artifact_dir / "selected_test.png", bbox_inches="tight")
    plt.close(fig)
