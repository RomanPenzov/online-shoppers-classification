"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FEATURES
from .data import validate_features


class FeaturePolicy(TransformerMixin, BaseEstimator):
    """Применяй один allowlist признаков при fit и при инференсе.

    PageValues, Revenue и любые неизвестные поля отбрасываются до preprocessing.
    Обучаемых статистик у преобразования нет.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FeaturePolicy:
        """Проверь входную схему и зафиксируй разрешённые колонки.

        Args:
            X: Входные данные, допускающие дополнительные колонки.
            y: Неиспользуемая цель для совместимости с Pipeline.

        Returns:
            Текущий преобразователь.

        Raises:
            ValueError: Если вход нарушает контракт признаков.
        """
        validate_features(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Верни только разрешённые признаки в фиксированном порядке.

        Args:
            X: Новые сессии, в том числе без PageValues и Revenue.

        Returns:
            Проверенная таблица из 16 признаков.

        Raises:
            ValueError: Если обязательные признаки отсутствуют или некорректны.
        """
        return validate_features(X)
