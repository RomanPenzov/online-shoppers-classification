from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from threadpoolctl import threadpool_limits

from online_shoppers.config import FEATURES, NUMERIC, RAW_COLUMNS, SEED, TARGET
from online_shoppers.data import load_data, validate_raw
from online_shoppers.evaluation import choose_threshold, positive_scores
from online_shoppers.models import build_candidates, make_pipeline


def synthetic_data(n: int = 80) -> pd.DataFrame:
    """Создай маленький фиктивный датасет без обращения к сети.

    Args:
        n: Число строк, не меньше 20.

    Returns:
        Таблица исходной схемы с двумя классами и разнообразными признаками.

    Raises:
        ValueError: Если запрошено меньше 20 строк.
    """
    if n < 20:
        raise ValueError("Нужно минимум 20 строк")
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame({column: rng.uniform(0, 1, n) for column in NUMERIC})
    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        frame[column] = rng.integers(1, 4, n)
    frame["Month"] = np.where(np.arange(n) % 2, "May", "Nov")
    frame["VisitorType"] = np.where(np.arange(n) % 3, "Returning_Visitor", "New_Visitor")
    frame["Weekend"] = np.arange(n) % 2 == 0
    frame["PageValues"] = rng.uniform(0, 100, n)
    frame[TARGET] = (frame["ProductRelated"] > 0.6).astype(int)
    return frame.loc[:, RAW_COLUMNS]


class TestPipelines(unittest.TestCase):
    """Проверь контракт данных и обучение pipeline на фиктивных сессиях."""

    def test_local_loader(self) -> None:
        """Проверь CSV roundtrip и отсутствие неявного скачивания."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            source = synthetic_data()
            source.to_csv(path, index=False)
            loaded = load_data(path)
            self.assertEqual(loaded.shape, source.shape)
            with self.assertRaises(FileNotFoundError):
                load_data(Path(directory) / "missing.csv")

    def test_invalid_schema_and_target(self) -> None:
        """Отклони пустую таблицу, неверную схему, target и бесконечности."""
        source = synthetic_data()
        cases = [
            source.iloc[:0],
            source.drop(columns="Browser"),
            source.assign(extra=1),
            source.assign(Revenue=2),
            source.assign(Revenue=np.nan),
            source.assign(BounceRates=np.inf),
            source.assign(Weekend="sometimes"),
            source.assign(Browser=1.5),
        ]
        for frame in cases:
            with (
                self.subTest(columns=list(frame.columns)),
                self.assertRaises(ValueError),
            ):
                validate_raw(frame)

    def test_policy_at_fit_and_inference(self) -> None:
        """Докажи инвариантность модели к запрещённым признакам."""
        source = synthetic_data()
        pipeline = make_pipeline(LogisticRegression(max_iter=5000), mode="scaled")
        first = clone(pipeline).fit(source, source[TARGET])
        changed = source.assign(PageValues=1e12, Revenue=1 - source[TARGET], unknown=999)
        second = clone(pipeline).fit(changed, source[TARGET])
        expected = positive_scores(first, source)
        np.testing.assert_allclose(expected, positive_scores(first, changed), atol=0, rtol=0)
        np.testing.assert_allclose(expected, positive_scores(second, changed), atol=0, rtol=0)
        np.testing.assert_allclose(expected, positive_scores(first, source[FEATURES]))
        transformed = first.named_steps["policy"].transform(source)
        self.assertEqual(list(transformed.columns), FEATURES)
        self.assertNotIn(TARGET, transformed.columns)
        self.assertNotIn("PageValues", transformed.columns)
        self.assertFalse(
            any(
                "PageValues" in name or "Revenue" in name
                for name in first.named_steps["preprocessor"].get_feature_names_out()
            )
        )

    def test_all_models_fit(self) -> None:
        """Обучи каждое семейство моделей на синтетических данных."""
        source = synthetic_data()
        for name, (pipeline, _) in build_candidates().items():
            with self.subTest(model=name), threadpool_limits(limits=1):
                fitted = clone(pipeline).fit(source, source[TARGET])
                self.assertEqual(positive_scores(fitted, source).shape, (len(source),))
                if name == "BernoulliNB":
                    transformed = fitted[:-1].transform(source)
                    self.assertTrue(np.isin(transformed, [0, 1]).all())

    def test_imputer_fit_only_on_training_rows(self) -> None:
        """Убедись, что transform validation не меняет статистики imputer."""
        source = synthetic_data()
        source.loc[:9, "Administrative"] = np.nan
        train, validation = source.iloc[:60].copy(), source.iloc[60:].copy()
        pipeline = make_pipeline(LogisticRegression(max_iter=5000), mode="scaled")
        fitted = pipeline.fit(train, train[TARGET])
        imputer = (
            fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
        )
        before = imputer.statistics_.copy()
        self.assertAlmostEqual(before[0], train["Administrative"].median())
        validation["Administrative"] = 1e9
        fitted.predict(validation)
        np.testing.assert_array_equal(before, imputer.statistics_)

    def test_unknown_categories_and_threshold(self) -> None:
        """Проверь новые категории и выбор порога на простом примере."""
        source = synthetic_data()
        fitted = make_pipeline(BernoulliNB(binarize=None), mode="binary").fit(
            source, source[TARGET]
        )
        novel = source.iloc[:3].copy()
        novel["Month"] = "unseen"
        self.assertTrue(np.isfinite(positive_scores(fitted, novel)).all())
        target = pd.Series([0, 0, 1, 1])
        score = np.array([-2.0, -1.0, 1.0, 2.0])
        self.assertEqual(choose_threshold(target, score), 1.0)
