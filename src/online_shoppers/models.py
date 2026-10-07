"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL, NUMERIC, SEED
from .features import FeaturePolicy


def make_pipeline(model: BaseEstimator | CatBoostClassifier, *, mode: str = "tree") -> Pipeline:
    """Собери preprocessing и классификатор с обязательной feature policy.

    Args:
        model: Необученный классификатор scikit-learn-совместимого API.
        mode: tree — числа без scaling; scaled — StandardScaler; binary — интервалы.

    Returns:
        Независимый pipeline, который можно клонировать для каждого фолда.

    Raises:
        ValueError: Если указан неизвестный режим обработки.
    """
    if mode not in {"tree", "scaled", "binary"}:
        raise ValueError(f"Неизвестный режим: {mode}")
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if mode == "scaled":
        numeric_steps.append(("scaler", StandardScaler()))
    elif mode == "binary":
        numeric_steps.append(
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=4,
                    encode="onehot-dense",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                    subsample=None,
                    random_state=SEED,
                ),
            )
        )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("policy", FeaturePolicy()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_candidates() -> dict[str, tuple[Pipeline, dict]]:
    """Создай модели и заранее заданные компактные сетки поиска.

    Returns:
        Словарь имя → (pipeline, сетка параметров); пустая сетка означает baseline.
    """
    from xgboost import XGBClassifier

    return {
        "Dummy": (make_pipeline(DummyClassifier(strategy="prior")), {}),
        "LogisticRegression": (
            make_pipeline(
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=SEED,
                ),
                mode="scaled",
            ),
            {},
        ),
        "DecisionTree": (
            make_pipeline(
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=50,
                    random_state=SEED,
                )
            ),
            {},
        ),
        "RandomForest": (
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=SEED,
                    n_jobs=1,
                )
            ),
            {"model__max_depth": [8, None]},
        ),
        "GradientBoosting": (
            make_pipeline(
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                    random_state=SEED,
                )
            ),
            {"model__max_depth": [2, 3]},
        ),
        "XGBoost": (
            make_pipeline(
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=SEED,
                    verbosity=0,
                )
            ),
            {"model__max_depth": [2, 4]},
        ),
        "LightGBM": (
            make_pipeline(
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    min_child_samples=30,
                    random_state=SEED,
                    n_jobs=1,
                    deterministic=True,
                    force_col_wise=True,
                    verbosity=-1,
                )
            ),
            {"model__num_leaves": [7, 15]},
        ),
        "CatBoost": (
            make_pipeline(
                CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    random_seed=SEED,
                    thread_count=1,
                    verbose=False,
                    allow_writing_files=False,
                    loss_function="Logloss",
                )
            ),
            {"model__depth": [4, 6]},
        ),
        "SVM_linear": (
            make_pipeline(
                SVC(
                    kernel="linear",
                    cache_size=512,
                    random_state=SEED,
                    max_iter=1000000,
                ),
                mode="scaled",
            ),
            {"model__C": [0.1, 1.0]},
        ),
        "SVM_rbf": (
            make_pipeline(
                SVC(
                    kernel="rbf",
                    cache_size=512,
                    random_state=SEED,
                    max_iter=1000000,
                ),
                mode="scaled",
            ),
            {"model__C": [1.0, 10.0], "model__gamma": ["scale", 0.01]},
        ),
        "BernoulliNB": (
            make_pipeline(BernoulliNB(binarize=None), mode="binary"),
            {"model__alpha": [0.1, 1.0]},
        ),
    }
