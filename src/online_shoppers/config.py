"""Переиспользуемая логика проекта Online Shoppers."""

from __future__ import annotations

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
EXPECTED_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "xgboost",
    "lightgbm",
    "catboost",
    "threadpoolctl",
]
NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]
CATEGORICAL = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Revenue"
RAW_COLUMNS = FEATURES + ["PageValues", TARGET]
