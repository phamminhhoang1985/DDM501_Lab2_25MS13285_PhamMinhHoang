"""
Preprocessing and feature engineering.

Two things happen here, and keeping them straight matters:

  add_derived_features  works on the DataFrame and encodes DOMAIN knowledge —
                        ratios and counts a credit analyst would compute by hand.
  build_preprocessor    returns an unfitted sklearn transformer that is part of
                        the model Pipeline, so scaling and encoding are FITTED ON
                        TRAINING DATA ONLY and travel with the model.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config import (
    BILL_FEATURES,
    CATEGORICAL_FEATURES,
    PAY_AMT_FEATURES,
    PAY_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the six engineered features listed in config.DERIVED_FEATURES."""
    out = df.copy()

    # Safe denominator: replace zero with NaN so we get NaN instead of inf
    limit_bal = out["LIMIT_BAL"].replace(0, np.nan)

    # Average bill amount across 6 months
    avg_bill = out[BILL_FEATURES].mean(axis=1)

    # utilisation_ratio: mean bill / credit limit, clipped to [0, 5]
    out["utilisation_ratio"] = (avg_bill / limit_bal).clip(0, 5)

    # payment_ratio: last month payment / last month bill, clipped to [0, 5]
    bill_amt1 = out["BILL_AMT1"].replace(0, np.nan)
    out["payment_ratio"] = (out["PAY_AMT1"] / bill_amt1).clip(0, 5)

    # max_delay: the worst payment delay status across 6 months
    out["max_delay"] = out[PAY_FEATURES].max(axis=1)

    # n_months_delayed: count of months where payment was delayed (PAY_* > 0)
    out["n_months_delayed"] = (out[PAY_FEATURES] > 0).sum(axis=1)

    # avg_bill_amt: average bill amount
    out["avg_bill_amt"] = avg_bill

    # avg_pay_amt: average payment amount
    out["avg_pay_amt"] = out[PAY_AMT_FEATURES].mean(axis=1)

    return out


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Unfitted transformer: one-hot the categoricals, impute and scale the rest."""
    # Which categorical columns are actually present in the feature list
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    num_cols = [c for c in feature_columns if c not in cat_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if cat_cols:
        transformers.append(("categorical", categorical_pipeline, cat_cols))
    if num_cols:
        transformers.append(("numerical", numeric_pipeline, num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """The full feature step: derive, then drop nothing and let the model decide."""
    return add_derived_features(df)
