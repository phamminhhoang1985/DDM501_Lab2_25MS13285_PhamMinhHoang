"""
Data validation stage — the quality gate in front of training.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from pipeline.config import (
    MAX_MISSING_FRACTION,
    MAX_POSITIVE_RATE,
    MIN_POSITIVE_RATE,
    MIN_ROWS,
    RAW_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails a check that must not be ignored."""


# Value domains, from the dataset documentation. (PROVIDED)
DOMAINS: Dict[str, Any] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
RANGES: Dict[str, tuple] = {
    "LIMIT_BAL": (10_000, 2_000_000),
    "AGE": (18, 100),
    **{c: (-2, 8) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
}


def validate_schema(df: pd.DataFrame) -> List[str]:
    """Level 1 — are the expected columns present, with usable types?"""
    errors = []
    expected = RAW_FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    # Only check dtype for columns that are actually present
    for col in expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"column '{col}' is not numeric (dtype={df[col].dtype})")
    return errors


def validate_statistics(df: pd.DataFrame) -> List[str]:
    """Level 2 — is the shape of the data what training assumes?"""
    errors = []
    if len(df) < MIN_ROWS:
        errors.append(f"too few rows: {len(df)} < {MIN_ROWS}")
    for col in df.columns:
        missing_frac = df[col].isna().mean()
        if missing_frac > MAX_MISSING_FRACTION:
            errors.append(
                f"column '{col}' has {missing_frac:.2%} missing values (max {MAX_MISSING_FRACTION:.2%})"
            )
    if TARGET in df.columns:
        pos_rate = df[TARGET].mean()
        if pos_rate < MIN_POSITIVE_RATE:
            errors.append(
                f"target positive rate {pos_rate:.4f} is below minimum {MIN_POSITIVE_RATE}"
            )
        if pos_rate > MAX_POSITIVE_RATE:
            errors.append(
                f"target positive rate {pos_rate:.4f} exceeds maximum {MAX_POSITIVE_RATE}"
            )
    return errors


def validate_semantics(df: pd.DataFrame) -> List[str]:
    """Level 3 — do the values mean what the business says they mean?"""
    errors = []
    # Check domain constraints (allowed categorical values)
    for col, allowed in DOMAINS.items():
        if col not in df.columns:
            continue
        # Only check non-null values
        actual = set(df[col].dropna().unique())
        invalid = actual - allowed
        if invalid:
            errors.append(
                f"column '{col}' contains out-of-domain values: {sorted(invalid)} (allowed: {sorted(allowed)})"
            )
    # Check range constraints
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        out_of_range = ((col_data < lo) | (col_data > hi)).sum()
        if out_of_range > 0:
            errors.append(
                f"column '{col}' has {out_of_range} values outside [{lo}, {hi}]"
            )
    # No negative payment amounts
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]
    for col in pay_amt_cols:
        neg_count = (df[col].dropna() < 0).sum()
        if neg_count > 0:
            errors.append(f"column '{col}' has {neg_count} negative values")
    return errors


def validate_dataset(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, Any]:
    """Run all three levels and return a report."""
    schema_errors = validate_schema(df)
    statistical_errors = validate_statistics(df)
    semantic_errors = validate_semantics(df)

    all_errors = schema_errors + statistical_errors + semantic_errors
    for err in all_errors:
        logger.error("Validation error: %s", err)

    report = {
        "passed": len(all_errors) == 0,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "schema_errors": schema_errors,
        "statistical_errors": statistical_errors,
        "semantic_errors": semantic_errors,
        "n_errors": len(all_errors),
    }

    if all_errors and raise_on_error:
        raise DataValidationError(
            f"Dataset failed validation with {len(all_errors)} error(s): {all_errors}"
        )

    return report
