"""Extract: load and validate CSV data.

This module handles data loading with schema validation to ensure data integrity
before passing to feature engineering and model training stages.
"""
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    FAILURE_FLAGS,
    ID_COL,
    PRODUCT_ID_COL,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
    TYPE_COL,
)

logger = logging.getLogger(__name__)


def _required_columns(require_target: bool) -> list[str]:
    """
    Get list of required columns based on dataset type.

    Args:
        require_target: If True, includes target column in requirements.

    Returns:
        List of required column names.
    """
    cols: list[str] = (
        [ID_COL, PRODUCT_ID_COL, TYPE_COL]
        + RAW_NUMERIC_FEATURES
        + FAILURE_FLAGS
    )
    if require_target:
        cols.append(TARGET_COL)
    return cols


def load_train(path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate training dataset.

    Args:
        path: Path to training CSV file. Defaults to configured TRAIN_CSV.

    Returns:
        Validated training DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If schema validation fails.

    Example:
        >>> df = load_train()
        >>> print(f"Loaded {len(df)} training samples")
    """
    path = path or TRAIN_CSV
    if not path.exists():
        logger.error(f"Training file not found: {path}")
        raise FileNotFoundError(f"Training file not found: {path}")

    logger.info(f"Loading training data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    validate_schema(df, require_target=True)
    logger.debug(f"Training schema validation passed")
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    """
    Load and validate test dataset.

    Args:
        path: Path to test CSV file. Defaults to configured TEST_CSV.

    Returns:
        Validated test DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If schema validation fails.

    Example:
        >>> df = load_test()
        >>> print(f"Loaded {len(df)} test samples")
    """
    path = path or TEST_CSV
    if not path.exists():
        logger.error(f"Test file not found: {path}")
        raise FileNotFoundError(f"Test file not found: {path}")

    logger.info(f"Loading test data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    validate_schema(df, require_target=False)
    logger.debug(f"Test schema validation passed")
    return df


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """
    Validate DataFrame against required schema.

    Checks for:
    - Required columns presence
    - Target column non-null values (if required)
    - Numeric features non-null values
    - Categorical features non-null values

    Args:
        df: DataFrame to validate.
        require_target: If True, validates target column.

    Raises:
        ValueError: If validation fails with descriptive message.

    Example:
        >>> validate_schema(df, require_target=True)
        >>> print("Schema is valid")
    """
    logger.debug("Starting schema validation")

    # Check required columns
    required = _required_columns(require_target)
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing columns: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    # Check target column
    if require_target and df[TARGET_COL].isna().any():
        msg = f"Null values in target column {TARGET_COL}"
        logger.error(msg)
        raise ValueError(msg)

    # Check numeric features
    for col in RAW_NUMERIC_FEATURES + ([TARGET_COL] if require_target else []):
        if col in df.columns and df[col].isna().any():
            msg = f"Null values in column {col}"
            logger.error(msg)
            raise ValueError(msg)

    # Check Type column
    if TYPE_COL in df.columns and df[TYPE_COL].isna().any():
        msg = f"Null values in column {TYPE_COL}"
        logger.error(msg)
        raise ValueError(msg)

    # Check failure flags
    for col in FAILURE_FLAGS:
        if col in df.columns and df[col].isna().any():
            msg = f"Null values in failure flag column {col}"
            logger.error(msg)
            raise ValueError(msg)

    logger.debug("Schema validation passed successfully")
