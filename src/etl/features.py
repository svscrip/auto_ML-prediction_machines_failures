"""Transform: feature engineering from case7.ipynb.

This module implements feature engineering transformations including:
- Cleaning contradictory rows
- Deriving physical features from raw measurements
- Computing aggregated statistics
"""
import logging

import pandas as pd

from src.config import (
    ENGINEERED_FEATURES,
    FAILURE_FLAGS,
    MODEL_FEATURES,
    PRODUCT_ID_COL,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    TYPE_COL,
)

logger = logging.getLogger(__name__)


def clean_contradictory_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with contradictory target labels.

    Removes rows where Machine failure=0 but failure flags are active,
    indicating data quality issues.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame with contradictory rows removed.

    Example:
        >>> df_clean = clean_contradictory_rows(df)
        >>> print(f"Removed {len(df) - len(df_clean)} contradictory rows")
    """
    mask = (df[TARGET_COL] == 0) & (df[FAILURE_FLAGS].sum(axis=1) > 0)
    removed_count = mask.sum()
    logger.info(f"Removed {removed_count} contradictory rows")
    return df.loc[~mask].copy()


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply ETL transformations and engineered features.

    Transformations:
    1. Clean contradictory rows (training only)
    2. Compute delta_temperature = Process - Air temperature
    3. Compute Power = Torque × RPM / 9550
    4. Compute air properties (mass, heat power)
    5. Compute efficiency = Power / (Power + air_heat_power)
    6. Compute cumulative failure count by equipment

    Args:
        df: Input DataFrame.
        is_train: If True, applies data cleaning. Defaults to True.

    Returns:
        DataFrame with all engineered features.

    Example:
        >>> df_features = build_features(df, is_train=True)
        >>> print(f"Created {len(ENGINEERED_FEATURES)} engineered features")
    """
    logger.debug(f"Starting feature engineering for {len(df)} rows")
    out = df.copy()

    # Step 1: Clean training data
    if is_train:
        out = clean_contradictory_rows(out)

    # Step 2: Temperature difference
    logger.debug("Computing delta_temperature")
    out["delta_temperature [K]"] = (
        out["Process temperature [K]"] - out["Air temperature [K]"]
    )

    # Step 3: Mechanical power
    logger.debug("Computing Power [kW]")
    out["Power [kW]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] / 9550

    # Step 4: Air mass and thermal properties
    logger.debug("Computing air properties")
    out["air_mass"] = 14.4 * out["Power [kW]"]
    out["air_heat_power [kW]"] = (
        out["air_mass"] * 1.005 * out["delta_temperature [K]"] / 60
    )

    # Step 5: Efficiency
    logger.debug("Computing efficiency")
    denominator = out["air_heat_power [kW]"] + out["Power [kW]"]
    out["efficiency [%]"] = (out["Power [kW]"] / denominator.replace(0, pd.NA)) * 100
    out["efficiency [%]"] = out["efficiency [%]"].fillna(0.0).clip(0.0, 100.0)

    # Step 6: Failure flags and cumulative failure count
    logger.debug("Computing cumulative failure statistics")
    flags = out[FAILURE_FLAGS].fillna(0).astype(int)
    out[FAILURE_FLAGS] = flags

    out = out.sort_values([TYPE_COL, PRODUCT_ID_COL, "Tool wear [min]"])
    out["failures_sum"] = flags.sum(axis=1)

    # Cumulative sum by equipment type and product ID
    cumsum = out.groupby([TYPE_COL, PRODUCT_ID_COL], observed=True)[
        "failures_sum"
    ].cumsum()
    # Subtract current row to count only past failures
    out["total_failures_cum"] = cumsum - out["failures_sum"]
    out = out.drop(columns=["failures_sum"])

    logger.info(f"Feature engineering completed: {len(ENGINEERED_FEATURES)} features created")
    return out.reset_index(drop=True)


def get_feature_matrix(
    df: pd.DataFrame, include_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Extract feature matrix and optional target vector.

    Ensures all required model features are present and properly typed.

    Args:
        df: DataFrame with engineered features.
        include_target: If True, returns target vector. Defaults to True.

    Returns:
        Tuple of (feature matrix X, target vector y or None).

    Raises:
        ValueError: If required features are missing.

    Example:
        >>> X, y = get_feature_matrix(df, include_target=True)
        >>> print(f"Features shape: {X.shape}")
    """
    logger.debug(f"Building feature matrix from {len(df)} rows")

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        msg = f"Features not built: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    X = df[MODEL_FEATURES].copy()

    # Type conversion
    for col in FAILURE_FLAGS:
        X[col] = X[col].astype(int)
    X[TYPE_COL] = X[TYPE_COL].astype(str)

    logger.debug(f"Feature matrix created: shape {X.shape}")

    y = None
    if include_target and TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)
        logger.debug(f"Target vector extracted: {y.sum()} positive samples")

    return X, y
