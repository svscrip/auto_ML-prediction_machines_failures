"""Project configuration for ML pipeline.

This module contains all configuration constants for data paths, feature definitions,
model hyperparameters, and monitoring thresholds.
"""
from pathlib import Path

# ============ Directory Configuration ============
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Root directory of the project."""

DATA_DIR: Path = PROJECT_ROOT / "keis7-main"
"""Directory containing raw data (train.csv, test.csv)."""

TRAIN_CSV: Path = DATA_DIR / "train.csv"
"""Path to training data file."""

TEST_CSV: Path = DATA_DIR / "test.csv"
"""Path to test data file."""

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
"""Directory for model artifacts, metrics, and logs."""

# ============ MLflow Configuration ============
MLFLOW_DB: Path = ARTIFACTS_DIR / "mlflow.db"
"""Path to MLflow SQLite database."""

MLFLOW_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "mlartifacts"
"""Directory for MLflow artifacts storage."""

MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
"""MLflow tracking URI for local SQLite backend."""

MLFLOW_ARTIFACTS_URI: str = f"file:///{MLFLOW_ARTIFACTS_DIR.resolve().as_posix()}"
"""MLflow artifacts URI for local file storage."""

MLFLOW_EXPERIMENT_NAME: str = "machine_failure_prediction"
"""Name of MLflow experiment."""

# ============ Column Names ============
TARGET_COL: str = "Machine failure"
"""Target column name for binary classification."""

ID_COL: str = "id"
"""Sample identifier column."""

PRODUCT_ID_COL: str = "Product ID"
"""Product identifier for grouping data."""

TYPE_COL: str = "Type"
"""Equipment type column (categorical feature)."""

# ============ Feature Groups ============
FAILURE_FLAGS: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]
"""Binary failure type flags:
- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure
- RNF: Random Failures
"""

RAW_NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
"""Original numeric features from the dataset."""

ENGINEERED_FEATURES: list[str] = [
    "delta_temperature [K]",
    "Power [kW]",
    "air_mass",
    "air_heat_power [kW]",
    "efficiency [%]",
    "total_failures_cum",
]
"""Engineered features created during ETL transformation."""

MODEL_FEATURES: list[str] = (
    RAW_NUMERIC_FEATURES + FAILURE_FLAGS + ENGINEERED_FEATURES + [TYPE_COL]
)
"""All features used for model training."""

CAT_FEATURES: list[str] = [TYPE_COL, *FAILURE_FLAGS]
"""Categorical features for CatBoost model."""

# ============ Model Hyperparameters ============
CATBOOST_PARAMS: dict[str, int | float | str] = {
    "iterations": 500,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "auto_class_weights": "Balanced",
}
"""CatBoost classifier hyperparameters for training."""

# ============ Training Configuration ============
TRAIN_TEST_SIZE: float = 0.2
"""Validation set split ratio."""

RANDOM_STATE: int = 42
"""Random seed for reproducibility."""

SMOKE_SAMPLE_SIZE: int = 5000
"""Sample size for smoke tests."""

PREDICTION_THRESHOLD: float = 0.5
"""Default probability threshold for binary classification."""

# ============ Monitoring Thresholds ============
RISK_THRESHOLDS: dict[str, float] = {
    "low": 0.45,
    "medium": 0.50,
}
"""Risk level thresholds based on failure probability.

Thresholds:
- Low: probability < 0.45
- Medium: 0.45 <= probability < 0.50
- High: probability >= 0.50
"""
