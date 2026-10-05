"""Single MLflow backend for this project (SQLite + local artifacts).

Configures MLflow tracking with local SQLite database and file-based artifact storage.
Handles experiment initialization and cross-machine compatibility checks.
"""
import logging

import mlflow
from mlflow.exceptions import MlflowException

from src.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_ARTIFACTS_URI,
    MLFLOW_DB,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)


def _normalize_artifact_uri(uri: str) -> str:
    """Normalize artifact URI for comparison.

    Args:
        uri: Raw artifact URI string.

    Returns:
        Normalized URI without file:// prefix and trailing slashes.
    """
    return uri.replace("\\", "/").rstrip("/").removeprefix("file").removeprefix("//")


def _artifact_uri_matches(actual: str, expected: str) -> bool:
    """Check if artifact URIs match.

    Handles differences in path separators and file:// prefixes.

    Args:
        actual: Actual artifact URI.
        expected: Expected artifact URI.

    Returns:
        True if URIs represent the same location.
    """
    actual_norm = _normalize_artifact_uri(actual)
    expected_norm = _normalize_artifact_uri(expected)
    return actual_norm == expected_norm or actual_norm.endswith("/artifacts/mlartifacts")


def _reset_tracking_store() -> None:
    """Drop local SQLite store when paths belong to another machine/folder.

    This ensures compatibility when moving the project directory or switching machines.
    """
    if MLFLOW_DB.exists():
        logger.warning(f"Resetting MLflow database at {MLFLOW_DB}")
        MLFLOW_DB.unlink()


def setup_mlflow() -> None:
    """Configure tracking URI and ensure experiment uses project artifact paths.

    Initializes MLflow with:
    - SQLite local tracking database
    - Local file artifact storage
    - Project experiment with consistent settings

    Handles:
    - Database resets for cross-machine compatibility
    - Experiment creation with proper artifact location
    - Duplicate experiment handling

    Raises:
        MlflowException: If critical MLflow configuration fails.

    Example:
        >>> setup_mlflow()
        >>> with mlflow.start_run():
        ...     mlflow.log_metric("loss", 0.5)
    """
    logger.info("Setting up MLflow tracking")

    # Create artifacts directory
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set tracking URI
    logger.debug(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Check existing experiment for cross-machine compatibility
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is not None and not _artifact_uri_matches(
        experiment.artifact_location, MLFLOW_ARTIFACTS_URI
    ):
        logger.warning(
            f"Experiment artifact location mismatch. Resetting database. "
            f"Actual: {experiment.artifact_location}, Expected: {MLFLOW_ARTIFACTS_URI}"
        )
        _reset_tracking_store()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create experiment if not exists
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        try:
            logger.info(f"Creating experiment: {MLFLOW_EXPERIMENT_NAME}")
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_URI,
            )
        except MlflowException as exc:
            if "already exists" not in str(exc).lower():
                logger.error(f"Failed to create MLflow experiment: {exc}")
                raise
            logger.warning(f"Experiment already exists, resetting to fix artifact location")
            _reset_tracking_store()
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_URI,
            )

    # Set active experiment
    logger.debug(f"Setting active experiment: {MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow setup completed successfully")
