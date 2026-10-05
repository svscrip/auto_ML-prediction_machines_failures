"""Train CatBoost model with MLflow tracking.

This module orchestrates the complete training pipeline including:
- Data loading and feature engineering
- Train/validation split
- Model training with hyperparameter tuning
- Metrics computation and visualization
- MLflow experiment tracking
- Model artifact storage
"""
import argparse
import json
import logging
import time
from pathlib import Path

import mlflow
import mlflow.catboost
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CATBOOST_PARAMS,
    CAT_FEATURES,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    SMOKE_SAMPLE_SIZE,
    TARGET_COL,
    TRAIN_TEST_SIZE,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_train
from src.logger import setup_logging
from src.mlflow_setup import setup_mlflow
from src.monitoring import (
    build_training_monitoring_summary,
    compute_data_quality_report,
    infrastructure_snapshot,
    save_monitoring_report,
)
from src.plots import (
    save_confusion_matrix,
    save_feature_importance,
    save_infrastructure_chart,
    save_model_metrics_chart,
    save_roc_curve,
)

logger = logging.getLogger(__name__)


def _get_categorical_feature_indices(
    feature_columns: pd.Index, categorical_features: list[str]
) -> list[int]:
    """Get indices of categorical features in the feature matrix.

    Args:
        feature_columns: Column index from feature DataFrame.
        categorical_features: List of categorical feature names.

    Returns:
        List of column indices for categorical features.
    """
    return [feature_columns.get_loc(c) for c in categorical_features if c in feature_columns]


def _compute_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities.

    Returns:
        Dictionary of metric names to float values.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_model(
    sample_size: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Train CatBoost model with comprehensive MLflow logging.

    Performs end-to-end model training with:
    1. Data loading and validation
    2. Feature engineering
    3. Train/validation split
    4. Model training with early stopping
    5. Metrics computation
    6. Artifact generation and logging

    Args:
        sample_size: Number of samples to use. If None, uses full dataset.
            For smoke tests, use SMOKE_SAMPLE_SIZE.
        output_dir: Directory for model artifacts. Defaults to ARTIFACTS_DIR.
        verbose: If True, logs detailed training progress. Defaults to True.

    Returns:
        Dictionary of training metrics.

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data validation fails.

    Example:
        >>> metrics = train_model(sample_size=5000)
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    """
    logger.info("=" * 80)
    logger.info("Starting model training pipeline")
    logger.info("=" * 80)

    output_dir = output_dir or ARTIFACTS_DIR
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"MLflow tracking enabled: SQLite backend")

    setup_mlflow()

    # ========== Data Loading ==========
    logger.info("Loading training data")
    df = load_train()
    logger.info(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if sample_size:
        logger.info(f"Sampling {sample_size} rows for training")
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)
        logger.info(f"Using {len(df)} samples")

    # ========== Feature Engineering ==========
    logger.info("Building features")
    df = build_features(df, is_train=True)
    logger.info(f"Features created: {len(df)} rows after cleaning")

    X, y = get_feature_matrix(df, include_target=True)
    logger.info(f"Feature matrix: X.shape={X.shape}, target class distribution: {y.value_counts().to_dict()}")

    # ========== Train/Validation Split ==========
    logger.info(f"Splitting data: train={1-TRAIN_TEST_SIZE:.0%}, validation={TRAIN_TEST_SIZE:.0%}")
    cat_idx = _get_categorical_feature_indices(X.columns, CAT_FEATURES)
    logger.debug(f"Categorical feature indices: {cat_idx}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Train set: {len(X_train)} samples, {y_train.sum()} positive"
    )
    logger.info(
        f"Validation set: {len(X_val)} samples, {y_val.sum()} positive"
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    # ========== Model Training ==========
    infra_before = infrastructure_snapshot()
    t0 = time.perf_counter()

    logger.info("Starting CatBoost training")
    logger.debug(f"Hyperparameters: {CATBOOST_PARAMS}")

    with mlflow.start_run(run_name="catboost_train"):
        # Log parameters
        for key, value in CATBOOST_PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_param("sample_size", sample_size or len(df))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        # Train model
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        logger.info("Model training completed")

        train_time = time.perf_counter() - t0
        logger.info(f"Training time: {train_time:.2f} seconds")
        mlflow.log_metric("train_time_sec", train_time)

        # ========== Metrics Computation ==========
        logger.info("Computing validation metrics")
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= PREDICTION_THRESHOLD).astype(int)

        metrics = _compute_classification_metrics(y_val, y_pred, y_proba)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        metrics["train_time_sec"] = train_time

        logger.info(f"Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        # ========== Infrastructure Monitoring ==========
        logger.info("Capturing infrastructure metrics")
        infra_after = infrastructure_snapshot()
        mlflow.log_metric("cpu_percent_before", infra_before["cpu_percent"])
        mlflow.log_metric("cpu_percent_after", infra_after["cpu_percent"])
        mlflow.log_metric("ram_used_percent_before", infra_before["ram_used_percent"])
        mlflow.log_metric("ram_used_percent_after", infra_after["ram_used_percent"])
        logger.debug(
            f"CPU: {infra_before['cpu_percent']:.1f}% → {infra_after['cpu_percent']:.1f}%"
        )
        logger.debug(
            f"RAM: {infra_before['ram_used_percent']:.1f}% → {infra_after['ram_used_percent']:.1f}%"
        )

        # ========== Feature Importance ==========
        logger.info("Computing feature importance")
        importance = dict(
            zip(X.columns, model.get_feature_importance().tolist())
        )
        metrics["feature_importance_top5"] = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        logger.debug(f"Top 5 features: {metrics['feature_importance_top5']}")

        # ========== Artifact Saving ==========
        logger.info("Saving model and artifacts")
        model_path = output_dir / "model.cbm"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))
        logger.info(f"Model saved: {model_path}")

        # Metrics JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))
        logger.debug(f"Metrics saved: {metrics_path}")

        # ========== Visualizations ==========
        logger.info("Generating visualizations")
        save_confusion_matrix(y_val, y_pred, plots_dir / "confusion_matrix.png")
        save_roc_curve(y_val, y_proba, plots_dir / "roc_curve.png")
        save_feature_importance(importance, plots_dir / "feature_importance.png")
        save_model_metrics_chart(plots_dir / "model_metrics.png", metrics)
        save_infrastructure_chart(
            plots_dir / "infrastructure_training.png",
            stage="training",
            before=infra_before,
            after=infra_after,
            duration_sec=train_time,
            duration_label="Training",
        )

        for plot in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(plot))
        logger.info(f"Plots saved to: {plots_dir}")

        # ========== Monitoring Reports ==========
        logger.info("Generating monitoring reports")
        data_report = compute_data_quality_report(df, "train_processed")
        data_report["infrastructure"] = {
            "before": infra_before,
            "after": infra_after,
        }
        save_monitoring_report(data_report, output_dir / "data_quality.json")
        mlflow.log_artifact(str(output_dir / "data_quality.json"))

        monitoring_summary = build_training_monitoring_summary(metrics, data_report)
        summary_path = output_dir / "monitoring_summary.json"
        save_monitoring_report(monitoring_summary, summary_path)
        mlflow.log_artifact(str(summary_path))
        logger.info(f"Reports saved to: {output_dir}")

        # MLflow model logging
        mlflow.catboost.log_model(model, "model")
        logger.info(f"MLflow run completed: {mlflow.active_run().info.run_id}")

    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 80)

    return metrics


def main() -> None:
    """Command-line interface for model training.

    Supports:
    - Full dataset training: `python -m src.train`
    - Sample training: `python -m src.train --smoke`
    - Custom size: `python -m src.train --sample-size 1000`
    """
    parser = argparse.ArgumentParser(
        description="Train machine failure prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.train              # Train on full dataset
  python -m src.train --smoke      # Train on 5000 samples
  python -m src.train --sample-size 10000  # Train on 10000 samples
        """,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use (default: full dataset)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Quick smoke test with {SMOKE_SAMPLE_SIZE} samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose logging (default: True)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="DEBUG" if args.verbose else "INFO")

    sample = args.sample_size or (SMOKE_SAMPLE_SIZE if args.smoke else None)
    metrics = train_model(sample_size=sample, verbose=args.verbose)
    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
