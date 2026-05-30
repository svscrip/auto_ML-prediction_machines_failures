"""Inference and business recommendations."""
import argparse
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    PRODUCT_ID_COL,
    RISK_THRESHOLDS,
    TARGET_COL,
    TYPE_COL,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_test, load_train
from src.monitoring import (
    build_inference_monitoring_summary,
    compare_distributions,
    compute_data_quality_report,
    infrastructure_snapshot,
    save_monitoring_report,
)
from src.plots import save_drift_chart, save_infrastructure_chart


def assign_risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["medium"]:
        return "Средний"
    if probability >= RISK_THRESHOLDS["low"]:
        return "Низкий"
    return "Низкий"


def assign_risk_level_vector(probs: pd.Series) -> pd.Series:
    q50 = probs.quantile(0.50)
    q90 = probs.quantile(0.90)
    return pd.cut(
        probs,
        bins=[-0.01, q50, q90, 1.01],
        labels=["Низкий", "Средний", "Высокий"],
    ).astype(str)


def predict(
    model_path: Path | None = None,
    output_dir: Path | None = None,
    use_train_for_drift: bool = True,
) -> pd.DataFrame:
    output_dir = output_dir or ARTIFACTS_DIR
    model_path = model_path or output_dir / "model.cbm"
    pipeline_t0 = time.perf_counter()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train"
        )

    test_raw = load_test()
    test_df = build_features(test_raw, is_train=False)
    X, _ = get_feature_matrix(test_df, include_target=False)

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    pool = Pool(X, cat_features=cat_idx)

    infra_before = infrastructure_snapshot()
    inference_t0 = time.perf_counter()
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probabilities = model.predict_proba(pool)[:, 1]
    inference_time_sec = time.perf_counter() - inference_t0
    infra_after = infrastructure_snapshot()

    result = test_df[[c for c in ["id", PRODUCT_ID_COL, TYPE_COL] if c in test_df.columns]].copy()
    if "id" not in result.columns:
        result = test_df[[PRODUCT_ID_COL, TYPE_COL]].copy()

    result["failure_probability"] = probabilities
    for col in ["efficiency [%]", "Tool wear [min]", "delta_temperature [K]", "Power [kW]", "air_mass"]:
        if col in test_df.columns:
            result[col] = test_df[col].values
    if "Power [kW]" in result.columns:
        result["mechanical_power [kW]"] = result["Power [kW]"]
    if "air_mass" in result.columns:
        result["air_mass [kg/s]"] = result["air_mass"]

    result["risk_level"] = assign_risk_level_vector(pd.Series(probabilities))

    predictions_path = output_dir / "predictions.csv"
    result.to_csv(predictions_path, index=False)

    recommendations = build_recommendations(result, test_df)
    rec_path = output_dir / "maintenance_recommendations.csv"
    recommendations.to_csv(rec_path, index=False)

    if use_train_for_drift:
        train_df = build_features(load_train(), is_train=True)
        drift = compare_distributions(train_df, test_df)
        test_quality = compute_data_quality_report(test_df, "test")
        pipeline_time_sec = time.perf_counter() - pipeline_t0
        high_risk = int((result["risk_level"] == "Высокий").sum())
        infra = {
            "before": infra_before,
            "after": infra_after,
        }
        drift_eval = build_inference_monitoring_summary(
            predictions_rows=len(result),
            high_risk_count=high_risk,
            drift=drift,
            test_quality=test_quality,
            infrastructure=infra,
            inference_time_sec=inference_time_sec,
            pipeline_time_sec=pipeline_time_sec,
        )
        report_path = output_dir / "inference_monitoring.json"
        save_monitoring_report(drift_eval, report_path)

        plots_dir = output_dir / "plots"
        if drift_eval["drift"]["features"]:
            save_drift_chart(drift_eval["drift"]["features"], plots_dir / "drift_psi.png")
        save_infrastructure_chart(
            plots_dir / "infrastructure_inference.png",
            stage="inference",
            before=infra_before,
            after=infra_after,
            duration_sec=inference_time_sec,
            duration_label="Inference",
        )

    return result


def build_recommendations(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Generate maintenance recommendations from prediction batch."""
    rows = []

    if "efficiency [%]" in predictions.columns:
        low_eff = (predictions["efficiency [%]"] < 50).mean()
        if low_eff > 0.1:
            rows.append(
                {
                    "Проблема": "Низкая эффективность системы",
                    "Решение": "Настроить систему при КПД < 50.0%",
                    "Приоритет": "Высокий",
                }
            )

    if "Tool wear [min]" in predictions.columns:
        wear_threshold = predictions["Tool wear [min]"].quantile(0.90)
        high_wear = (predictions["Tool wear [min]"] > wear_threshold).sum()
        if high_wear > 0:
            rows.append(
                {
                    "Проблема": "Критический износ инструмента",
                    "Решение": f"Плановые замены при износе > {wear_threshold:.0f} мин",
                    "Приоритет": "Высокий",
                }
            )

    high_risk = (predictions["risk_level"] == "Высокий").sum()
    if high_risk > 0:
        rows.append(
            {
                "Проблема": "Высокий прогноз отказа",
                "Решение": "Целевое обслуживание оборудования с risk_level=Высокий",
                "Приоритет": "Высокий",
            }
        )

    if not rows:
        rows.append(
            {
                "Проблема": "Стабильное состояние",
                "Решение": "Продолжить мониторинг KPI",
                "Приоритет": "Низкий",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = predict(model_path=args.model, output_dir=args.output_dir)
    print(f"Predictions saved: {len(result)} rows")


if __name__ == "__main__":
    main()
