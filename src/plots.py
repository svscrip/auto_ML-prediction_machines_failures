"""Generate evaluation plots for artifacts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_feature_importance(importances: dict, path: Path, top_n: int = 15) -> None:
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_monitoring_dashboard(
    path: Path,
    metrics: dict,
    infrastructure: dict | None = None,
    drift_features: dict | None = None,
) -> None:
    """Static monitoring dashboard (metrics, CPU/RAM, PSI) for README and reports."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("ML Pipeline Monitoring Dashboard", fontsize=14, fontweight="bold")

    metric_names = ["roc_auc", "recall", "precision", "f1"]
    metric_values = [metrics.get(name, 0) for name in metric_names]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    axes[0, 0].bar(metric_names, metric_values, color=colors)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Model Metrics (validation)")
    axes[0, 0].set_ylabel("Score")
    for i, value in enumerate(metric_values):
        axes[0, 0].text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    train_time = metrics.get("train_time_sec", 0)
    infra = infrastructure or {}
    before = infra.get("before", {})
    after = infra.get("after", {})
    labels = ["Train time (s)", "CPU before %", "CPU after %", "RAM after %"]
    values = [
        train_time,
        before.get("cpu_percent", 0),
        after.get("cpu_percent", 0),
        after.get("ram_used_percent", 0),
    ]
    axes[0, 1].bar(labels, values, color="steelblue")
    axes[0, 1].set_title("Infrastructure (training run)")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for i, value in enumerate(values):
        axes[0, 1].text(i, value + max(values) * 0.02, f"{value:.1f}", ha="center", fontsize=8)

    if drift_features:
        features = list(drift_features.keys())
        psi_values = [drift_features[f]["psi"] for f in features]
        bar_colors = [
            "#2ecc71" if v < 0.1 else "#f1c40f" if v < 0.25 else "#e74c3c"
            for v in psi_values
        ]
        axes[1, 0].barh(features, psi_values, color=bar_colors)
        axes[1, 0].axvline(0.1, color="orange", linestyle="--", linewidth=1, label="warning 0.1")
        axes[1, 0].axvline(0.25, color="red", linestyle="--", linewidth=1, label="critical 0.25")
        axes[1, 0].set_xlabel("PSI")
        axes[1, 0].set_title("Data Drift (train → test)")
        axes[1, 0].legend(fontsize=8)

    importance = metrics.get("feature_importance_top5") or {}
    if importance:
        names = list(importance.keys())
        imp_values = list(importance.values())
        axes[1, 1].barh(names[::-1], imp_values[::-1], color="#16a085")
        axes[1, 1].set_xlabel("Importance")
        axes[1, 1].set_title("Top-5 Features")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "No feature importance", ha="center", va="center")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_drift_chart(drift_features: dict, path: Path) -> None:
    """PSI drift chart for inference monitoring."""
    features = list(drift_features.keys())
    psi_values = [drift_features[f]["psi"] for f in features]
    colors = ["#2ecc71" if v < 0.1 else "#f1c40f" if v < 0.25 else "#e74c3c" for v in psi_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, psi_values, color=colors)
    ax.axvline(0.1, color="orange", linestyle="--", label="PSI warning (0.1)")
    ax.axvline(0.25, color="red", linestyle="--", label="PSI critical (0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift: train → test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
