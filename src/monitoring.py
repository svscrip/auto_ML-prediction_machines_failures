import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.config import RAW_NUMERIC_FEATURES, TARGET_COL


def compute_data_quality_report(df: pd.DataFrame, label: str = "train") -> dict:
    """
    Формирует отчет о качестве данных переданного датасета.
    
    Возвращает информацию о количестве строк в датасете (rows), количество пустых значений (null_counts),
    арифмитическое среднее и стандартное отклонение для колонок с числовым типом данных.
    """
    report = {
        "dataset": label,
        "rows": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "target_rate": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None,
        "type_distribution": df["Type"].value_counts().to_dict() if "Type" in df.columns else {},
    }
    for col in RAW_NUMERIC_FEATURES:
        if col in df.columns:
            report[f"mean_{col}"] = float(df[col].mean())
            report[f"std_{col}"] = float(df[col].std())
    return report


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """
    Рассчитывает индекс стабильности популяции (PSI) для оценки смещения распределений.

    PSI (Population Stability Index) измеряет, насколько распределение фактических данных
    (например, за текущий период) отличается от ожидаемого распределения (например, за базовый период).
    Метрика широко используется в кредитном скоринге и мониторинге моделей машинного обучения
    для обнаружения дрейфа признаков.
    
    Интерпретация результатов (эмпирическое правило):
        - PSI < 0.1   : Распределения практически идентичны (изменений нет).
        - 0.1 <= PSI < 0.25 : Небольшой сдвиг (требуется внимание).
        - PSI >= 0.25  : Значительный сдвиг (распределение изменилось кардинально, модель требует переобучения).
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = pd.cut(expected, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    actual_pct = pd.cut(actual, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    aligned = pd.concat([expected_pct, actual_pct], axis=1, join="outer").fillna(0.0001)
    aligned.columns = ["expected", "actual"]
    psi = ((aligned["actual"] - aligned["expected"]) * np.log(
        aligned["actual"] / aligned["expected"]
    )).sum()
    return float(psi)


def compare_distributions(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Выполняет сравнение распределений целевого датасета с новым датасетом
    с помощью разницы между средними значениями соответствующих колонок и метрики PSI
    """
    drift = {}
    for col in RAW_NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        drift[col] = {
            "mean_shift": float(current[col].mean() - reference[col].mean()),
            "psi": population_stability_index(reference[col], current[col]),
        }
    return drift


def infrastructure_snapshot() -> dict:
    """
    Возвращает информацию об используемых ресурсах
    """
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_percent": mem.percent,
        "swap_memory_used": psutil.swap_memory().used,
        "swap_memory_free": psutil.swap_memory().free
    }


def save_monitoring_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
