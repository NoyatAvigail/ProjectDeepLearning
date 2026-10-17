"""הערכה סופית. Test משמש כאן בלבד, אחרי שכל ההחלטות כבר התקבלו על Validation."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import NUM_CLASSES, REPORTS_DIR


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def save_evaluation(
    experiment_id: str,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
    class_names: dict[int, str],
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    metrics = metrics_from_predictions(y_true_arr, y_pred_arr)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(NUM_CLASSES)),
        target_names=[class_names[i] for i in range(NUM_CLASSES)],
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(NUM_CLASSES)), normalize="true")
    payload = {
        "experiment_id": experiment_id,
        "split": split_name,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix_normalized": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    out = REPORTS_DIR / f"{experiment_id}_{split_name}_eval.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def improvement_vs_baseline(baseline: float, current: float) -> dict:
    """
    שיפור יחסי: (current - baseline) / baseline.
    הפרש בנקודות אחוז: (current - baseline) * 100 כאשר המדד בטווח 0-1.
    אין כאן עיגול 'מייפה' שמסתיר כישלון ביעד ה-15%.
    """
    if baseline <= 0:
        relative = None
    else:
        relative = (current - baseline) / baseline * 100.0
    points = (current - baseline) * 100.0
    return {
        "baseline": baseline,
        "current": current,
        "relative_percent": relative,
        "absolute_percentage_points": points,
        "meets_15_percent_relative": relative is not None and relative >= 15.0,
    }
