"""
נקודת הכניסה למחזור החיים המלא:
טעינה → ניקוי → Augmentation (Train בלבד) → אימון 4 ניסויים → הערכה → ויזואליזציה.

הרצה משורש הפרויקט:
    python -m src.run_pipeline
    python -m src.run_pipeline --epochs 3
    python -m src.run_pipeline --skip-train
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.config import (
    BACKBONE_NAME,
    EPOCHS,
    EXPERIMENTS,
    NUM_CLASSES,
    OUTPUT_DIR,
    PRIMARY_METRIC,
    REPORTS_DIR,
    SEED,
)
from src.data_cleaning import clean_splits
from src.data_loading import load_raw_splits
from src.evaluate import improvement_vs_baseline, save_evaluation
from src.reproducibility import seed_everything
from src.visualize import (
    plot_ablation,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_learning_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="צינור סיווג חדרי פנים — Deep Learning")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="מספר epochs לכל ניסוי")
    parser.add_argument("--skip-train", action="store_true", help="רק טעינה, ניקוי וגרפי התפלגות")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="הגבלה לצורך בדיקת צינור (לא להגשה). דוגמאות לכל מחלקה בכל פיצול.",
    )
    parser.add_argument(
        "--eval-split-for-ablation",
        choices=["val", "test"],
        default="val",
        help="הפיצול לגרף ה-Ablation. ברירת המחדל היא val כדי לא לבחור מודל לפי Test.",
    )
    return parser.parse_args()


def _class_names_list(mapping: dict[int, str]) -> list[str]:
    return [mapping[i] for i in range(NUM_CLASSES)]


def _require_images_or_explain(frame: pd.DataFrame) -> None:
    from src.datasets import count_available_images

    available = count_available_images(frame)
    if available == 0:
        message = (
            "\n============================================================\n"
            "לא נמצאו קבצי תמונה (.jpg) בפרויקט.\n"
            "קבצי ה-CSV קיימים ותקינים, אבל האימון דורש את התמונות עצמן.\n\n"
            "המיקום שנבדק בפועל בפרויקט:\n"
            "  Dataset/Pinterest Interior Design Images and Metadata/<room_type>/<style>/<filename>.jpg\n"
            "לדוגמה:\n"
            "  Dataset/Pinterest Interior Design Images and Metadata/kitchen/modern/kitchen_modern_224.jpg\n"
            "ב-CSV רשום גם ../data/raw/... והקוד תומך בשני המבנים.\n"
            "============================================================\n"
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "missing_images.txt").write_text(message, encoding="utf-8")
        raise FileNotFoundError(message)


def _save_cleaning_table(report) -> None:
    rows = []
    for step in report.steps:
        rows.append(
            {
                "step": step["step"],
                "train_before": step["before"].get("train"),
                "val_before": step["before"].get("val"),
                "test_before": step["before"].get("test"),
                "train_after": step["after"].get("train"),
                "val_after": step["after"].get("val"),
                "test_after": step["after"].get("test"),
                "removed": step["removed"],
                "detail": step["detail"],
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "cleaning_step_counts.csv", index=False, encoding="utf-8")


def run_analysis() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    print("=== Loading data step ===")
    raw = load_raw_splits()
    mapping = {
        int(k): v
        for k, v in json.loads((REPORTS_DIR / "label_mapping.json").read_text(encoding="utf-8")).items()
    }
    plot_class_distribution(
        raw["train"],
        "Class distribution in Train before cleaning",
        "class_distribution_train_before_cleaning.png",
    )
    plot_class_distribution(
        pd.concat([raw["train"], raw["val"]], ignore_index=True),
        "Class distribution in Train+Val before cleaning (identifying imbalance)",
        "class_distribution_before_cleaning.png",
    )

    print("=== Data cleaning step ===")
    cleaned, report = clean_splits(raw)
    plot_class_distribution(
        cleaned["train"],
        "Class distribution in Train after cleaning",
        "class_distribution_train_after_cleaning.png",
    )
    plot_class_distribution(
        pd.concat([cleaned["train"], cleaned["val"]], ignore_index=True),
        "Class distribution in Train+Val after cleaning",
        "class_distribution_after_cleaning.png",
    )
    _save_cleaning_table(report)
    return raw, cleaned, mapping


def _subsample_per_class(frame: pd.DataFrame, max_per_class: int | None) -> pd.DataFrame:
    if max_per_class is None:
        return frame
    parts = []
    for _, group in frame.groupby("encoded_label"):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=SEED))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def run_training(args: argparse.Namespace, raw, cleaned, mapping) -> None:
    import torch
    from torch import nn

    from src.datasets import make_loader
    from src.models import build_model, count_trainable_parameters
    from src.reproducibility import get_device
    from src.train import evaluate_loader, train_one_experiment

    _require_images_or_explain(cleaned["train"])
    device = get_device()
    class_names = _class_names_list(mapping)
    results = []

    for exp_id, spec in EXPERIMENTS.items():
        print(f"\n=== Experiment {exp_id}: {spec['title']} ===")
        frames = cleaned if spec["use_cleaned"] else raw
        train_df = _subsample_per_class(frames["train"], args.max_per_class)
        val_df = _subsample_per_class(frames["val"], args.max_per_class)
        test_df = _subsample_per_class(frames["test"], args.max_per_class)
        if args.max_per_class:
            print(
                f"[{exp_id}] Test state: max_per_class={args.max_per_class} "
                f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
            )

        train_loader = make_loader(train_df, augment=spec["use_augmentation"], shuffle=True)
        val_loader = make_loader(val_df, augment=False, shuffle=False)
        test_loader = make_loader(test_df, augment=False, shuffle=False)

        model = build_model(unfreeze_backbone=spec["unfreeze_backbone"])
        print(f"[{exp_id}] backbone={BACKBONE_NAME}, trainable={count_trainable_parameters(model):,}")

        train_out = train_one_experiment(
            experiment_id=exp_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_frame=train_df,
            device=device,
            lr=spec["lr"],
            use_class_weights=spec["use_class_weights"],
            num_classes=NUM_CLASSES,
            epochs=args.epochs,
        )
        plot_learning_curves(train_out["history"], exp_id)

        try:
            checkpoint = torch.load(train_out["best_checkpoint"], map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(train_out["best_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        criterion = nn.CrossEntropyLoss()

        val_eval = evaluate_loader(model, val_loader, criterion, device, NUM_CLASSES)
        test_eval = evaluate_loader(model, test_loader, criterion, device, NUM_CLASSES)
        val_payload = save_evaluation(exp_id, "val", val_eval["y_true"], val_eval["y_pred"], mapping)
        test_payload = save_evaluation(exp_id, "test", test_eval["y_true"], test_eval["y_pred"], mapping)

        plot_confusion_matrix(
            np.array(val_payload["confusion_matrix_normalized"]),
            class_names,
            exp_id,
            "val",
        )
        plot_confusion_matrix(
            np.array(test_payload["confusion_matrix_normalized"]),
            class_names,
            exp_id,
            "test",
        )

        row = {
            "experiment_id": exp_id,
            "title": spec["title"],
            "is_smoke_run": args.max_per_class is not None,
            "max_per_class": args.max_per_class,
            "use_cleaned": spec["use_cleaned"],
            "use_augmentation": spec["use_augmentation"],
            "unfreeze_backbone": spec["unfreeze_backbone"],
            "use_class_weights": spec["use_class_weights"],
            "trainable_params": count_trainable_parameters(model),
            "val_accuracy": val_eval["accuracy"],
            "val_precision_macro": val_eval["precision_macro"],
            "val_recall_macro": val_eval["recall_macro"],
            "val_f1_macro": val_eval["f1_macro"],
            "test_accuracy": test_eval["accuracy"],
            "test_precision_macro": test_eval["precision_macro"],
            "test_recall_macro": test_eval["recall_macro"],
            "test_f1_macro": test_eval["f1_macro"],
        }
        results.append(row)
        print(
            f"[{exp_id}] VAL acc={row['val_accuracy']:.4f} f1={row['val_f1_macro']:.4f} | "
            f"TEST acc={row['test_accuracy']:.4f} f1={row['test_f1_macro']:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "ablation_results.csv", index=False, encoding="utf-8")

    metric_col = f"{args.eval_split_for_ablation}_f1_macro"
    acc_col = f"{args.eval_split_for_ablation}_accuracy"
    plot_ablation(results_df, metric_col, "ablation_f1_macro.png")
    plot_ablation(results_df, acc_col, "ablation_accuracy.png")

    baseline_row = results_df[results_df["experiment_id"] == "exp1_baseline_frozen_raw"].iloc[0]
    final_row = results_df[results_df["experiment_id"] == "exp4_finetune_augmented"].iloc[0]
    comparison = {
        "definition": (
            "המודל הסופי הוגדר מראש כ-exp4 (Fine-Tuning + ניקוי + Augmentation). "
            "ה-Baseline הוא exp1 (EfficientNet-B0 קפוא, נתונים גולמיים, ללא Augmentation). "
            "שיפור יחסי = (final-baseline)/baseline * 100. "
            "הפרש בנקודות אחוז = (final-baseline)*100 כאשר המדד בטווח 0-1. "
            "גרף ה-Ablation נבנה כברירת מחדל על Validation. "
            "מספרי ה-Test הם אומדן לפרוטוקול שנקבע מראש, ולא כלי לבחירת מודל."
        ),
        "primary_metric": PRIMARY_METRIC,
        "validation": {
            "accuracy": improvement_vs_baseline(float(baseline_row["val_accuracy"]), float(final_row["val_accuracy"])),
            "f1_macro": improvement_vs_baseline(float(baseline_row["val_f1_macro"]), float(final_row["val_f1_macro"])),
        },
        "test": {
            "accuracy": improvement_vs_baseline(float(baseline_row["test_accuracy"]), float(final_row["test_accuracy"])),
            "f1_macro": improvement_vs_baseline(float(baseline_row["test_f1_macro"]), float(final_row["test_f1_macro"])),
        },
        "assignment_target": "שיפור יחסי של 15% לפחות ב-Accuracy או ב-F1 לעומת Baseline",
    }
    val_ok = (
        comparison["validation"]["accuracy"]["meets_15_percent_relative"]
        or comparison["validation"]["f1_macro"]["meets_15_percent_relative"]
    )
    test_ok = (
        comparison["test"]["accuracy"]["meets_15_percent_relative"]
        or comparison["test"]["f1_macro"]["meets_15_percent_relative"]
    )
    comparison["is_smoke_run"] = args.max_per_class is not None
    comparison["valid_for_assignment_claim"] = args.max_per_class is None
    if args.max_per_class is not None:
        comparison["warning"] = (
            "הרצה עם --max-per-class היא בדיקת צינור בלבד. "
            "אסור לדווח את המדדים האלה כיעד ה-15% של המטלה. "
            "להגשה יש להריץ: python -m src.run_pipeline"
        )
        val_ok = False
        test_ok = False
    comparison["target_met_on_validation"] = bool(val_ok)
    comparison["target_met_on_test"] = bool(test_ok)
    if not comparison["valid_for_assignment_claim"]:
        print("[warning] This is a test run. The metrics are not valid for the 15% improvement target.")
    if not (val_ok or test_ok):
        comparison["next_steps_if_target_missed"] = [
            "להאריך את האימון עם יותר epochs ועם patience רחב יותר.",
            "לשחרר עוד בלוקים ב-Fine-Tuning או לעבור ל-EfficientNet-B2.",
            "להוסיף Mixup/CutMix על Train בלבד.",
            "להוסיף WeightedRandomSampler בנוסף למשקלות ב-Loss.",
            "לבדוק ידנית את outputs/reports/flagged_mislabeled.csv.",
        ]

    (REPORTS_DIR / "improvement_vs_baseline.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "experiment_settings.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "backbone": BACKBONE_NAME,
                "epochs_requested": args.epochs,
                "experiments": EXPERIMENTS,
                "ablation_split": args.eval_split_for_ablation,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n=== summary of improvement vs Baseline (Validation) ===")
    print(json.dumps(comparison["validation"], ensure_ascii=False, indent=2))
    print("=== Test for the protocol set in advance ===")
    print(json.dumps(comparison["test"], ensure_ascii=False, indent=2))
    print(f"Reports saved in: {REPORTS_DIR}")


def run(args: argparse.Namespace) -> int:
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw, cleaned, mapping = run_analysis()
    if args.skip_train:
        print("Skipping training due to --skip-train. The reports and graphs at the CSV level are saved in outputs/")
        return 0
    run_training(args, raw, cleaned, mapping)
    return 0


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
