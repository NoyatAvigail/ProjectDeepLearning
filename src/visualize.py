"""
ויזואליזציות שנדרשות במטלה:
1. sns.countplot — התפלגות מחלקות לפני ואחרי ניקוי
2. plt.plot — עקומות Loss ו-Accuracy
3. sns.heatmap — מטריצת בלבול מנורמלת
4. sns.barplot — Ablation Study לאורך 4 ניסויים
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", font="Tahoma")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {path}")
    return path


def plot_class_distribution(frame: pd.DataFrame, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    order = frame["class_name"].value_counts().index
    sns.countplot(data=frame, x="class_name", order=order, ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("מחלקה (סוג חדר + סגנון)")
    ax.set_ylabel("מספר דוגמאות")
    ax.tick_params(axis="x", rotation=75)
    return _save(fig, filename)


def plot_learning_curves(history: dict, experiment_id: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"עקומת Loss — {experiment_id}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_title(f"עקומת Accuracy — {experiment_id}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    return _save(fig, f"learning_curves_{experiment_id}.png")


def plot_confusion_matrix(cm_normalized, class_names: list[str], experiment_id: str, split_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"מטריצת בלבול מנורמלת — {experiment_id} ({split_name})")
    ax.set_xlabel("חיזוי")
    ax.set_ylabel("תווית אמיתית")
    ax.tick_params(axis="x", rotation=75)
    ax.tick_params(axis="y", rotation=0)
    return _save(fig, f"confusion_matrix_{experiment_id}_{split_name}.png")


def plot_ablation(results_df: pd.DataFrame, metric: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=results_df, x="experiment_id", y=metric, ax=ax, color="#72B7B2")
    ax.set_title(f"Ablation Study — השוואת {metric} בארבעה שלבי ניסוי")
    ax.set_xlabel("שלב ניסוי")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    return _save(fig, filename)
