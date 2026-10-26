"""אימון על Train והערכה ביניים על Validation בלבד. Test לא נכנס לכאן."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import EPOCHS, LABEL_SMOOTHING, MODELS_DIR, PATIENCE, WEIGHT_DECAY
from src.evaluate import metrics_from_predictions


def _class_weights(frame, num_classes: int, device: torch.device) -> torch.Tensor | None:
    counts = frame["encoded_label"].value_counts().to_dict()
    weights = []
    n = len(frame)
    for cls in range(num_classes):
        c = counts.get(cls, 1)
        weights.append(n / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_experiment(
    experiment_id: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_frame,
    device: torch.device,
    lr: float,
    use_class_weights: bool,
    num_classes: int,
    epochs: int = EPOCHS,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    weights = _class_weights(train_frame, num_classes, device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
    }
    best_f1 = -1.0
    best_path = MODELS_DIR / f"{experiment_id}_best.pt"
    patience_left = PATIENCE

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"{experiment_id} epoch {epoch}/{epochs} train"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_stats = evaluate_loader(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_stats["f1_macro"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_stats["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_stats["accuracy"])
        history["val_f1_macro"].append(val_stats["f1_macro"])

        print(
            f"[{experiment_id}] epoch {epoch}: "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.4f} "
            f"f1_macro={val_stats['f1_macro']:.4f}"
        )

        if val_stats["f1_macro"] > best_f1:
            best_f1 = val_stats["f1_macro"]
            patience_left = PATIENCE
            torch.save(
                {
                    "experiment_id": experiment_id,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_stats,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{experiment_id}] Early Stopping at epoch {epoch}")
                break

    history_path = MODELS_DIR / f"{experiment_id}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"history": history, "best_checkpoint": str(best_path), "best_val_f1_macro": best_f1}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    metrics = metrics_from_predictions(np.array(y_true), np.array(y_pred), num_classes)
    metrics["loss"] = running_loss / max(total, 1)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics
