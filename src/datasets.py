"""
יצירת Dataset ו-DataLoader.
Augmentation מוחל אך ורק על Train. Val/Test מקבלים עיבוד דטרמיניסטי זהה.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import BATCH_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS, SEED
from src.data_loading import detect_image_root, resolve_image_file


def build_transforms(augment: bool) -> transforms.Compose:
    """
    Train: Flip, Rotation קל, ColorJitter, RandomResizedCrop — להקטנת Overfitting.
    Val/Test: Resize + CenterCrop + Normalize בלבד, בלי אקראיות.
    """
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class InteriorDataset(Dataset):
    """כל דוגמה היא תמונת חדר עם תווית משולבת (סוג חדר + סגנון)."""

    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_root = detect_image_root()

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, row) -> Path:
        if "resolved_path" in row and isinstance(row["resolved_path"], str) and row["resolved_path"]:
            path = Path(row["resolved_path"])
            if path.exists():
                return path
        path = resolve_image_file(row["image_path_norm"], row["filename"], self.image_root)
        if path is None:
            raise FileNotFoundError(
                f"התמונה לא נמצאה: {row['image_path_norm']}\n"
                "הניחו את קבצי ה-jpg תחת data/raw/<room_type>/<style>/"
            )
        return path

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self._resolve(row)
        with Image.open(path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        label = int(row["encoded_label"])
        return tensor, label


def make_loader(frame: pd.DataFrame, augment: bool, shuffle: bool) -> DataLoader:
    dataset = InteriorDataset(frame, build_transforms(augment=augment))
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def count_available_images(frame: pd.DataFrame) -> int:
    """כמה שורות באמת מצביעות על קובץ קיים — משמש לבדיקה לפני אימון."""
    root = detect_image_root()
    if root is None:
        return 0
    n = 0
    for _, row in frame.iterrows():
        resolved = row.get("resolved_path")
        if isinstance(resolved, str) and resolved and Path(resolved).exists():
            n += 1
            continue
        if resolve_image_file(row["image_path_norm"], row["filename"], root) is not None:
            n += 1
    return n
