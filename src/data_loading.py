"""
טעינת קבצי הנתונים האמיתיים.
המודול הזה לא ממציא עמודות: הוא קורא את ה-CSV, בודק שדות חובה, ומנרמל נתיבים.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    IMAGE_ROOT_CANDIDATES,
    METADATA_CSV,
    REPORTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)

REQUIRED_SPLIT_COLUMNS = ("image_path", "room_type", "style", "encoded_label")
REQUIRED_META_COLUMNS = ("image_path", "room_type", "style")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"קובץ חובה חסר: {path}\n"
            "ודאו שתיקיית Dataset/Pinterest Interior Design Images and Metadata "
            "מכילה את train_data.csv, val_data.csv, test_data.csv ו-metadata.csv."
        )


def normalize_image_path(raw_path: str) -> str:
    """מאחד לוכסנים של Windows/Unix ומסיר קידומת ./ כדי שאפשר להשוות נתיבים."""
    text = str(raw_path).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def class_name_from_row(room_type: str, style: str) -> str:
    return f"{room_type}__{style}"


def build_label_mapping(frames: list[pd.DataFrame]) -> dict[int, str]:
    """בונה מיפוי encoded_label -> שם מחלקה מתוך הנתונים בפועל, לא מתוך הנחה."""
    combined = pd.concat(frames, ignore_index=True)
    mapping: dict[int, str] = {}
    collisions: list[str] = []
    for _, row in combined.iterrows():
        code = int(row["encoded_label"])
        name = class_name_from_row(row["room_type"], row["style"])
        if code in mapping and mapping[code] != name:
            collisions.append(f"{code}: {mapping[code]} מול {name}")
        mapping[code] = name
    if collisions:
        raise ValueError("נמצאה סתירה במיפוי התוויות: " + "; ".join(collisions))
    return dict(sorted(mapping.items()))


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], csv_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"ב-{csv_name} חסרות עמודות חיוניות: {missing}. "
            f"העמודות שנמצאו: {list(df.columns)}. "
            "אין להמשיך באימון לפני הבהרת מבנה הנתונים."
        )


def load_raw_splits() -> dict[str, pd.DataFrame]:
    """טוען Train/Val/Test/Metadata ובודק עמודות חובה."""
    _require_file(TRAIN_CSV)
    _require_file(VAL_CSV)
    _require_file(TEST_CSV)
    _require_file(METADATA_CSV)

    splits: dict[str, pd.DataFrame] = {}
    for split_name, path, required in [
        ("train", TRAIN_CSV, REQUIRED_SPLIT_COLUMNS),
        ("val", VAL_CSV, REQUIRED_SPLIT_COLUMNS),
        ("test", TEST_CSV, REQUIRED_SPLIT_COLUMNS),
        ("metadata", METADATA_CSV, REQUIRED_META_COLUMNS),
    ]:
        df = pd.read_csv(path)
        _validate_columns(df, required, path.name)
        df = df.copy()
        df["split"] = split_name
        df["image_path_raw"] = df["image_path"]
        df["image_path_norm"] = df["image_path"].map(normalize_image_path)
        df["filename"] = df["image_path_norm"].map(lambda p: Path(p).name)
        df["room_type"] = df["room_type"].astype(str).str.strip()
        df["style"] = df["style"].astype(str).str.strip()
        df["class_name"] = [
            class_name_from_row(r, s) for r, s in zip(df["room_type"], df["style"])
        ]
        if "encoded_label" in df.columns:
            df["encoded_label"] = df["encoded_label"].astype(int)
        splits[split_name] = df

    mapping = build_label_mapping([splits["train"], splits["val"], splits["test"]])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[טעינה] train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}, metadata={len(splits['metadata'])}")
    print(f"[טעינה] מספר מחלקות מתוך הנתונים: {len(mapping)}")
    return splits


_IMAGE_INDEX: dict[str, Path] | None = None
_IMAGE_INDEX_ROOT: Path | None = None


def _has_jpeg(folder: Path) -> bool:
    if not folder.exists():
        return False
    return next(folder.rglob("*.jpg"), None) is not None or next(folder.rglob("*.jpeg"), None) is not None


def detect_image_root(sample_relative_path: str | None = None) -> Path | None:
    """
    מאתר את תיקיית התמונות לפי מועמדים שנבדקו בדיסק.
    בפועל התמונות יושבות תחת תיקיית ה-CSV במבנה <room>/<style>/<file>.
    """
    _ = sample_relative_path
    for candidate in IMAGE_ROOT_CANDIDATES:
        if _has_jpeg(candidate):
            return candidate
    return None


def build_image_index(image_root: Path) -> dict[str, Path]:
    """אינדקס חד-פעמי: שם קובץ -> נתיב מלא. מונע rglob לכל שורה."""
    global _IMAGE_INDEX, _IMAGE_INDEX_ROOT
    if _IMAGE_INDEX is not None and _IMAGE_INDEX_ROOT == image_root:
        return _IMAGE_INDEX
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.is_file():
            index.setdefault(path.name, path)
    _IMAGE_INDEX = index
    _IMAGE_INDEX_ROOT = image_root
    print(f"[טעינה] נבנה אינדקס תמונות: {len(index)} קבצים תחת {image_root}")
    return index


def resolve_image_file(image_path_norm: str, filename: str, image_root: Path | None) -> Path | None:
    """ממפה שורת CSV לקובץ תמונה ממשי. מחזיר None אם הקובץ לא נמצא."""
    if image_root is None:
        return None

    rel = image_path_norm
    if rel.startswith("../"):
        rel = rel[3:]

    parts = Path(rel).parts
    if len(parts) >= 3:
        structured = image_root / parts[-3] / parts[-2] / parts[-1]
        if structured.exists():
            return structured

    if (image_root / rel).exists():
        return image_root / rel
    direct = image_root / filename
    if direct.exists():
        return direct
    return build_image_index(image_root).get(filename)
