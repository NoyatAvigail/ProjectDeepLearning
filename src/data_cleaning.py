"""
ניקוי נתונים גולמיים (Dirty Data) עם תיעוד מלא.

עקרונות:
1. לא משנים תווית אוטומטית בלי ראיה חזקה.
2. Test לא משמש לבחירת כללי ניקוי סטטיסטיים (אין ספי pixel שנלמדים מ-Train ומוחלים על Test).
3. כפילות בין פיצולים מטופלת בהסרת הדוגמה מ-Train בלבד, כדי לא לדלוף מידע אל Test.
4. חשד ל-Mislabeled בלי אימות ויזואלי → סימון / החרגה, לא "תיקון".
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import (
    AHASH_SIZE,
    CLEANED_DIR,
    NEAR_DUP_HAMMING,
    REPORTS_DIR,
)
from src.data_loading import build_image_index, detect_image_root, resolve_image_file

MIN_IMAGE_SIDE = 32
MAX_ASPECT_RATIO = 6.0


@dataclass
class CleaningReport:
    """דוח מצטבר של כל פעולות הניקוי, כולל ספירות לפני/אחרי."""

    steps: list[dict] = field(default_factory=list)
    flagged_mislabeled: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    image_root: str | None = None
    images_available: bool = False

    def add_step(self, name: str, before: dict[str, int], after: dict[str, int], removed: int, detail: str) -> None:
        self.steps.append(
            {
                "step": name,
                "before": before,
                "after": after,
                "removed": removed,
                "detail": detail,
            }
        )

    def to_dict(self) -> dict:
        return {
            "image_root": self.image_root,
            "images_available": self.images_available,
            "steps": self.steps,
            "flagged_mislabeled": self.flagged_mislabeled,
            "notes": self.notes,
        }


def _counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {k: int(len(v)) for k, v in splits.items() if k != "metadata"}


def _drop_by_index(df: pd.DataFrame, drop_idx: pd.Index, reason_col: str, reason: str) -> pd.DataFrame:
    out = df.copy()
    if reason_col not in out.columns:
        out[reason_col] = ""
    out.loc[drop_idx, reason_col] = reason
    return out.drop(index=drop_idx).reset_index(drop=True)


def average_hash(path: Path, hash_size: int = AHASH_SIZE) -> str | None:
    """Average Hash פשוט לזיהוי כפילויות קרובות בלי תלות בספריות חיצוניות."""
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = list(gray.getdata())
    except (OSError, UnidentifiedImageError):
        return None
    avg = sum(pixels) / max(len(pixels), 1)
    return "".join("1" if px > avg else "0" for px in pixels)


def hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 64), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    """בדיקות אובייקטיביות לקובץ בודד: קריאות, גודל, יחס ממדים."""
    info = {"ok": False, "reason": "", "width": None, "height": None, "mode": None}
    if not path.exists() or path.stat().st_size == 0:
        info["reason"] = "missing_or_empty"
        return info
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
    except (OSError, UnidentifiedImageError, ValueError):
        info["reason"] = "unreadable"
        return info
    if min(width, height) < MIN_IMAGE_SIDE:
        info["reason"] = "too_small"
        return info
    aspect = max(width, height) / max(min(width, height), 1)
    if aspect > MAX_ASPECT_RATIO:
        info["reason"] = "extreme_aspect_ratio"
        return info
    info.update(ok=True, width=width, height=height, mode=mode)
    return info


def _path_folder_labels(image_path_norm: str) -> tuple[str | None, str | None]:
    parts = Path(image_path_norm).parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return None, None


def clean_splits(raw_splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], CleaningReport]:
    """
    מריץ את כל שלבי הניקוי על עותק של הנתונים.
    Train/Val/Test מטופלים בנפרד. Metadata משמש לאימות תוויות בלבד.
    """
    report = CleaningReport()
    splits = {k: v.copy() for k, v in raw_splits.items()}
    for key in ("train", "val", "test"):
        splits[key]["exclude_reason"] = ""

    image_root = detect_image_root()
    report.image_root = str(image_root) if image_root else None
    report.images_available = image_root is not None
    if image_root is None:
        report.notes.append(
            "לא נמצאו קבצי תמונה בפרויקט. ניקוי ברמת CSV יתבצע, "
            "אך לא ניתן לאמת קבצים פגומים או כפילויות ויזואליות. "
            "יש להניח את התמונות תחת data/raw/<room_type>/<style>/<filename>.jpg"
        )
        print("[ניקוי] אזהרה: תמונות לא נמצאו. ראו את הדוח.")
    else:
        print(f"[ניקוי] שורש תמונות: {image_root}")

    # --- 1. ערכים חסרים / ריקים ---
    before = _counts(splits)
    removed_total = 0
    for key in ("train", "val", "test"):
        df = splits[key]
        bad = df[
            df["image_path_norm"].isna()
            | (df["image_path_norm"].astype(str).str.len() == 0)
            | df["room_type"].isna()
            | df["style"].isna()
            | (df["room_type"] == "")
            | (df["style"] == "")
        ].index
        removed_total += len(bad)
        splits[key] = _drop_by_index(df, bad, "exclude_reason", "missing_fields")
    report.add_step("missing_fields", before, _counts(splits), removed_total, "הסרת שורות עם נתיב או תווית ריקים")

    # --- 2. כפילויות מדויקות בתוך כל פיצול ---
    before = _counts(splits)
    removed_total = 0
    for key in ("train", "val", "test"):
        df = splits[key]
        dup_path = df.duplicated(subset=["image_path_norm"], keep="first")
        dup_name = df.duplicated(subset=["filename"], keep="first")
        bad = df[dup_path | dup_name].index
        removed_total += len(bad)
        splits[key] = _drop_by_index(df, bad, "exclude_reason", "exact_duplicate_in_split")
    report.add_step("duplicates_in_split", before, _counts(splits), removed_total, "הסרת כפילות נתיב/שם קובץ בתוך אותו פיצול")

    # --- 3. Data Leakage בין פיצולים לפי נתיב ושם קובץ ---
    before = _counts(splits)
    val_names = set(splits["val"]["filename"]) | set(splits["val"]["image_path_norm"])
    test_names = set(splits["test"]["filename"]) | set(splits["test"]["image_path_norm"])
    train_leak = splits["train"][
        splits["train"]["filename"].isin(val_names | test_names)
        | splits["train"]["image_path_norm"].isin(val_names | test_names)
    ].index
    # גם חפיפה val-test: מסירים מ-val כדי לשמור את test קבוע להערכה סופית
    val_leak = splits["val"][
        splits["val"]["filename"].isin(set(splits["test"]["filename"]))
        | splits["val"]["image_path_norm"].isin(set(splits["test"]["image_path_norm"]))
    ].index
    removed = len(train_leak) + len(val_leak)
    splits["train"] = _drop_by_index(splits["train"], train_leak, "exclude_reason", "leakage_to_val_or_test")
    splits["val"] = _drop_by_index(splits["val"], val_leak, "exclude_reason", "leakage_to_test")
    report.add_step(
        "cross_split_path_leakage",
        before,
        _counts(splits),
        removed,
        "הסרת דליפה לפי נתיב/שם קובץ. Test לא נמחק כדי לשמור הערכה יציבה.",
    )

    # --- 4. חשד ל-Mislabeled: תיקיית הנתיב מול עמודות התווית ---
    before = _counts(splits)
    removed_total = 0
    for key in ("train", "val", "test"):
        df = splits[key]
        drop_idx = []
        for idx, row in df.iterrows():
            folder_room, folder_style = _path_folder_labels(row["image_path_norm"])
            if folder_room and folder_style:
                if folder_room != row["room_type"] or folder_style != row["style"]:
                    item = {
                        "split": key,
                        "filename": row["filename"],
                        "csv_label": row["class_name"],
                        "path_label": f"{folder_room}__{folder_style}",
                        "action": "excluded",
                        "rule": "folder_vs_csv_mismatch",
                    }
                    report.flagged_mislabeled.append(item)
                    drop_idx.append(idx)
        removed_total += len(drop_idx)
        splits[key] = _drop_by_index(df, pd.Index(drop_idx), "exclude_reason", "path_label_mismatch")
    report.add_step(
        "mislabeled_path_vs_columns",
        before,
        _counts(splits),
        removed_total,
        "סתירה בין תיקיית הנתיב לבין room_type/style. הדוגמה הוחרגה ולא תוקנה אוטומטית.",
    )

    # --- 5. אימות מול metadata לפי שם קובץ ---
    before = _counts(splits)
    meta = splits["metadata"].copy()
    meta_map = {
        fn: (room, style)
        for fn, room, style in zip(meta["filename"], meta["room_type"], meta["style"])
    }
    removed_total = 0
    for key in ("train", "val", "test"):
        df = splits[key]
        drop_idx = []
        for idx, row in df.iterrows():
            if row["filename"] not in meta_map:
                continue
            m_room, m_style = meta_map[row["filename"]]
            if m_room != row["room_type"] or m_style != row["style"]:
                report.flagged_mislabeled.append(
                    {
                        "split": key,
                        "filename": row["filename"],
                        "csv_label": row["class_name"],
                        "metadata_label": f"{m_room}__{m_style}",
                        "action": "excluded",
                        "rule": "metadata_vs_split_mismatch",
                    }
                )
                drop_idx.append(idx)
        removed_total += len(drop_idx)
        splits[key] = _drop_by_index(df, pd.Index(drop_idx), "exclude_reason", "metadata_label_mismatch")
    report.add_step(
        "metadata_consistency",
        before,
        _counts(splits),
        removed_total,
        "סתירה מול metadata.csv לפי שם קובץ. אין תיקון אוטומטי של התווית.",
    )

    # --- 6. עקביות encoded_label מול (room_type, style) ---
    before = _counts(splits)
    combined = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    combo_to_codes: dict[str, set[int]] = defaultdict(set)
    for _, row in combined.iterrows():
        combo_to_codes[row["class_name"]].add(int(row["encoded_label"]))
    ambiguous = {k: sorted(v) for k, v in combo_to_codes.items() if len(v) > 1}
    removed_total = 0
    if ambiguous:
        report.notes.append(f"זיהוי encoded_label לא חד-ערכי: {ambiguous}. השורות האלה יוחרגו.")
        for key in ("train", "val", "test"):
            df = splits[key]
            bad = df[df["class_name"].isin(ambiguous.keys())].index
            removed_total += len(bad)
            splits[key] = _drop_by_index(df, bad, "exclude_reason", "ambiguous_encoded_label")
    report.add_step(
        "encoded_label_consistency",
        before,
        _counts(splits),
        removed_total,
        "encoded_label חייב להיות חד-ערכי לכל צירוף room_type+style.",
    )

    # --- 7. קבצים פגומים / חריגים — רק אם התמונות קיימות ---
    before = _counts(splits)
    removed_total = 0
    hash_records: list[tuple[str, str, str, Path]] = []  # split, class, hash, path
    if image_root is not None:
        build_image_index(image_root)
        for key in ("train", "val", "test"):
            df = splits[key]
            drop_idx = []
            resolved_paths = []
            for idx, row in df.iterrows():
                path = resolve_image_file(row["image_path_norm"], row["filename"], image_root)
                if path is None:
                    drop_idx.append(idx)
                    continue
                info = inspect_image(path)
                if not info["ok"]:
                    drop_idx.append(idx)
                    continue
                resolved_paths.append((idx, path))
                digest = average_hash(path)
                checksum = file_md5(path)
                if digest:
                    hash_records.append((key, row["class_name"], digest, checksum, path))
            removed_total += len(drop_idx)
            splits[key] = _drop_by_index(df, pd.Index(drop_idx), "exclude_reason", "missing_corrupt_or_outlier")
            fname_to_path = {Path(p).name: str(p) for _, p in resolved_paths}
            splits[key]["resolved_path"] = splits[key]["filename"].map(fname_to_path)
        report.add_step(
            "corrupt_missing_outliers",
            before,
            _counts(splits),
            removed_total,
            "הסרת קבצים חסרים, לא קריאים, קטנים מדי או עם יחס ממדים קיצוני.",
        )

        # --- 8. כפילות תוכן מדויקת + כפילות קרובה בין פיצולים ---
        before = _counts(splits)
        holdout_md5 = {md5 for split, _, _, md5, _ in hash_records if split in {"val", "test"}}
        holdout_ahash = {ah for split, _, ah, _, _ in hash_records if split in {"val", "test"}}
        train_df = splits["train"]
        drop_idx = []
        train_meta = {
            Path(p).name: (ah, md5) for split, _, ah, md5, p in hash_records if split == "train"
        }
        for idx, row in train_df.iterrows():
            meta = train_meta.get(row["filename"])
            if not meta:
                continue
            ah, md5 = meta
            if md5 in holdout_md5:
                drop_idx.append(idx)
                continue
            if any(hamming(ah, other) <= NEAR_DUP_HAMMING for other in holdout_ahash):
                drop_idx.append(idx)
        splits["train"] = _drop_by_index(train_df, pd.Index(drop_idx), "exclude_reason", "near_duplicate_leakage")
        report.add_step(
            "near_duplicate_leakage",
            before,
            _counts(splits),
            len(drop_idx),
            f"הסרת Train עם MD5 זהה ל-Val/Test או Average Hash Hamming<={NEAR_DUP_HAMMING} (גודל {AHASH_SIZE}x{AHASH_SIZE}).",
        )

        # כפילות זהה עם תווית שונה ב-Train — חשד ל-Mislabeled, החרגה בלי תיקון
        before = _counts(splits)
        by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for split, cls, ah, md5, p in hash_records:
            if split != "train":
                continue
            by_hash[ah].append((cls, Path(p).name))
            by_hash[f"md5:{md5}"].append((cls, Path(p).name))
        conflict_names = set()
        for key, items in by_hash.items():
            classes = {cls for cls, _ in items}
            if len(classes) <= 1:
                continue
            names = [name for _, name in items]
            conflict_names.update(names)
            report.flagged_mislabeled.append(
                {
                    "split": "train",
                    "hash_key": key[:24],
                    "filenames": names[:8],
                    "labels": sorted(classes),
                    "action": "excluded_all_in_group",
                    "rule": "identical_hash_conflicting_labels",
                }
            )
            bad = splits["train"][splits["train"]["filename"].isin(conflict_names)].index
        splits["train"] = _drop_by_index(splits["train"], bad, "exclude_reason", "conflicting_near_duplicate")
        report.add_step(
            "conflicting_near_duplicates",
            before,
            _counts(splits),
            len(bad),
            "תמונות עם אותו MD5 או אותו Average Hash ועם תוויות שונות הוחרגו. אין תיקון תווית אוטומטי.",
        )
    else:
        for key in ("train", "val", "test"):
            splits[key]["resolved_path"] = ""
        report.add_step(
            "corrupt_missing_outliers",
            before,
            _counts(splits),
            0,
            "דולג: אין תמונות לבדיקת קבצים פגומים.",
        )

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("train", "val", "test"):
        out_path = CLEANED_DIR / f"{key}_cleaned.csv"
        splits[key].to_csv(out_path, index=False, encoding="utf-8")
    (REPORTS_DIR / "cleaning_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if report.flagged_mislabeled:
        pd.DataFrame(report.flagged_mislabeled).to_csv(
            REPORTS_DIR / "flagged_mislabeled.csv",
            index=False,
            encoding="utf-8",
        )
    print("[ניקוי] סיכום ספירות אחרי ניקוי:", _counts(splits))
    return splits, report
