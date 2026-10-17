"""
הגדרות מרכזיות לפרויקט.
כל הנתיבים והיפר-פרמטרים מרוכזים כאן כדי שהניסויים יהיו ניתנים לשחזור.
"""

from __future__ import annotations

from pathlib import Path

# שורש הפרויקט: תיקייה אחת מעל src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# מיקום קבצי ה-CSV שסופקו במטלה (נבדק בפועל בפרויקט)
DATASET_DIR = PROJECT_ROOT / "Dataset" / "Pinterest Interior Design Images and Metadata"
TRAIN_CSV = DATASET_DIR / "train_data.csv"
VAL_CSV = DATASET_DIR / "val_data.csv"
TEST_CSV = DATASET_DIR / "test_data.csv"
METADATA_CSV = DATASET_DIR / "metadata.csv"

# בפועל התמונות נמצאות ליד ה-CSV: <DATASET_DIR>/<room_type>/<style>/<file>.jpg
# ב-CSV רשום ../data/raw/... ולכן יש גם מועמדים תואמים לתיעוד המקורי.
IMAGE_ROOT_CANDIDATES = [
    DATASET_DIR,
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "Dataset" / "data" / "raw",
    DATASET_DIR / "data" / "raw",
    DATASET_DIR.parent / "data" / "raw",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
CLEANED_DIR = OUTPUT_DIR / "cleaned_csv"

# זריעת אקראיות — חובה לשחזור תוצאות
SEED = 42

# 20 מחלקות: 4 סוגי חדר × 5 סגנונות. המיפוי אומת מתוך ה-CSV.
NUM_CLASSES = 20
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # ב-Windows ערך 0 יציב יותר מתהליכי משנה

# אימון
EPOCHS = 12
PATIENCE = 4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# ImageNet — כי EfficientNet/ResNet אומנו כך במקור
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# כפילות מדויקת: MD5 של הקובץ.
# כפילות קרובה: Average Hash 16x16. סף Hamming נמוך כדי לא למחוק חדרי פנים שונים שנראים דומה.
AHASH_SIZE = 16
NEAR_DUP_HAMMING = 4
CONFLICT_HAMMING = 0

# מדד ראשי להשוואת ניסויים (נשמר גם Accuracy)
PRIMARY_METRIC = "f1_macro"

# ארבעת שלבי ה-Ablation Study
EXPERIMENTS = {
    "exp1_baseline_frozen_raw": {
        "title": "Baseline: EfficientNet-B0 קפוא, נתונים גולמיים, ללא Augmentation",
        "use_cleaned": False,
        "use_augmentation": False,
        "unfreeze_backbone": False,
        "use_class_weights": False,
        "lr": 1e-3,
    },
    "exp2_cleaned_frozen": {
        "title": "ניקוי נתונים + רשת קפואה, ללא Augmentation",
        "use_cleaned": True,
        "use_augmentation": False,
        "unfreeze_backbone": False,
        "use_class_weights": False,
        "lr": 1e-3,
    },
    "exp3_cleaned_augmented_frozen": {
        "title": "ניקוי + Data Augmentation (Train בלבד) + רשת קפואה",
        "use_cleaned": True,
        "use_augmentation": True,
        "unfreeze_backbone": False,
        "use_class_weights": True,
        "lr": 1e-3,
    },
    "exp4_finetune_augmented": {
        "title": "Fine-Tuning מלא + ניקוי + Augmentation",
        "use_cleaned": True,
        "use_augmentation": True,
        "unfreeze_backbone": True,
        "use_class_weights": True,
        "lr": 3e-4,
    },
}

BACKBONE_NAME = "efficientnet_b0"
