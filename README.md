# סיווג סוג חדר וסגנון עיצוב — Deep Learning

פרויקט מעשי לפי מטלת Pre-trained Architectures: ניקוי Dirty Data, Transfer Learning, Fine-Tuning, Data Augmentation, ויזואליזציות, ומחקר Ablation בארבעה שלבים.

האג'נט העתידי (הצעת אקססוריז) **אינו חלק משלב זה**. כאן מאומן רק מודל הראייה הממוחשבת.

## מה מצאתי בנתונים בפועל

לא הנחתי מבנה מראש. אחרי קריאת הקבצים:

| קובץ | שורות | עמודות |
|---|---|---|
| `train_data.csv` | 2648 | `image_path, room_type, style, encoded_label` |
| `val_data.csv` | 663 | אותן עמודות |
| `test_data.csv` | 828 | אותן עמודות |
| `metadata.csv` | 4139 | `image_path, room_type, style` |

- 20 מחלקות: 4 סוגי חדר (`bathroom`, `bedroom`, `kitchen`, `living_room`) × 5 סגנונות (`boho`, `industrial`, `minimalist`, `modern`, `scandinavian`).
- `encoded_label` עקבי עם הצירוף (למשל 13 = `kitchen__modern`).
- אין כפילות נתיבים בתוך פיצול ואין חפיפת שמות קובץ בין Train/Val/Test.
- `metadata.csv` מתאר את אותן תמונות, עם לוכסנים שונים (`\` מול `/`). אחרי נרמול, כל שורות הפיצולים מופיעות ב-metadata.
- ב-CSV רשום `../data/raw/<room>/<style>/<file>.jpg`, אבל **בדיסק** התמונות נמצאות ליד ה-CSV:

```text
Dataset/Pinterest Interior Design Images and Metadata/<room_type>/<style>/<filename>.jpg
Dataset/Pinterest Interior Design Images and Metadata/kitchen/modern/kitchen_modern_224.jpg
```

הקוד פותר את הפער הזה אוטומטית (תיקיית החדר + שם הקובץ), בלי לשנות את קבצי המקור.

## התקנה

מומלץ סביבה וירטואלית. נבדק מול Python 3.11–3.12; ב-Python 3.14 ייתכן שגלגל PyTorch עדיין לא זמין.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

אם אין GPU, האימון ירוץ על CPU (איטי יותר). הקוד בודק זאת אוטומטית.

## הרצה

מתוך שורש הפרויקט:

```bash
# ניתוח + ניקוי + גרפי התפלגות בלבד (לא דורש תמונות ולא דורש GPU)
python -m src.run_pipeline --skip-train

# מחזור מלא: ארבעה ניסויים, גרפים, שמירת מודל ותוצאות
python -m src.run_pipeline

# הרצה מלאה (מומלץ GPU; על CPU זה ארוך)
python -m src.run_pipeline

# בדיקת צינור קצרה בלבד — לא לתוצאות הגשה
python -m src.run_pipeline --epochs 1 --max-per-class 8
```

## ארבעת הניסויים (Ablation)

כולם על אותה משימה (20 מחלקות) ואותו מדד (`f1_macro` וגם Accuracy):

1. **Baseline** — EfficientNet-B0 עם משקולות ImageNet, backbone קפוא, נתונים גולמיים, בלי Augmentation.
2. **ניקוי** — אותה רשת קפואה אחרי ניקוי.
3. **ניקוי + Augmentation** — Flip/Rotation/ColorJitter/RandomResizedCrop על **Train בלבד**, עם class weights.
4. **Fine-Tuning** — שחרור 3 הבלוקים האחרונים + ניקוי + Augmentation. זה המודל הסופי שהוגדר מראש.

Validation משמש ל-Early Stopping ולבחירת checkpoint. Test לא משמש לכיוון היפר-פרמטרים. גרף ה-Ablation נשמר כברירת מחדל לפי Validation. מספרי Test מדווחים לפרוטוקול שנקבע מראש (exp1 מול exp4), לא כדי לבחור את הניסוי המנצח אחרי מעשה.

בדיקת הצינור על המחשב הזה אומתה (טעינה, ניקוי, ארבעה ניסויים, גרפים). האימון המלא על CPU ארוך; להגשה יש להריץ בלי `--max-per-class`.

## שיפור של 15%

הקוד מחשב בפועל:

- שיפור יחסי: `(final - baseline) / baseline * 100`
- הפרש בנקודות אחוז: `(final - baseline) * 100`

אם היעד לא הושג, התוצאות האמיתיות נשמרות ב-`outputs/reports/improvement_vs_baseline.json` יחד עם הצעות לשיפור. **אין המצאת מדדים.**

## ארטיפקטים

אחרי הרצה מלאה:

- `outputs/figures/` — countplot, learning curves, heatmap, barplot
- `outputs/models/` — משקולות והיסטוריית אימון
- `outputs/reports/` — דוח ניקוי, Ablation CSV, מטריקות, מיפוי תוויות
- `outputs/cleaned_csv/` — הפיצולים אחרי ניקוי

## כללי ניקוי (מתועדים)

- הסרת שורות ריקות וכפילויות בתוך פיצול.
- דליפה בין פיצולים: מסירים מ-Train (ואם צריך מ-Val), לא מוחקים את Test.
- סתירת תיקייה מול תווית, או סתירה מול metadata: **החרגה**, לא תיקון אוטומטי.
- קובץ פגום / קטן מדי / יחס ממדים קיצוני: הסרה.
- כפילות קרובה (Average Hash) מול Val/Test: הסרה מ-Train.
- כפילות קרובה עם תוויות שונות ב-Train: החרגת שתי הדוגמאות וסימון ב-`flagged_mislabeled.csv`.
