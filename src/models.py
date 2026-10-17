"""
שלב א' — בחירת ארכיטקטורה Pre-trained וטעינת משקולות ImageNet.
EfficientNet-B0 נבחר כאיזון טוב בין דיוק לעלות חישוב, מתאים ל-20 מחלקות תמונה.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import NUM_CLASSES


def build_model(unfreeze_backbone: bool) -> nn.Module:
    """
    טוען EfficientNet-B0 עם משקולות ImageNet.
    במצב Baseline הרשת הקפיאה משמשת כחילוץ מאפיינים, ומאומן רק ראש הסיווג.
    ב-Fine-Tuning משחררים גם את הבלוקים האחרונים של הרשת.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    for param in model.features.parameters():
        param.requires_grad = False

    if unfreeze_backbone:
        # Fine-Tuning מבוקר: משחררים את 3 הבלוקים האחרונים ולא את כל הרשת בבת אחת
        for block in model.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
