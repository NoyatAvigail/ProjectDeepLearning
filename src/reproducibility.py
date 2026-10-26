"""זריעת אקראיות ובדיקת חומרה — הבסיס לניסוי שניתן לשחזר."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from src.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """קובע seed לכל מקורות האקראיות שבהם נשתמש באימון."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> Any:
    """
    מחזיר CUDA אם זמין, אחרת CPU.
    לא מפעיל אימון על GPU שלא קיים — זה מונע קריסות שקטות.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            print(f"[Hardware] GPU available: {name}")
            return device
        print("[Hardware] No GPU found. The training will run on CPU and will be slower.")
        return torch.device("cpu")
    except ImportError as exc:
        raise ImportError(
            "PyTorch אינו מותקן. הריצו: python -m pip install -r requirements.txt"
        ) from exc
