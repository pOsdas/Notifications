import torch
import os
from typing import Tuple
from functools import lru_cache
from django.conf import settings

from app.api.v1.features.phone_cnn_v2 import predict_prob


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

MODEL = None
ARTIFACT_PREFIX = settings.CHAR_CNN_MODEL_PREFIX
DEFAULT_THRESHOLD = 0.5


@lru_cache(maxsize=4096)
def _cached_score(text: str) -> float:
    """Внутренний кеш, вычисляет score через predict_prob(MODEL, text)"""
    if MODEL is None:
        raise RuntimeError("CharCNN модель не загружена")
    score = predict_prob(MODEL, text)
    return float(score)


def phone_score(text: str, use_cache: bool = True) -> float:
    """
    Возвращает вероятность [0.0, 1.0], что text — телефон.
    use_cache=True использует LRU-кеш
    """
    if use_cache:
        return _cached_score(text)
    else:
        if MODEL is None:
            raise RuntimeError("CharCNN модель не загружена")
        return float(predict_prob(MODEL, text))


def is_phone_number(text: str, threshold: float = DEFAULT_THRESHOLD, use_cache: bool = True) -> Tuple[bool, float]:
    """
    Возвращает (is_phone: bool, score: float).
    is_phone = score >= threshold.
    """
    score = phone_score(text, use_cache=use_cache)
    return (score >= threshold, score)


def is_phone_bool(text: str, threshold: float = DEFAULT_THRESHOLD, use_cache: bool = True) -> bool:
    return is_phone_number(text, threshold=threshold, use_cache=use_cache)[0]

