from __future__ import annotations

import math
from typing import List

import numpy as np


class PageHinkley:
    """Project implementation reproduced from the notebook logic."""

    def __init__(self, delta: float = 0.01, threshold: float = 5.0, alpha: float = 0.999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = 0.0
        self.count = 0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        self.sum += value - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        return (self.sum - self.min_sum) > self.threshold


def embed_drift_score(sequence: np.ndarray) -> float:
    if sequence.size == 0:
        return 0.0
    early_window = sequence[: max(1, len(sequence) // 3)]
    recent_window = sequence[-max(1, len(sequence) // 3):]
    early_mean = np.mean(early_window, axis=0) if early_window.ndim > 1 else np.asarray([float(np.mean(early_window))])
    recent_mean = np.mean(recent_window, axis=0) if recent_window.ndim > 1 else np.asarray([float(np.mean(recent_window))])
    score = float(np.linalg.norm(early_mean - recent_mean))
    return max(0.0, float(score))


def classify_drift_score(score: float) -> str:
    if score < 1.0:
        return "Low Drift"
    if score < 3.0:
        return "Moderate Drift"
    return "High Drift"


def page_hinkley_flag(sequence_scores: List[float]) -> bool:
    ph = PageHinkley(delta=0.01, threshold=5.0, alpha=0.999)
    for value in sequence_scores:
        if ph.update(float(value)):
            return True
    return False


def normalize_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default
