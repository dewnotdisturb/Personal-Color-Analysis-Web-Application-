"""Load the trained KNN classifier and predict a season from features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import ColorFeatures
from .train import MODEL_PATH, SCALER_PATH, train_and_save


@dataclass(frozen=True)
class Prediction:
    season: str
    confidence: float
    probabilities: dict[str, float]


def _ensure_trained() -> None:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        train_and_save()


class SeasonClassifier:
    def __init__(self) -> None:
        _ensure_trained()
        self._knn = joblib.load(MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)

    def predict(self, color_features: ColorFeatures) -> Prediction:
        vector = color_features.to_vector().reshape(1, -1)
        scaled = self._scaler.transform(vector)
        proba = self._knn.predict_proba(scaled)[0]
        classes = self._knn.classes_
        probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        return Prediction(
            season=classes[best_idx],
            confidence=float(proba[best_idx]),
            probabilities=probabilities,
        )


_classifier: SeasonClassifier | None = None


def get_classifier() -> SeasonClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SeasonClassifier()
    return _classifier
