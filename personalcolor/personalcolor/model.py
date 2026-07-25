"""Load the trained KNN classifier and predict a season from features."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np

from .features import ColorFeatures
from .train import MODEL_PATH, SCALER_PATH, train_and_save


@dataclass(frozen=True)
class Prediction:
    season: str
    confidence: float  # winning class's probability
    probabilities: dict[str, float]  # full distribution over all 4 seasons


def _ensure_trained() -> None:
    """Train on first use if no saved model exists yet (e.g. fresh checkout, no manual train.py step)."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        train_and_save()


class SeasonClassifier:
    """Thin wrapper around the persisted scaler + KNN pair, loaded once and reused."""

    def __init__(self) -> None:
        _ensure_trained()
        self._knn = joblib.load(MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)

    def predict(self, color_features: ColorFeatures) -> Prediction:
        # Must scale with the SAME fitted scaler used at training time,
        # or the distances KNN compares against are meaningless.
        vector = color_features.to_vector().reshape(1, -1)  # sklearn expects a 2D (n_samples, n_features) array
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


# Module-level singleton: loading the model from disk on every request
# would be wasteful, so it's loaded once on first use and reused.
_classifier: SeasonClassifier | None = None


def get_classifier() -> SeasonClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SeasonClassifier()
    return _classifier
