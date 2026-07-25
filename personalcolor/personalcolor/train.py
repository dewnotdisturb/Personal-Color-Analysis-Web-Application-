"""Train the seasonal-palette KNN classifier.

IMPORTANT — read before trusting any accuracy number this script prints:
there is no large public dataset of photos labeled with "correct"
seasonal color analysis (it's an inherently subjective, consultant-judged
label). Instead of pretending otherwise, this script generates a
SYNTHETIC dataset from the published color-theory decision rules (see
personalcolor/palette.py docstring): each season occupies a region of
[lightness, undertone-hue, chroma, skin/hair contrast] space, sampled
with Gaussian noise so the classes overlap at the boundaries the way
real people do.

The resulting "accuracy" measures whether KNN can recover a
theory-defined decision boundary from noisy samples — a sanity check on
the feature design, NOT a claim about real-world classification
accuracy. Swapping in genuine expert-labeled photos (same feature
pipeline) is the natural next step and is called out in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "knn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
REPORT_PATH = MODEL_DIR / "eval_report.txt"

# (theta_deg_mean, theta_deg_std) for undertone hue in the Lab a/b plane.
# Noisy enough that warm/cool overlap somewhat at the boundary, like real skin tones do.
_WARM_THETA = (55.0, 18.0)
_COOL_THETA = (-15.0, 18.0)

# (chroma_mean, chroma_std) — radius in the Lab a/b plane.
_CLEAR_CHROMA = (27.0, 7.0)
_SOFT_CHROMA = (16.0, 6.0)

# (lightness_mean, lightness_std) — OpenCV 8-bit Lab L channel.
_LIGHT_L = (170.0, 20.0)
_DEEP_L = (136.0, 20.0)

# (contrast_mean, contrast_std) — |skin_L - hair_L| proxy for clear/soft.
_HIGH_CONTRAST = (40.0, 15.0)
_LOW_CONTRAST = (22.0, 15.0)

_SEASON_PARAMS = {
    # season: (theta, chroma, lightness, contrast)
    "Spring": (_WARM_THETA, _CLEAR_CHROMA, _LIGHT_L, _HIGH_CONTRAST),
    "Summer": (_COOL_THETA, _SOFT_CHROMA, _LIGHT_L, _LOW_CONTRAST),
    "Autumn": (_WARM_THETA, _SOFT_CHROMA, _DEEP_L, _LOW_CONTRAST),
    "Winter": (_COOL_THETA, _CLEAR_CHROMA, _DEEP_L, _HIGH_CONTRAST),
}


def _sample_season(season: str, n: int, rng: np.random.Generator) -> np.ndarray:
    (theta_mu, theta_sd), (chroma_mu, chroma_sd), (l_mu, l_sd), (con_mu, con_sd) = _SEASON_PARAMS[season]

    theta = np.deg2rad(rng.normal(theta_mu, theta_sd, n))
    chroma = np.clip(rng.normal(chroma_mu, chroma_sd, n), 2.0, None)
    lightness = np.clip(rng.normal(l_mu, l_sd, n), 0.0, 255.0)
    contrast = np.clip(rng.normal(con_mu, con_sd, n), 0.0, None)

    a = 128.0 + chroma * np.cos(theta)
    b = 128.0 + chroma * np.sin(theta)

    return np.stack([lightness, a, b, contrast, chroma], axis=1)


def generate_synthetic_dataset(n_per_season: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for season in _SEASON_PARAMS:
        features.append(_sample_season(season, n_per_season, rng))
        labels.extend([season] * n_per_season)
    return np.concatenate(features, axis=0), np.array(labels)


def train_and_save(n_per_season: int = 300, k: int = 9, seed: int = 42) -> str:
    X, y = generate_synthetic_dataset(n_per_season, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train_scaled, y_train)

    report = classification_report(y_test, knn.predict(X_test_scaled))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    REPORT_PATH.write_text(
        "Synthetic held-out set classification report "
        "(sanity check on decision-boundary separability, not real-world accuracy):\n\n"
        + report
    )

    metadata = {
        "n_per_season": n_per_season,
        "k": k,
        "seed": seed,
        "feature_order": ["skin_L", "skin_a", "skin_b", "contrast", "chroma"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return report


if __name__ == "__main__":
    print(train_and_save())
