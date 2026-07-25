"""OpenCV-based feature extraction for personal color analysis.

Pipeline: detect a face with a Haar cascade, sample proportional
sub-regions of the face box (forehead/cheeks for skin, a band above the
face for hair) rather than precise landmarks, and reduce each region to
a robust (median) color. Colors are converted to CIELAB (perceptually
uniform, good for "warm vs. cool" and lightness) and HSV (convenient for
chroma/saturation) and combined into a single numeric feature vector fed
to the KNN classifier.

This is an approximation, not landmark-based sampling (e.g. mediapipe
face mesh) — documented as a known limitation, see README.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class NoFaceDetectedError(Exception):
    """Raised when no face can be located in the supplied image."""


@dataclass(frozen=True)
class ColorFeatures:
    """Raw + derived color measurements for one photo.

    skin_lab/hair_lab are OpenCV's 8-bit CIELAB encoding: L in [0,255]
    (scaled from the usual 0-100), a and b in [0,255] with 128 = neutral
    (0). That's why the properties below subtract 128 from a/b before
    doing math with them.
    """

    skin_lab: tuple[float, float, float]
    skin_hsv: tuple[float, float, float]
    hair_lab: tuple[float, float, float]

    @property
    def contrast(self) -> float:
        """Lightness gap between skin and hair; proxy for 'clear vs soft'."""
        return abs(self.skin_lab[0] - self.hair_lab[0])

    @property
    def chroma(self) -> float:
        """Saturation of the skin tone: distance from neutral gray in the Lab a/b plane."""
        _, a, b = self.skin_lab
        return float(np.hypot(a - 128.0, b - 128.0))

    @property
    def warmth(self) -> float:
        """Signed undertone score: positive = warm (yellow/gold), negative = cool (pink/blue).

        b-128 is how yellow (+) or blue (-) the tone is; a-128 is how red
        (+) or green (-) it is. Subtracting a scaled-down 'a' from 'b'
        biases the score toward yellow/blue (the classic warm/cool axis)
        while still letting strong redness pull it slightly cooler.
        """
        _, a, b = self.skin_lab
        return float(b - 128.0) - 0.35 * float(a - 128.0)

    def to_vector(self) -> np.ndarray:
        """Pack features into the fixed-order array the KNN model expects."""
        l, a, b = self.skin_lab
        return np.array([l, a, b, self.contrast, self.chroma], dtype=float)


def _detect_face(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Run the Haar cascade and return (x, y, w, h) of the most prominent face."""
    cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected in the uploaded image.")
    # Largest detected face wins (closest/most prominent subject).
    return max(faces, key=lambda f: f[2] * f[3])


def _clip_rect(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a sample rectangle to stay inside the image bounds (hair band can go off the top edge)."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


def _median_bgr(bgr: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Median color of a rectangle — robust to a stray highlight/shadow pixel that a mean would skew."""
    x, y, w, h = rect
    patch = bgr[y : y + h, x : x + w].reshape(-1, 3)
    return np.median(patch, axis=0)


def _bgr_to_lab(bgr: np.ndarray) -> tuple[float, float, float]:
    """Convert one averaged BGR color to Lab via OpenCV's 1x1-pixel trick (cvtColor needs an image, not a triplet)."""
    pixel = np.uint8([[bgr]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return float(lab[0]), float(lab[1]), float(lab[2])


def _bgr_to_hsv(bgr: np.ndarray) -> tuple[float, float, float]:
    """Same 1x1-pixel trick as above, converting to HSV instead."""
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def extract_color_features(bgr_image: np.ndarray) -> ColorFeatures:
    """Detect a face and extract skin/hair color features from an image.

    Raises NoFaceDetectedError if no face is found.
    """
    height, width = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    fx, fy, fw, fh = _detect_face(gray)

    # No landmark model available, so regions are just fixed proportions
    # of the face box — forehead near the top-center, cheeks lower on
    # each side, hair in a band extending above the box. Good enough for
    # a front-facing photo; a real landmark model would be more precise.
    forehead = _clip_rect(fx + int(0.30 * fw), fy + int(0.08 * fh), int(0.40 * fw), int(0.15 * fh), width, height)
    left_cheek = _clip_rect(fx + int(0.12 * fw), fy + int(0.55 * fh), int(0.20 * fw), int(0.18 * fh), width, height)
    right_cheek = _clip_rect(fx + int(0.68 * fw), fy + int(0.55 * fh), int(0.20 * fw), int(0.18 * fh), width, height)
    hair_band = _clip_rect(fx, fy - int(0.35 * fh), fw, int(0.28 * fh), width, height)

    # Median the three skin patches together so one badly-lit cheek
    # doesn't dominate the reading.
    skin_samples = np.array(
        [_median_bgr(bgr_image, forehead), _median_bgr(bgr_image, left_cheek), _median_bgr(bgr_image, right_cheek)]
    )
    skin_bgr = np.median(skin_samples, axis=0)
    hair_bgr = _median_bgr(bgr_image, hair_band)

    return ColorFeatures(
        skin_lab=_bgr_to_lab(skin_bgr),
        skin_hsv=_bgr_to_hsv(skin_bgr),
        hair_lab=_bgr_to_lab(hair_bgr),
    )


def extract_from_path(image_path: str) -> ColorFeatures:
    """Convenience entry point for scripts/tests: read a file from disk, then run the same pipeline."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    return extract_color_features(image)


def extract_from_bytes(data: bytes) -> ColorFeatures:
    """Entry point used by Flask: decode an in-memory upload (no temp file needed) and run the pipeline."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data")
    return extract_color_features(image)
