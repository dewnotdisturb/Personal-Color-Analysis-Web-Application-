import numpy as np
import pytest

from personalcolor import features
from personalcolor.features import ColorFeatures, NoFaceDetectedError


def test_no_face_detected_raises_on_blank_image():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(NoFaceDetectedError):
        features.extract_color_features(blank)


def test_color_features_contrast_is_lightness_gap():
    cf = ColorFeatures(skin_lab=(180.0, 130.0, 140.0), skin_hsv=(20.0, 60.0, 200.0), hair_lab=(90.0, 128.0, 128.0))
    assert cf.contrast == pytest.approx(90.0)


def test_color_features_chroma_is_ab_radius():
    cf = ColorFeatures(skin_lab=(150.0, 128.0 + 3.0, 128.0 + 4.0), skin_hsv=(0, 0, 0), hair_lab=(100.0, 128.0, 128.0))
    assert cf.chroma == pytest.approx(5.0)  # 3-4-5 triangle


def test_color_features_warmth_sign():
    warm = ColorFeatures(skin_lab=(160.0, 128.0, 150.0), skin_hsv=(0, 0, 0), hair_lab=(100.0, 128.0, 128.0))
    cool = ColorFeatures(skin_lab=(160.0, 150.0, 118.0), skin_hsv=(0, 0, 0), hair_lab=(100.0, 128.0, 128.0))
    assert warm.warmth > 0
    assert cool.warmth < 0


def test_to_vector_order_and_values():
    cf = ColorFeatures(skin_lab=(180.0, 129.0, 132.0), skin_hsv=(0, 0, 0), hair_lab=(120.0, 128.0, 128.0))
    vec = cf.to_vector()
    assert vec.shape == (5,)
    assert vec[0] == pytest.approx(180.0)
    assert vec[1] == pytest.approx(129.0)
    assert vec[2] == pytest.approx(132.0)
    assert vec[3] == pytest.approx(cf.contrast)
    assert vec[4] == pytest.approx(cf.chroma)


def test_clip_rect_stays_within_bounds():
    x, y, w, h = features._clip_rect(-10, -10, 50, 50, width=40, height=40)
    assert x >= 0 and y >= 0
    assert x + w <= 40
    assert y + h <= 40


def test_extract_color_features_with_mocked_face(monkeypatch):
    # Build a synthetic image with a warm-toned upper "skin" block and a
    # darker "hair" block above it, then force face detection onto the
    # skin block so we can test the rest of the pipeline deterministically.
    image = np.zeros((300, 200, 3), dtype=np.uint8)
    image[:, :] = (60, 90, 150)  # BGR: warm skin-ish tone everywhere
    image[0:60, :] = (30, 30, 30)  # darker band above, stands in for hair

    monkeypatch.setattr(features, "_detect_face", lambda gray: (20, 80, 160, 160))

    result = features.extract_color_features(image)
    assert isinstance(result, ColorFeatures)
    assert result.skin_lab[0] > 0
    assert result.contrast >= 0
