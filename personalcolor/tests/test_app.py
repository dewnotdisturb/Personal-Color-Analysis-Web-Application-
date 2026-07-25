import cv2
import numpy as np
import pytest

import app as app_module
from personalcolor.features import ColorFeatures
from personalcolor.model import Prediction


@pytest.fixture()
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


def _blank_png_bytes() -> bytes:
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    return buf.tobytes()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_index_get(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Personal Color Analysis" in resp.data


def test_analyze_missing_file(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"choose an image" in resp.data


def test_analyze_no_face_detected(client):
    data = {"photo": (__import__("io").BytesIO(_blank_png_bytes()), "blank.png")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"No face detected" in resp.data


def test_api_analyze_no_face_detected_returns_422(client):
    data = {"photo": (__import__("io").BytesIO(_blank_png_bytes()), "blank.png")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 422
    assert "error" in resp.get_json()


def test_api_analyze_rejects_bad_extension(client):
    data = {"photo": (__import__("io").BytesIO(b"not an image"), "notes.txt")}
    resp = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_analyze_success_path(client, monkeypatch):
    fake_features = ColorFeatures(skin_lab=(180.0, 132.0, 150.0), skin_hsv=(20, 60, 200), hair_lab=(90.0, 128.0, 128.0))
    fake_prediction = Prediction(
        season="Spring",
        confidence=0.87,
        probabilities={"Spring": 0.87, "Summer": 0.05, "Autumn": 0.05, "Winter": 0.03},
    )

    class FakeClassifier:
        def predict(self, _features):
            return fake_prediction

    monkeypatch.setattr(app_module, "extract_from_bytes", lambda _data: fake_features)
    monkeypatch.setattr(app_module, "get_classifier", lambda: FakeClassifier())

    data = {"photo": (__import__("io").BytesIO(b"irrelevant-bytes-since-extraction-is-mocked"), "photo.jpg")}
    resp = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Spring" in resp.data
    assert b"87.0%" in resp.data
