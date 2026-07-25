from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from personalcolor.features import NoFaceDetectedError, extract_from_bytes
from personalcolor.model import get_classifier
from personalcolor.palette import get_palette

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB — reject huge uploads before they hit memory/OpenCV

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed_file(filename: str) -> bool:
    """Extension allow-list check (defense in depth — imdecode failure is the real safety net)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _analyze(image_bytes: bytes) -> dict:
    """Shared pipeline for both the HTML and JSON endpoints: bytes in, JSON-serializable dict out.

    Raises NoFaceDetectedError / ValueError, which each route translates
    into its own error response format (HTML flash vs. JSON body).
    """
    color_features = extract_from_bytes(image_bytes)
    prediction = get_classifier().predict(color_features)
    palette = get_palette(prediction.season)
    return {
        "season": prediction.season,
        "confidence": round(prediction.confidence, 3),
        "probabilities": {k: round(v, 3) for k, v in prediction.probabilities.items()},
        "tagline": palette.tagline,
        "swatches": list(palette.swatches),
        # Raw features included so the result page (and API consumers)
        # can show *why* a season was picked, not just the label —
        # important for a model whose "accuracy" is a documented caveat.
        "features": {
            "skin_lab": [round(v, 1) for v in color_features.skin_lab],
            "hair_lab": [round(v, 1) for v in color_features.hair_lab],
            "contrast": round(color_features.contrast, 1),
            "chroma": round(color_features.chroma, 1),
            "warmth": round(color_features.warmth, 1),
        },
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Browser-facing endpoint: renders the upload form again with an error, or the result page."""
    file = request.files.get("photo")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image to upload.")
    if not _allowed_file(file.filename):
        return render_template("index.html", error="Please upload a PNG or JPEG image.")

    try:
        result = _analyze(file.read())
    except NoFaceDetectedError:
        return render_template("index.html", error="No face detected — try a clearer, front-facing photo.")
    except ValueError as exc:
        return render_template("index.html", error=str(exc))

    return render_template("result.html", result=result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON twin of /analyze for programmatic callers — same pipeline, HTTP status codes instead of pages."""
    file = request.files.get("photo")
    if file is None or file.filename == "":
        return jsonify({"error": "no file provided"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "unsupported file type"}), 400

    try:
        result = _analyze(file.read())
    except NoFaceDetectedError as exc:
        return jsonify({"error": str(exc)}), 422  # 422: well-formed request, semantically unprocessable
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Liveness check for whatever hosts this (load balancer, uptime monitor, etc.)."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
