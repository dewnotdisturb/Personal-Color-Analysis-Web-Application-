# Personal Color Analysis Web App

Flask web app that classifies an uploaded photo into one of the four
classic **seasonal color palettes** (Spring / Summer / Autumn / Winter)
using OpenCV feature extraction and a scikit-learn KNN classifier.

Upload a front-facing photo → the app detects your face, samples skin
and hair tone, converts them to perceptually-uniform CIELAB color space,
and classifies the result into a season with a recommended color
palette.

## How it works

**1. Feature extraction (OpenCV)** — `personalcolor/features.py`

- Detect the face with a Haar cascade (`haarcascade_frontalface_default.xml`,
  ships with `opencv-python`).
- Sample proportional sub-regions of the face box: forehead + both
  cheeks for skin, a band above the face for hair. This is an
  approximation, not landmark-based sampling (see Limitations).
- Reduce each region to a robust color via the pixel **median** (resists
  outliers from specular highlights/shadows better than a mean).
- Convert to **CIELAB** (perceptually uniform — good for measuring
  undertone and lightness) and derive:
  - **lightness** (skin `L`)
  - **warmth** (undertone, from `a`/`b`)
  - **chroma** (saturation, `a`/`b` radius)
  - **contrast** (skin vs. hair lightness gap)

**2. Classification (scikit-learn)** — `personalcolor/train.py`, `model.py`

A `StandardScaler` + `KNeighborsClassifier(n_neighbors=9, weights="distance")`
maps the 5-dimensional feature vector `[L, a, b, contrast, chroma]` to a
season.

**3. Serving (Flask)** — `app.py`

- `GET /` — upload form
- `POST /analyze` — HTML result page (season, confidence, palette swatches)
- `POST /api/analyze` — same pipeline as JSON, for programmatic use
- `GET /health` — liveness check

## ⚠️ Read this before quoting an accuracy number

There is no large public dataset of photos labeled with a "correct"
personal-color-analysis season — it's an inherently subjective,
consultant-judged label, not an objective ground truth like a cat/dog
photo. Rather than fake that, **the training set is synthetic**,
generated directly from the published color-theory decision rules:

```
              warm            cool
  light     Spring          Summer
  deep      Autumn          Winter

  Spring = warm + light + clear (high chroma)
  Summer = cool + light + soft  (low chroma)
  Autumn = warm + deep  + soft  (low chroma)
  Winter = cool + deep  + clear (high chroma)
```

Each season is a Gaussian cluster in `[lightness, undertone-hue, chroma,
contrast]` space (see `_SEASON_PARAMS` in `train.py`), with enough noise
that classes overlap at the boundaries the way real people's coloring
does. The held-out classification report the training script prints
(~90% accuracy, saved to `model/eval_report.txt`) measures **whether KNN
can recover a theory-defined decision boundary from noisy samples** — a
sanity check on the feature design, **not a claim about real-world
classification accuracy against expert human judgment.**

The natural next step — swapping in genuine expert-labeled photos through
the same feature pipeline — is exactly the kind of "what would you do
differently in production" answer this is designed to prompt in an
interview.

## Known limitations (by design, for a portfolio-scoped project)

- **Region sampling is proportional, not landmark-based.** A production
  version would use a facial landmark model (e.g. mediapipe face mesh)
  to sample actual cheek/forehead/iris pixels instead of fixed
  proportions of the face bounding box.
- **No eye-color feature.** Eye color is part of real personal-color
  consultations; it's omitted here because reliable eye localization
  needs landmarks, not just a Haar face box.
- **Single photo, uncontrolled lighting.** No white-balance calibration
  step — a strongly warm/cool ambient light will bias the undertone
  reading. Flagged as a real accuracy risk, not silently ignored.
- **4-season model, not 12/16-tone.** The simplification is standard
  practice for an initial consultation; documented as the extension point.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the KNN model (also runs automatically on first request if missing)
python -m personalcolor.train

# Run the app
python app.py    # http://127.0.0.1:5000
```

## Deployment

Deployed on [Render](https://render.com) using the `render.yaml` blueprint at
the repo root (`rootDir: personalcolor` points Render at this subfolder).
Gunicorn serves the app in production instead of Flask's dev server.

To deploy your own copy:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. In the Render dashboard: **New > Blueprint**, connect the repo, and Render
   picks up `render.yaml` automatically — free plan, build command
   `pip install -r requirements.txt`, start command
   `gunicorn app:app --bind 0.0.0.0:$PORT`, health check on `/health`.
3. First deploy takes a few minutes (installing OpenCV/scikit-learn). The KNN
   model trains automatically on the first request if `model/*.joblib` isn't
   present (it's gitignored — see `personalcolor/model.py:_ensure_trained`).
4. Render's free tier spins down after inactivity, so the first request after
   a period of idleness will be slow (cold start), not just an accuracy issue.

No manual dashboard config needed beyond connecting the repo — `render.yaml`
defines everything.

## Tests

```bash
pytest
```

Covers: color-feature math (contrast/chroma/warmth formulas), the
synthetic dataset generator, model training + artifact persistence, and
Flask routes (including the no-face-detected and API error paths).

## Project structure

```
app.py                      Flask routes
personalcolor/
  features.py                OpenCV face/region detection + Lab feature extraction
  palette.py                 Season → palette swatch definitions
  train.py                   Synthetic dataset + KNN training
  model.py                   Load trained model, predict
templates/, static/          UI
tests/                       pytest suite
model/                       Trained model artifacts (knn_model.joblib, scaler.joblib, eval_report.txt)
```
