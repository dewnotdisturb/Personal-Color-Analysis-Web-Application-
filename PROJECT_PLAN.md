# ShadeMatch — ML Engineering Portfolio Project

## 1. The problem

Foundation and concealer shade mismatch is one of the highest-cost failure
modes in beauty e-commerce:

- Online cosmetics return rates run **20–30%+**, and shade mismatch is
  consistently the #1 cited reason in retailer post-purchase surveys.
- Every mismatched shade costs a brand shipping (both ways), restocking,
  and — the harder cost to recover — customer trust. Shoppers who get
  burned once rarely buy foundation online from that brand again.
- The industry's own watershed moment (Fenty Beauty's 40-shade launch,
  2017) proved this wasn't a hard technical limit, it was a **failure to
  measure and design for the full range of skin tones**. Most brands still
  ship narrow, poorly-normalized shade ranges, and there is no shared
  standard across brands — "Fenty 250" and "Fenty Deep 4" describe
  different points in color space, with no crosswalk between them.

This makes it a project that scores on three axes recruiters actually
care about at mid-level:

1. **Business impact is quantifiable** (returns, conversion, CAC payback).
2. **The hard part is real engineering**, not a Kaggle leaderboard —
   lighting normalization, cross-brand data reconciliation, and
   calibration are all non-trivial.
3. **It has a responsible-ML story**: you can explicitly measure and
   report accuracy *by skin tone group*, which is exactly the kind of
   rigor that separates a mid-level candidate from a tutorial-follower.

## 2. What you're building

**ShadeMatch**: given a selfie, return the top-N best-matching foundation
shades across several brands, with a documented, auditable accuracy
breakdown by skin-tone group.

Two things must both be true for this to be credible:
- It has to actually work end-to-end (upload photo → get real
  recommendations from real product data).
- It has to be honest about where it's weak (report accuracy gaps, don't
  hide them — this is the differentiator over a typical portfolio piece).

## 3. Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Data layer  │────▶│  ML pipeline      │────▶│  Serving layer      │
│             │     │                    │     │                     │
│ Shade DB    │     │ Face/skin detect   │     │ FastAPI             │
│ (multi-     │     │ Illumination norm  │     │  POST /match        │
│  brand,     │     │ Color extraction   │     │  -> top-3 shades/   │
│  normalized │     │ Lab-space matching │     │     brand           │
│  to Lab     │     │                    │     │                     │
│  color      │     │ Eval harness       │     │ Minimal demo UI     │
│  space)     │     │ (per-Fitzpatrick-  │     │ (upload -> results) │
│             │     │  group accuracy)   │     │                     │
└─────────────┘     └──────────────────┘     └────────────────────┘
```

### 3.1 Data layer
- **Shade data**: product name, brand, shade name/number, and Lab/RGB
  values for each shade. Sourced from public product pages or an
  open dataset (e.g. existing Sephora/Ulta shade-range scrapes on
  Kaggle/GitHub) — avoid live scraping brittle retailer sites for a
  portfolio project unless you want to maintain it.
- **Normalization is the actual data-engineering work here**: every
  brand names/numbers shades differently. The pipeline converts each
  brand's stated shade (however it's described — hex, RGB, or just a
  product photo swatch) into a common **CIELAB** representation, since
  Lab space is perceptually uniform (better for "nearest shade" than RGB).
- **Face/skin-tone training & eval data**: use an existing labeled
  dataset with Fitzpatrick skin-type annotations (e.g. academic
  dermatology/CV datasets built for this purpose) rather than collecting
  your own biometric data — cleaner ethically and legally for a public
  portfolio repo.

### 3.2 ML pipeline
1. Face detection (off-the-shelf, e.g. mediapipe/OpenCV) → sample skin
   regions (cheek, forehead), avoiding eyes/lips/shadows.
2. Illumination/white-balance correction — this is the single biggest
   accuracy lever and the most "real engineering" part of the project.
   Naive RGB averaging under a yellow indoor light vs. daylight gives
   wildly different results; you'll need a calibration step (e.g. a
   gray-card or auto white-balance heuristic).
3. Convert sampled skin color to Lab space.
4. Match: calibrated k-NN against the normalized shade DB. (Start here —
   it's interpretable and easy to evaluate. A learned embedding model is
   a legitimate v2 if you want more "ML" on the resume, but don't start
   there; a well-evaluated simple model beats a poorly-evaluated fancy one.)
5. **Evaluation harness**: accuracy (is the true/closest shade in the
   top-3?) computed **overall and broken out by Fitzpatrick group**. This
   report is a first-class deliverable, not an afterthought — it's the
   artifact you screenshot for your portfolio page and talk through in
   interviews.

### 3.3 Serving layer
- FastAPI service, single `/match` endpoint, containerized (Docker).
- Minimal front end (even a bare HTML upload form is fine) so recruiters
  can actually try it live, not just read code.
- Deploy somewhere free/cheap (Fly.io, Render, HF Spaces) so the demo
  link works when a recruiter clicks it.

## 4. Tech stack

- **Python**: pandas/numpy, OpenCV or mediapipe, scikit-learn (k-NN +
  calibration), FastAPI, pytest.
- **Data**: Postgres (or DuckDB for simplicity) for the shade DB, a small
  dbt project if you want to show SQL/warehouse modeling skill alongside
  the ML — optional but strengthens the "data engineer" angle too.
- **Infra**: Docker, GitHub Actions (lint + test on PR), deployed demo.

## 5. Milestone roadmap

| Milestone | Deliverable |
|---|---|
| M1 | Shade DB: ingest + normalize 3–4 brands' shade ranges into Lab space; basic data-quality checks |
| M2 | Face/skin detection + color extraction pipeline on a single test image, notebook-level |
| M3 | Illumination correction; before/after accuracy comparison |
| M4 | k-NN matcher + eval harness with per-Fitzpatrick-group breakdown |
| M5 | FastAPI service wrapping the pipeline; Dockerized |
| M6 | Deployed demo + README with the bias report front and center |
| M7 (stretch) | Swap k-NN for a learned embedding model; A/B the two in the eval report |

## 6. What goes on the portfolio page / resume

- One-line pitch: *"Built and deployed a foundation shade-matching system
  addressing a top driver of beauty e-commerce returns; measured and
  reported accuracy across skin-tone groups to surface and quantify
  model bias."*
- Link to live demo + repo.
- The bias/eval report as a pinned image — this is the single artifact
  most likely to make a hiring manager stop scrolling.

## 7. Explicitly out of scope (say so if asked)

- Collecting your own face-image dataset (ethical/legal overhead not
  worth it for a portfolio project — use existing labeled datasets).
- Supporting every brand/every shade (pick 3–4 well-documented brands;
  breadth is not the point, correctness and honesty about limitations is).
- Production-grade auth/rate-limiting on the demo API (mention it as a
  "next steps" line, don't build it).
