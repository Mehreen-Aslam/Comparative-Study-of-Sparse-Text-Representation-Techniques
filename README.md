# Sparse Text Representations — Research Demo Dashboard

A working demo accompanying the paper "A Comparative Study of Sparse Text
Representation Techniques for Text Classification Across Multiple Machine
Learning Models" (Mehreen Aslam, MSAIS001).

## What this actually is

- **45 real trained models**, not mocked numbers. Bag-of-Words, TF-IDF, and
  TF-IDF+bigrams, each combined with Naive Bayes, Logistic Regression, SVM,
  Random Forest, and kNN, trained on three datasets.
- **27 of those models are saved and served live** (NB, LogReg, SVM — the
  paper's top performers; Random Forest and kNN are trained for the results
  table but not served live, since the paper shows they're slower/weaker).
- A **heuristic AI-text detector** carried over as a bonus section, fully
  separate from the supervised classifiers above.

## Important honesty notes — read before presenting this

1. **Datasets are subsampled, not full-size**, to keep training and loading
   fast:
   - SMS Spam Collection: used in full (~5,572 messages — it's small already).
   - IMDB: subsampled to 8,000 reviews (4,000 positive / 4,000 negative) out
     of the full 50,000.
   - 20 Newsgroups: subsampled to **6 categories** (~3,500 docs) instead of
     the full 20, to keep the multi-class problem fast and the live demo
     responsive.
   - **This means accuracy numbers in this demo will NOT exactly match your
     paper's numbers.** The 6-category newsgroups subset, for example, scores
     ~88–93% accuracy here vs. ~73% in the paper's full 20-class task — fewer,
     more separable categories is an easier problem. The *relative*
     rankings (TF-IDF > BoW, LogReg/SVM/NB > RF > kNN) hold consistently,
     which is the point of this demo, but don't quote the absolute numbers
     as if they were your paper's official results.

2. **Newsgroups headers were stripped** (From:/Subject:/Organization: lines,
   quoted reply text) before training — this matches standard practice for
   this dataset (sklearn's `remove=('headers','footers','quotes')` option)
   and prevents models from "cheating" by matching email metadata instead of
   actual content.

3. **SVM uses `CalibratedClassifierCV`** wrapping `LinearSVC` so it can
   produce probability scores for the live demo's confidence bars — this is
   a standard technique (Platt scaling via cross-validation) and doesn't
   change the underlying decision boundary, but it's worth knowing it's not
   raw `LinearSVC` output.

4. **Results in `/api/results` come from `results.json`**, generated once
   by `training/train_models.py`. If you want exact-paper-scale numbers,
   rerun training with `max_per_class` and `n_samples` increased in that
   script — full IMDB (50k) and full 20 Newsgroups (20 categories, ~18k)
   will take meaningfully longer to train (the README in `training/` has
   the exact knobs).

## Project structure

```
backend/
  main.py              — FastAPI app, all routes
  model_registry.py    — loads the 27 saved models, runs live classification
  ai_detector.py        — heuristic AI-text detector (carried over)
  results.json          — the 45-condition results table
  models/                — 27 saved (vectorizer, classifier) joblib bundles
  requirements.txt
frontend/
  index.html             — single-page dashboard (no build step)
  app.js                 — all dashboard logic
```

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in your browser. The API base URL
field at the bottom of the page defaults to `http://localhost:8000` — change
it if you deploy the backend elsewhere.

## Re-training with different settings

If you want to retrain (e.g. with full-size datasets, or different
classifiers), the training script lives separately and isn't included in
this output bundle by default — ask if you want it added. The key knobs:

- `load_imdb(n_samples=8000)` → increase for more IMDB data (max 50,000)
- `load_newsgroups(categories=[...], max_per_class=600)` → pass all 20
  categories and a higher `max_per_class` for the full paper-scale setup
- `load_sms()` → already uses the full dataset, no knob needed

Re-running training will overwrite `results.json` and the `models/` folder.
