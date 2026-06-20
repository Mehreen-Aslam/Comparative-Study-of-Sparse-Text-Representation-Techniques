"""
main.py
--------
FastAPI backend for the research paper demo dashboard.

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model_registry as mr
from ai_detector import analyze_text

app = FastAPI(title="Sparse Representations Research Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_PATH = Path(__file__).parent / "results.json"
with open(RESULTS_PATH) as f:
    PAPER_RESULTS = json.load(f)

model_count = mr.load_all_models()
print(f"Loaded {model_count} trained models for live classification.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    dataset: str
    text: str


class DetectRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Routes — paper results
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "Sparse Representations Research Demo"}


@app.get("/api/results")
def get_results():
    """Full 45-condition results table from training."""
    return {"results": PAPER_RESULTS}


@app.get("/api/results/summary")
def get_results_summary():
    """Aggregate summary stats for the key-findings charts."""
    by_rep = {}
    by_clf = {}
    for r in PAPER_RESULTS:
        if "error" in r:
            continue
        rep = r["representation"]
        clf = r["classifier"]
        by_rep.setdefault(rep, []).append(r["accuracy"])
        by_clf.setdefault(clf, []).append(r["accuracy"])

    rep_summary = [
        {"representation": rep, "avg_accuracy": round(sum(vals) / len(vals), 2)}
        for rep, vals in by_rep.items()
    ]
    clf_summary = [
        {"classifier": clf, "avg_accuracy": round(sum(vals) / len(vals), 2)}
        for clf, vals in by_clf.items()
    ]

    return {
        "by_representation": sorted(rep_summary, key=lambda x: -x["avg_accuracy"]),
        "by_classifier": sorted(clf_summary, key=lambda x: -x["avg_accuracy"]),
    }


# ---------------------------------------------------------------------------
# Routes — live classification demo
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def get_datasets():
    """List datasets available for the live demo, with available model combos."""
    out = []
    for key, meta in mr.DATASETS.items():
        out.append({
            "id": key,
            "label": meta["label"],
            "classes_description": meta["classes_description"],
            "task_type": meta["task_type"],
            "available_combinations": mr.get_available_combinations(key),
        })
    return {"datasets": out}


@app.post("/api/classify")
def classify_text(req: ClassifyRequest):
    """Run text through all representation+classifier combos for a dataset."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    if len(req.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text is too short to classify meaningfully.")
    try:
        results = mr.classify(req.dataset, req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "dataset": req.dataset,
        "text": req.text,
        "predictions": results,
    }


# ---------------------------------------------------------------------------
# Routes — AI detector (heuristic, carried over from prior tool)
# ---------------------------------------------------------------------------

@app.post("/api/detect")
def detect_ai_text(req: DetectRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    result = analyze_text(req.text)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
