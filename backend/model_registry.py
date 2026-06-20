"""
model_registry.py
-------------------
Loads all saved (vectorizer, classifier) bundles once at startup and
exposes a simple classify() function for the live demo.
"""

import re
from pathlib import Path
from typing import Dict, List

import joblib

MODELS_DIR = Path(__file__).parent / "models"

DATASETS = {
    "sms": {
        "label": "SMS Spam Detection",
        "classes_description": "ham (legitimate) vs spam",
        "task_type": "binary",
    },
    "imdb": {
        "label": "IMDB Movie Review Sentiment",
        "classes_description": "positive vs negative",
        "task_type": "binary",
    },
    "newsgroups": {
        "label": "20 Newsgroups Topic Classification (6-category subset)",
        "classes_description": "rec.autos, sci.space, rec.sport.baseball, soc.religion.christian, talk.politics.guns, comp.graphics",
        "task_type": "multiclass",
    },
}

REPRESENTATIONS = {
    "bow": "Bag-of-Words",
    "tfidf": "TF-IDF",
    "tfidf_bigram": "TF-IDF + Bigrams",
}

CLASSIFIERS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (Linear)",
}

_registry: Dict[str, dict] = {}


def _clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_all_models():
    """Load every saved model bundle into memory once at startup."""
    count = 0
    for path in MODELS_DIR.glob("*.joblib"):
        dataset, rep, clf = path.stem.split("__")
        key = f"{dataset}__{rep}__{clf}"
        bundle = joblib.load(path)
        _registry[key] = bundle
        count += 1
    return count


def get_available_combinations(dataset: str) -> List[dict]:
    """List which representation+classifier combos are available for a dataset."""
    combos = []
    for key in _registry:
        ds, rep, clf = key.split("__")
        if ds == dataset:
            combos.append({
                "representation": rep,
                "representation_label": REPRESENTATIONS.get(rep, rep),
                "classifier": clf,
                "classifier_label": CLASSIFIERS.get(clf, clf),
            })
    return sorted(combos, key=lambda c: (c["representation"], c["classifier"]))


def classify(dataset: str, text: str) -> List[dict]:
    """
    Run the given text through every available representation+classifier
    combination for the chosen dataset. Returns predictions with confidence
    scores, sorted by representation then classifier for stable display.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    text = _clean_text(text)
    results = []

    for key, bundle in _registry.items():
        ds, rep, clf = key.split("__")
        if ds != dataset:
            continue

        vectorizer = bundle["vectorizer"]
        classifier = bundle["classifier"]

        X = vectorizer.transform([text])
        pred = classifier.predict(X)[0]

        confidence = None
        prob_breakdown = None
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X)[0]
            classes = classifier.classes_
            prob_breakdown = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, probs)}
            confidence = round(float(max(probs)) * 100, 1)

        results.append({
            "representation": rep,
            "representation_label": REPRESENTATIONS.get(rep, rep),
            "classifier": clf,
            "classifier_label": CLASSIFIERS.get(clf, clf),
            "prediction": str(pred),
            "confidence": confidence,
            "probability_breakdown": prob_breakdown,
        })

    results.sort(key=lambda r: (r["representation"], r["classifier"]))
    return results
