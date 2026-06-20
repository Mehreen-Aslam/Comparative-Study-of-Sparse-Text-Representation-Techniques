"""
train_models.py
-----------------
Trains real models replicating the paper's experimental design, on
subsampled data for fast training. Saves:
  - models/*.joblib       (vectorizer + classifier pairs, for live serving)
  - results.json          (accuracy/precision/recall/F1/train_time per condition)

Design choices vs the paper (documented honestly):
  - Datasets subsampled for speed: SMS used in full (~5.5k, already small),
    IMDB subsampled to 8,000 reviews, 20 Newsgroups subsampled to 6 categories
    (~3,500 docs) instead of all 20, to keep training fast while preserving
    a genuine multi-class problem.
  - Representations: BoW, TF-IDF, TF-IDF+bigrams (same as paper).
  - Classifiers trained for ALL representations (replicates paper's full
    grid): Naive Bayes, Logistic Regression, SVM, Random Forest, kNN.
  - Only Naive Bayes, Logistic Regression, and SVM are saved/served for the
    LIVE demo (paper shows Random Forest/kNN underperform and are slower;
    their numbers still appear in the results table from full training here).
"""

import json
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
LIVE_SERVE_CLASSIFIERS = {"naive_bayes", "logistic_regression", "svm"}


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)  # strip HTML (IMDB has <br/> tags)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_sms():
    df = pd.read_csv(DATA_DIR / "sms_spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["text"] = df["text"].apply(clean_text)
    return df["text"].tolist(), df["label"].tolist()


def load_imdb(n_samples=8000):
    df = pd.read_csv(DATA_DIR / "imdb.csv")
    per_class = n_samples // 2
    pos = df[df["sentiment"] == "positive"].sample(n=per_class, random_state=RANDOM_SEED)
    neg = df[df["sentiment"] == "negative"].sample(n=per_class, random_state=RANDOM_SEED)
    sub = pd.concat([pos, neg], ignore_index=True)
    sub["review"] = sub["review"].apply(clean_text)
    return sub["review"].tolist(), sub["sentiment"].tolist()


def strip_newsgroup_headers(text: str) -> str:
    """
    Remove email-style headers (From:, Subject:, Lines:, etc.) and quoted
    reply lines (starting with '>'), matching the standard preprocessing
    convention for this dataset (sklearn's `remove=('headers','footers',
    'quotes')` option) — otherwise classifiers can trivially "cheat" by
    matching sender addresses or organization names instead of content.
    """
    lines = text.split("\n")
    body_lines = []
    in_header = True
    for line in lines:
        if in_header:
            if re.match(r"^[A-Za-z-]+:\s", line) or line.strip() == "":
                continue
            else:
                in_header = False
        if line.strip().startswith(">") or "writes:" in line.lower():
            continue
        body_lines.append(line)
    return " ".join(body_lines)


def load_newsgroups(categories=None, max_per_class=600):
    """
    Note on this JSON mirror's structure: despite the key name, `target_names`
    here is NOT an id->category lookup table — it's already the resolved
    string label for each document (parallel to `target`, which holds the
    same label as an integer id). So we use `target_names[key]` directly as
    each document's category string.
    """
    with open(DATA_DIR / "newsgroups.json") as f:
        data = json.load(f)
    content = data["content"]
    target_names = data["target_names"]

    if categories is None:
        categories = ["sci.space", "rec.sport.baseball", "talk.politics.guns",
                      "comp.graphics", "soc.religion.christian", "rec.autos"]
    categories_set = set(categories)

    texts, labels = [], []
    counts = {c: 0 for c in categories}
    for key in content.keys():
        label = target_names.get(key)
        if label not in categories_set:
            continue
        if counts[label] >= max_per_class:
            continue
        texts.append(clean_text(strip_newsgroup_headers(content[key])))
        labels.append(label)
        counts[label] += 1

    return texts, labels


def build_vectorizer(rep_type: str):
    if rep_type == "bow":
        return CountVectorizer(stop_words="english", max_features=10000)
    elif rep_type == "tfidf":
        return TfidfVectorizer(stop_words="english", max_features=10000, sublinear_tf=True)
    elif rep_type == "tfidf_bigram":
        return TfidfVectorizer(stop_words="english", max_features=10000,
                                sublinear_tf=True, ngram_range=(1, 2))
    raise ValueError(rep_type)


def build_classifier(clf_type: str):
    if clf_type == "naive_bayes":
        return MultinomialNB()
    elif clf_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    elif clf_type == "svm":
        # LinearSVC has no predict_proba; calibrate for probability outputs
        # needed by the live demo's "confidence" display.
        base = LinearSVC(random_state=RANDOM_SEED, max_iter=5000)
        return CalibratedClassifierCV(base, cv=3)
    elif clf_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    elif clf_type == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine")
    raise ValueError(clf_type)


def run_condition(dataset_name, texts, labels, rep_type, clf_type):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )

    vectorizer = build_vectorizer(rep_type)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = build_classifier(clf_type)

    start = time.time()
    clf.fit(X_train_vec, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    result = {
        "dataset": dataset_name,
        "representation": rep_type,
        "classifier": clf_type,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "train_time_seconds": round(train_time, 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save model if it's one we serve live
    if clf_type in LIVE_SERVE_CLASSIFIERS:
        model_path = MODELS_DIR / f"{dataset_name}__{rep_type}__{clf_type}.joblib"
        joblib.dump({"vectorizer": vectorizer, "classifier": clf}, model_path)

    return result


def main():
    print("Loading datasets...")
    sms_texts, sms_labels = load_sms()
    print(f"  SMS: {len(sms_texts)} docs, classes: {set(sms_labels)}")

    imdb_texts, imdb_labels = load_imdb(n_samples=8000)
    print(f"  IMDB: {len(imdb_texts)} docs, classes: {set(imdb_labels)}")

    ng_texts, ng_labels = load_newsgroups()
    print(f"  20 Newsgroups (6-category subset): {len(ng_texts)} docs, classes: {set(ng_labels)}")

    datasets = {
        "sms": (sms_texts, sms_labels),
        "imdb": (imdb_texts, imdb_labels),
        "newsgroups": (ng_texts, ng_labels),
    }

    representations = ["bow", "tfidf", "tfidf_bigram"]
    classifiers = ["naive_bayes", "logistic_regression", "svm", "random_forest", "knn"]

    results = []
    total = len(datasets) * len(representations) * len(classifiers)
    count = 0

    for ds_name, (texts, labels) in datasets.items():
        for rep in representations:
            for clf in classifiers:
                count += 1
                print(f"[{count}/{total}] {ds_name} | {rep} | {clf} ...", end=" ", flush=True)
                try:
                    result = run_condition(ds_name, texts, labels, rep, clf)
                    results.append(result)
                    print(f"acc={result['accuracy']}% f1={result['f1']}% time={result['train_time_seconds']}s")
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append({
                        "dataset": ds_name, "representation": rep, "classifier": clf,
                        "error": str(e),
                    })

    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")
    print(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
