"""
ai_detector.py
----------------
Heuristic AI-generated-text detector.

IMPORTANT (read this before presenting it to clients):
This is NOT a trained classifier and does NOT match the accuracy of commercial
tools like GPTZero, Originality.ai, or Turnitin's AI detector. Those are trained
on millions of labeled human/AI samples. This module uses well-documented
*proxy signals* that correlate with AI-generated text:

  1. Burstiness        - human writing has high variance in sentence length;
                          AI text tends to be more uniform.
  2. Repetition score   - AI text often reuses phrasing/n-grams more than humans.
  3. Vocabulary richness (type-token ratio) - AI text sometimes has lower
                          lexical diversity over long passages.
  4. Sentence-opener diversity - humans vary how they start sentences more.
  5. Average word length consistency.

Each signal returns 0-100 (100 = "looks more AI-like" for that signal).
The final score is a weighted average. ALWAYS display this as a heuristic
"likelihood" with the signal breakdown, never as a definitive verdict.
"""

import re
import math
from collections import Counter
from typing import List, Dict


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text)
    # Basic sentence splitter (handles . ! ? followed by space/capital)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def burstiness_score(sentences: List[str]) -> float:
    """Lower variance in sentence length -> more AI-like -> higher score."""
    lengths = [len(split_words(s)) for s in sentences if split_words(s)]
    if len(lengths) < 3:
        return 50.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 50.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    coeff_var = std_dev / mean  # coefficient of variation

    # Human text typically has coeff_var ~0.5-0.9+; AI text often ~0.2-0.4
    # Map inversely: low coeff_var -> high AI score
    score = max(0.0, min(100.0, 100 - (coeff_var * 110)))
    return round(score, 1)


def repetition_score(words: List[str], n: int = 3) -> float:
    """Higher repeated n-gram ratio -> more AI-like -> higher score."""
    if len(words) < n + 1:
        return 50.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    ratio = repeated / len(ngrams) if ngrams else 0
    score = min(100.0, ratio * 300)
    return round(score, 1)


def vocabulary_richness_score(words: List[str]) -> float:
    """
    Type-token ratio is highly length-dependent and unreliable below ~150
    words (short texts naturally have high TTR regardless of authorship),
    so this signal only activates on longer passages and is weighted lightly
    in the overall score.
    """
    if len(words) < 150:
        return 50.0  # neutral — not enough data to trust this signal
    ttr = len(set(words)) / len(words)
    # Over longer passages, AI text tends to plateau lower (~0.35-0.45)
    # while human text stays more varied (~0.45-0.55+).
    score = max(0.0, min(100.0, (0.45 - ttr) * 200 + 50))
    return round(score, 1)


def sentence_opener_diversity_score(sentences: List[str]) -> float:
    """Low diversity in how sentences start -> more AI-like -> higher score."""
    openers = []
    for s in sentences:
        w = split_words(s)
        if w:
            openers.append(w[0])
    if len(openers) < 6:
        return 50.0  # too few sentences to trust this signal
    diversity = len(set(openers)) / len(openers)
    score = max(0.0, min(100.0, (1 - diversity) * 100))
    return round(score, 1)


def avg_word_length_consistency_score(sentences: List[str]) -> float:
    """AI text often has very consistent average word length per sentence."""
    avgs = []
    for s in sentences:
        w = split_words(s)
        if w:
            avgs.append(sum(len(x) for x in w) / len(w))
    if len(avgs) < 3:
        return 50.0
    mean = sum(avgs) / len(avgs)
    variance = sum((a - mean) ** 2 for a in avgs) / len(avgs)
    std_dev = math.sqrt(variance)
    coeff_var = std_dev / mean if mean else 0
    score = max(0.0, min(100.0, 100 - (coeff_var * 250)))
    return round(score, 1)


def formal_connective_density_score(words: List[str]) -> float:
    """Frequency of formal connectives (furthermore, moreover, etc.) and
    absence of casual/first-person markers, relative to document length.
    This directly targets the stiff, essay-like register common in
    AI-generated text, independent of sentence-length statistics."""
    if len(words) < 20:
        return 50.0

    formal_set = {
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "thus", "hence", "accordingly", "subsequently",
        "nevertheless", "nonetheless", "overall", "utilization",
        "facilitate", "facilitates", "implementation", "significant",
        "represents", "advancement", "methodology",
    }
    casual_set = {
        "honestly", "anyway", "like", "kinda", "sorta", "yeah", "lol",
        "tbh", "basically", "literally", "actually", "i", "im", "my",
        "we", "gonna", "wanna", "stuff", "things", "really", "just",
    }

    formal_hits = sum(1 for w in words if w in formal_set)
    casual_hits = sum(1 for w in words if w in casual_set)

    formal_rate = formal_hits / len(words) * 100
    casual_rate = casual_hits / len(words) * 100

    # More formal connectives -> higher score; more casual markers -> lower
    score = 50 + (formal_rate * 18) - (casual_rate * 6)
    return round(max(0.0, min(100.0, score)), 1)


def analyze_text(text: str) -> Dict:
    sentences = split_sentences(text)
    words = split_words(text)

    if len(words) < 30 or len(sentences) < 3:
        return {
            "error": "Text too short for reliable analysis. Provide at least ~30 words / 3 sentences.",
            "word_count": len(words),
            "sentence_count": len(sentences),
        }

    signals = {
        "burstiness": burstiness_score(sentences),
        "repetition": repetition_score(words),
        "vocabulary_richness": vocabulary_richness_score(words),
        "sentence_opener_diversity": sentence_opener_diversity_score(sentences),
        "word_length_consistency": avg_word_length_consistency_score(sentences),
        "formal_register": formal_connective_density_score(words),
    }

    weights = {
        "burstiness": 0.22,
        "repetition": 0.08,
        "vocabulary_richness": 0.10,
        "sentence_opener_diversity": 0.10,
        "word_length_consistency": 0.20,
        "formal_register": 0.30,
    }

    overall = sum(signals[k] * weights[k] for k in signals)
    overall = round(overall, 1)

    if overall >= 60:
        label = "Likely AI-generated"
    elif overall >= 40:
        label = "Mixed / Uncertain signals"
    else:
        label = "Likely human-written"

    return {
        "overall_score": overall,
        "label": label,
        "signals": signals,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "flagged_sentences": flag_sentences(sentences),
        "disclaimer": (
            "This is a heuristic estimate based on writing-pattern signals "
            "(sentence-length variance, repetition, vocabulary diversity, etc.), "
            "not a trained AI-detection model. Treat as a starting signal, not proof."
        ),
    }


def flag_sentences(sentences: List[str]) -> List[Dict]:
    """
    Score each sentence relative to the document's own distribution of
    sentence length and word length. AI-generated passages tend to sit in a
    narrow band of "average" sentence/word length; sentences whose length
    profile clusters tightly with the document average AND uses formal
    connective openers (furthermore, moreover, additionally...) score higher.

    This is intentionally a relative/comparative heuristic rather than an
    absolute one — it flags sentences that look uniform *relative to this
    specific document*, which works even on short passages where a sliding
    window has too few neighbors to be meaningful.
    """
    n = len(sentences)
    if n == 0:
        return []

    lengths = [len(split_words(s)) for s in sentences]
    mean_len = sum(lengths) / n
    variance = sum((l - mean_len) ** 2 for l in lengths) / n
    std_len = max(variance ** 0.5, 1e-6)

    formal_openers = {
        "furthermore", "moreover", "additionally", "consequently",
        "therefore", "thus", "hence", "accordingly", "subsequently",
        "nevertheless", "nonetheless", "overall",
    }

    casual_markers = {
        "honestly", "anyway", "like", "kinda", "sorta", "yeah", "lol",
        "tbh", "basically", "literally", "actually", "i", "my", "we",
    }

    results = []
    for i, sent in enumerate(sentences):
        words = split_words(sent)
        if not words:
            results.append({"index": i, "text": sent, "score": 50.0, "flagged": False})
            continue

        # How close is this sentence's length to the document's mean?
        z = abs(lengths[i] - mean_len) / std_len
        closeness_score = max(0.0, 100 - z * 45)

        # Formal connective opener bump (strong signal)
        opener_bump = 30 if words[0] in formal_openers else 0

        # Casual marker penalty — first-person pronouns and filler words
        # are strong human-writing signals, pull the score down when present.
        casual_hits = sum(1 for w in words if w in casual_markers)
        casual_penalty = min(casual_hits * 12, 30)

        # Average word length vs document average word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        doc_avg_word_len = sum(len(w) for s in sentences for w in split_words(s)) / max(
            sum(len(split_words(s)) for s in sentences), 1
        )
        word_len_closeness = max(0.0, 100 - abs(avg_word_len - doc_avg_word_len) * 30)

        score = (
            closeness_score * 0.25
            + word_len_closeness * 0.25
            + min(opener_bump, 30)
            - casual_penalty
        )
        score = round(max(0.0, min(score, 100.0)), 1)

        results.append({
            "index": i,
            "text": sent,
            "score": score,
            "flagged": score >= 60,
        })
    return results
