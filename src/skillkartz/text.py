"""Tiny dependency-free semantic-similarity layer.

The design doc (section 9) calls for "vector similarity over embedded job
postings". A real deployment would use a sentence-embedding model + a vector DB;
for a self-contained capstone we use a TF-IDF vector space with cosine
similarity, which is enough to demonstrate the *behaviour* the design depends on:
phrasing variants ("ML" vs "machine learning") and related terms ("pandas",
"scripting") pulling the right postings into the count, and the semantic-drift
failure mode the Governance Bot has to guard against.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfIndex:
    """Minimal TF-IDF index. Documents are (id, text) pairs."""

    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._vectors: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._built = False

    def build(self, documents: Iterable[tuple[str, str]]) -> "TfidfIndex":
        raw_tf: dict[str, Counter] = {}
        df: Counter = Counter()
        for doc_id, text in documents:
            toks = tokenize(text)
            tf = Counter(toks)
            raw_tf[doc_id] = tf
            for term in tf:
                df[term] += 1
            self._doc_ids.append(doc_id)

        n_docs = max(len(self._doc_ids), 1)
        self._idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1.0
            for term, count in df.items()
        }

        for doc_id, tf in raw_tf.items():
            length = sum(tf.values()) or 1
            vec = {
                term: (freq / length) * self._idf.get(term, 0.0)
                for term, freq in tf.items()
            }
            self._vectors[doc_id] = _normalize(vec)
        self._built = True
        return self

    def query_vector(self, terms: Iterable[str]) -> dict[str, float]:
        tf = Counter()
        for term in terms:
            tf.update(tokenize(term))
        length = sum(tf.values()) or 1
        vec = {
            term: (freq / length) * self._idf.get(term, 0.0)
            for term, freq in tf.items()
        }
        return _normalize(vec)

    def score(self, doc_id: str, query_vec: dict[str, float]) -> float:
        doc_vec = self._vectors.get(doc_id, {})
        if len(query_vec) > len(doc_vec):
            query_vec, doc_vec = doc_vec, query_vec
        return sum(w * doc_vec.get(term, 0.0) for term, w in query_vec.items())

    def __contains__(self, doc_id: str) -> bool:  # pragma: no cover - convenience
        return doc_id in self._vectors


def _normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm == 0:
        return vec
    return {term: w / norm for term, w in vec.items()}
