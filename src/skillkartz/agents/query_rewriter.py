"""Query Rewriter Bot - input normalisation, ahead of intent capture.

The Career Advisor Bot resolves the skill / sector / location by scanning the
raw text for *exact* known terms (a word-boundary regex). A single typo -
"pyhtin", "retial", "chicgo" - means nothing matches and the run stalls on a
clarification turn.

This bot sits in front of that step. It snaps mistyped tokens (and short
phrases) onto the closest known skill, sector or location term using a
character-level similarity ratio (`difflib.SequenceMatcher`), and hands the
Career Advisor a cleaned string plus an audit trail of what it changed.

It is deliberately conservative and fully deterministic:

* a correction only lands at or above ``SETTINGS.rewriter_similarity_floor``
  (0.82) - anything less certain is left alone for the "ask rather than guess"
  guardrail to handle;
* single tokens shorter than ``SETTINGS.rewriter_min_token_len`` and a small
  set of common query words are never touched;
* it never invents a term that is not already in the taxonomy, and it never
  looks at, or changes, any number.

Least privilege (design doc section 12): reads the taxonomy vocabulary only;
produces text and notes; cannot reach the postings index or the forecast.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..config import SETTINGS
from ..datastore import DataStore
from ..models import QueryRewrite, RewriteCorrection

# Frequent words in these queries that happen to be 4+ letters; kept off the
# correction table so "jobs", "demand" etc. are never nudged toward a skill.
_SKIP_WORDS = {
    "jobs", "role", "roles", "work", "demand", "need", "needs", "want", "wants",
    "learn", "should", "worth", "which", "what", "does", "will", "have", "into",
    "from", "with", "this", "that", "there", "their", "much", "many", "good",
    "best", "high", "some", "about", "your", "market", "career", "skill",
    "skills", "sector", "sectors", "location",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#-]*")


class QueryRewriterBot:
    name = "Query Rewriter Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store
        # vocab entry -> (canonical form, category)
        self._vocab: dict[str, tuple[str, str]] = {}

        for canonical, meta in store.canonical_skills().items():
            self._add(canonical, canonical, "skill")
            for syn in meta.get("synonyms", []):
                self._add(syn, canonical, "skill")
        for sector in store.sectors():
            self._add(sector, sector, "sector")
        for loc in store.locations():
            self._add(loc, loc, "location")

        # Spaceless aliases for multi-word terms so a single mistyped token
        # ("newyork", "powerbi") still resolves to the spaced canonical form
        # the downstream word-boundary matcher expects. Tracked separately so
        # the rewrite loop still emits a correction for them.
        self._alias_keys: set[str] = set()
        for key, (canonical, category) in list(self._vocab.items()):
            if " " in key:
                alias = key.replace(" ", "")
                if alias not in self._vocab:
                    self._vocab[alias] = (canonical, category)
                    self._alias_keys.add(alias)

        self._max_ngram = max((len(k.split()) for k in self._vocab), default=1)
        # phrases grouped by word count, for windowed matching
        self._by_len: dict[int, list[str]] = {}
        for key in self._vocab:
            self._by_len.setdefault(len(key.split()), []).append(key)

    def _add(self, phrase: str, canonical: str, category: str) -> None:
        self._vocab[phrase.strip().lower()] = (canonical, category)

    # ------------------------------------------------------------------ #
    def rewrite(self, raw_text: str) -> QueryRewrite:
        text = raw_text or ""
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

        corrections: list[RewriteCorrection] = []
        spans: list[tuple[int, int, str]] = []  # (char_start, char_end, replacement)

        i = 0
        while i < len(tokens):
            advanced = False
            # try the longest phrase window first so "data analsis" beats "analsis"
            for n in range(min(self._max_ngram, len(tokens) - i), 0, -1):
                window = tokens[i:i + n]
                phrase = " ".join(w.lower() for w, _, _ in window)

                if phrase in self._vocab and phrase not in self._alias_keys:
                    i += n  # already a known, canonical term - leave it
                    advanced = True
                    break

                if n == 1 and (
                    len(phrase) < SETTINGS.rewriter_min_token_len
                    or phrase in _SKIP_WORDS
                ):
                    continue

                match = self._closest(phrase, n)
                if match is None:
                    continue
                key, ratio = match
                canonical, category = self._vocab[key]
                char_start, char_end = window[0][1], window[-1][2]
                original_span = text[char_start:char_end]
                corrections.append(RewriteCorrection(
                    span=original_span, replacement=canonical,
                    category=category, similarity=round(ratio, 3)))
                spans.append((char_start, char_end, canonical))
                i += n
                advanced = True
                break

            if not advanced:
                i += 1

        normalized = self._splice(text, spans)
        notes = [
            f"read '{c.span}' as the {c.category} '{c.replacement}' "
            f"({c.similarity:.2f} similar)"
            for c in corrections
        ]
        return QueryRewrite(
            original_text=raw_text,
            normalized_text=normalized,
            corrections=corrections,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    def _closest(self, phrase: str, n: int) -> tuple[str, float] | None:
        best_key: str | None = None
        best_ratio = 0.0
        for key in self._by_len.get(n, ()):
            ratio = SequenceMatcher(None, phrase, key).ratio()
            if ratio > best_ratio:
                best_key, best_ratio = key, ratio
        if best_key is not None and best_ratio >= SETTINGS.rewriter_similarity_floor:
            return best_key, best_ratio
        return None

    @staticmethod
    def _splice(text: str, spans: list[tuple[int, int, str]]) -> str:
        if not spans:
            return text
        spans.sort()
        out: list[str] = []
        cursor = 0
        for start, end, replacement in spans:
            out.append(text[cursor:start])
            out.append(replacement)
            cursor = end
        out.append(text[cursor:])
        return "".join(out)
