"""Skill Intelligence Bot - normalisation and synonym resolution.

Design doc section 9: "vector similarity over a skill taxonomy to resolve
synonyms and phrasing variants ('ML' <-> 'Machine Learning') ... before a query
reaches the postings index". Least-privilege (section 12): this bot reads the
taxonomy and proposes query terms; it cannot write to the postings index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..datastore import DataStore
from ..text import tokenize


@dataclass
class SkillResolution:
    input_skill: str
    canonical: Optional[str]
    matched_via: str  # "canonical" | "synonym" | "fuzzy" | "unresolved"
    synonyms: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    exact_skill_names: set = field(default_factory=set)

    @property
    def resolved(self) -> bool:
        return self.canonical is not None


class SkillIntelligenceBot:
    name = "Skill Intelligence Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def resolve(self, skill: str) -> SkillResolution:
        canonical_skills = self.store.canonical_skills()
        skill_low = (skill or "").strip().lower()

        # 1. exact canonical match
        for canonical in canonical_skills:
            if canonical.lower() == skill_low:
                return self._build(canonical, "canonical", canonical_skills[canonical])

        # 2. synonym match
        for canonical, meta in canonical_skills.items():
            if skill_low in {s.lower() for s in meta.get("synonyms", [])}:
                return self._build(canonical, "synonym", meta)

        # 3. fuzzy: token overlap between the input and a canonical name/synonym
        #    (punctuation like '-' / '_' / '.' is treated as a word break here so
        #    "power-bi" matches "Power BI")
        def _fuzzy_tokens(s: str) -> set:
            return set(tokenize(s.replace("-", " ").replace("_", " ").replace(".", " ")))

        want = _fuzzy_tokens(skill_low)
        best: Optional[str] = None
        best_overlap = 0.0
        for canonical, meta in canonical_skills.items():
            names = [canonical] + list(meta.get("synonyms", []))
            for name in names:
                have = _fuzzy_tokens(name)
                if not have:
                    continue
                overlap = len(want & have) / len(want | have)
                if overlap > best_overlap:
                    best, best_overlap = canonical, overlap
        if best and best_overlap >= 0.34:
            return self._build(best, "fuzzy", canonical_skills[best])

        return SkillResolution(
            input_skill=skill, canonical=None, matched_via="unresolved",
            query_terms=[skill] if skill else [],
        )

    def _build(self, canonical: str, via: str, meta: dict) -> SkillResolution:
        synonyms = list(meta.get("synonyms", []))
        related = list(meta.get("related", []))
        query_terms = [canonical] + synonyms + related
        return SkillResolution(
            input_skill=canonical,
            canonical=canonical,
            matched_via=via,
            synonyms=synonyms,
            related_terms=related,
            query_terms=query_terms,
            # Only the canonical name counts toward the percentage. Related
            # canonical skills (e.g. ML -> Python) are deliberately excluded to
            # avoid inflating the count (semantic drift, design doc s.9).
            exact_skill_names={canonical},
        )
