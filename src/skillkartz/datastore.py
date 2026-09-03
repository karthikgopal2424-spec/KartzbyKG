"""Loads the approved corpora and provides retrieval + dedupe.

This is the shared "Tools & Data Infrastructure" layer from design doc section 7:
one index that every specialist bot draws on rather than owning a private copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from .config import SETTINGS
from .models import JobPosting, RetrievalResult
from .text import TfidfIndex


@dataclass
class Course:
    id: str
    skill: str
    title: str
    provider: str
    cost_usd: float
    duration_weeks: int
    format: str
    level: str
    rating: float
    active: bool


@dataclass
class LocalSignal:
    id: str
    skill: str
    sector: str
    location: str
    employer: str
    posting_date: str
    source: str
    source_reliability: float
    licensing_ok: bool
    description: str


class DataStore:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        retrieval_backend: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir or SETTINGS.data_dir)
        self.retrieval_backend = (retrieval_backend or SETTINGS.retrieval_backend).lower()
        self.postings: list[JobPosting] = []
        self.taxonomy: dict = {}
        self.courses: list[Course] = []
        self.local_signals: list[LocalSignal] = []
        self._index = self._make_index(self.retrieval_backend)
        self._as_of: str = date.today().isoformat()
        self._load()

    @staticmethod
    def _make_index(backend: str):
        """Both indexes expose the same build/query_vector/score surface, so
        ``retrieve`` is backend-agnostic."""
        if backend in ("tfidf", "", None):
            return TfidfIndex()
        if backend == "langchain":
            from .retrieval_langchain import LangChainSemanticIndex
            return LangChainSemanticIndex()
        raise ValueError(
            f"unknown retrieval_backend {backend!r} (use 'tfidf' or 'langchain')"
        )

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        jp = json.loads((self.data_dir / "job_postings.json").read_text())
        self._as_of = jp.get("as_of", self._as_of)
        self.postings = [JobPosting(**p) for p in jp["postings"]]

        self.taxonomy = json.loads((self.data_dir / "skill_taxonomy.json").read_text())

        cc = json.loads((self.data_dir / "course_catalog.json").read_text())
        self.courses = [Course(**c) for c in cc["courses"]]

        ls_path = self.data_dir / "local_signals.json"
        if ls_path.exists():
            ls = json.loads(ls_path.read_text())
            self.local_signals = [LocalSignal(**r) for r in ls["records"]]

        self._index.build(
            (p.id, f"{p.title}. {p.description}. skills: {', '.join(p.skills)}")
            for p in self.postings
        )

    @property
    def as_of(self) -> str:
        return self._as_of

    # ------------------------------------------------------------------ #
    # Skill taxonomy helpers (used by the Skill Intelligence Bot)
    # ------------------------------------------------------------------ #
    def canonical_skills(self) -> dict:
        return self.taxonomy.get("canonical_skills", {})

    def sectors(self) -> list[str]:
        return self.taxonomy.get("sectors", [])

    def locations(self) -> list[str]:
        return self.taxonomy.get("locations", [])

    # ------------------------------------------------------------------ #
    # Retrieval (Job Market Research Bot)
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query_terms: list[str],
        canonical_skill: str,
        sector: Optional[str],
        location: Optional[str],
        exact_skill_names: Optional[set[str]] = None,
        expansion_phrases: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """Semantic search + metadata filter, returning the FULL matching set.

        Design doc section 9: "retrieval-for-computation rather than
        retrieval-for-generation" - the Forecast Bot needs every matching
        posting to compute an accurate percentage, not a top-k slice.
        """
        exact_skill_names = exact_skill_names or {canonical_skill}
        exact_lower = {s.lower() for s in exact_skill_names}
        phrases = [p.lower() for p in (expansion_phrases or []) if len(p) >= 4]
        qvec = self._index.query_vector(query_terms)

        # 1. metadata filter for the sector universe (the denominator)
        raw_universe = [
            p for p in self.postings
            if (sector is None or p.sector == sector)
            and (location is None or p.location == location)
        ]
        universe = self._dedupe(raw_universe)
        dupes_removed = len(raw_universe) - len(universe)

        # 2. counting rule: taxonomy exact-match on the canonical skill against
        #    the posting's structured skill list. This keeps the percentage
        #    auditable against ground truth (design doc s.9 "retrieval-for-
        #    computation").
        matched: list[JobPosting] = []
        semantic_only = 0
        for p in universe:
            skills_lower = {s.lower() for s in p.skills}
            exact_hit = bool(skills_lower & exact_lower)
            if exact_hit:
                matched.append(p)
                continue
            # A semantic-only hit: the posting text names a genuine related
            # phrase ("pandas", "g-code", ...) but does not list the skill
            # itself. NOT counted by default - this is exactly the semantic-
            # drift surface the Governance Bot watches. Recorded so the risk
            # is visible and quantified.
            desc = p.description.lower()
            sem_score = self._index.score(p.id, qvec)
            if sem_score >= SETTINGS.retrieval_similarity_floor and any(
                phrase in desc for phrase in phrases
            ):
                semantic_only += 1

        dates = sorted(p.posting_date for p in universe)
        sources = sorted({p.source for p in universe})

        # "Insufficient data" (design doc s.8) means the corpus can't support a
        # standard forecast - NOT merely low demand. A large sector universe with
        # few skill matches is a real "low demand" signal and is forecast
        # normally. The niche path fires only when the universe itself is thin,
        # or a structurally offline-leaning skill has almost no online matches.
        meta = self.canonical_skills().get(canonical_skill, {})
        skill_is_niche = bool(meta.get("niche"))
        thin_universe = len(universe) < 2 * SETTINGS.small_sample_floor
        insufficient = thin_universe or (
            skill_is_niche and len(matched) <= SETTINGS.insufficient_data_postings
        )

        return RetrievalResult(
            query_terms=query_terms,
            canonical_skill=canonical_skill,
            sector=sector,
            location=location,
            matched=matched,
            sector_universe=universe,
            duplicates_removed=max(dupes_removed, 0),
            semantic_only_candidates=semantic_only,
            sources=sources,
            earliest_date=dates[0] if dates else None,
            latest_date=dates[-1] if dates else None,
            insufficient_data=insufficient,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dedupe(postings: list[JobPosting]) -> list[JobPosting]:
        seen: set[tuple] = set()
        out: list[JobPosting] = []
        for p in postings:
            key = (p.title.strip().lower(), p.employer.strip().lower(),
                   p.sector, p.location, p.description.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def sector_breakdown(
        self,
        exact_skill_names: set[str],
        location: Optional[str],
    ) -> list[tuple[str, int, int, list[str]]]:
        """Per-sector (sector, skill_postings, total_postings, dates) for ranking."""
        exact_lower = {s.lower() for s in exact_skill_names}
        rows: list[tuple[str, int, int, list[str]]] = []
        for sector in self.sectors():
            universe = self._dedupe([
                p for p in self.postings
                if p.sector == sector and (location is None or p.location == location)
            ])
            if not universe:
                continue
            hits = [p for p in universe
                    if {s.lower() for s in p.skills} & exact_lower]
            rows.append((sector, len(hits), len(universe),
                         sorted(p.posting_date for p in hits)))
        return rows

    # ------------------------------------------------------------------ #
    # Co-occurrence mining (feeds Tree-of-Thought candidate generation)
    # ------------------------------------------------------------------ #
    def cooccurring_skills(self, skill: str, sector: str, top_n: int = 8) -> list[str]:
        from collections import Counter

        counts: Counter = Counter()
        for p in self.postings:
            if p.sector != sector:
                continue
            if skill in p.skills:
                for other in p.skills:
                    if other != skill:
                        counts[other] += 1
        return [s for s, _ in counts.most_common(top_n)]

    def sector_skill_pct(self, skill: str, sector: str) -> float:
        universe = self._dedupe([p for p in self.postings if p.sector == sector])
        if not universe:
            return 0.0
        hits = sum(1 for p in universe if skill in p.skills)
        return round(100.0 * hits / len(universe), 2)

    # ------------------------------------------------------------------ #
    # Course catalog (Upskilling Bot)
    # ------------------------------------------------------------------ #
    def courses_for(self, skill: str, active_only: bool = True) -> list[Course]:
        return [
            c for c in self.courses
            if c.skill == skill and (c.active or not active_only)
        ]

    def course_by_id(self, course_id: str) -> Optional[Course]:
        for c in self.courses:
            if c.id == course_id:
                return c
        return None

    # ------------------------------------------------------------------ #
    # Local / offline signals (Local & Niche Skill Intelligence Agent)
    # ------------------------------------------------------------------ #
    def local_signals_for(self, skill: str, location: Optional[str]) -> list[LocalSignal]:
        return [
            r for r in self.local_signals
            if r.skill == skill and (location is None or r.location == location)
        ]
