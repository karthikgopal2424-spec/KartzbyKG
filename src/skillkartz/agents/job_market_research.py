"""Job Market Research Bot - retrieval, filtering, dedupe, freshness.

Design doc section 4: "retrieves relevant postings and labor data, filters
records, removes duplicates, and records source dates". Least-privilege
(section 12): read-only over the postings index.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from ..config import SETTINGS
from ..datastore import DataStore
from ..models import RetrievalResult


class JobMarketResearchBot:
    name = "Job Market Research Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def research(
        self,
        query_terms: list[str],
        canonical_skill: str,
        sector: Optional[str],
        location: Optional[str],
        exact_skill_names: Optional[set] = None,
        expansion_phrases: Optional[list] = None,
    ) -> RetrievalResult:
        return self.store.retrieve(
            query_terms=query_terms,
            canonical_skill=canonical_skill,
            sector=sector,
            location=location,
            exact_skill_names=exact_skill_names,
            expansion_phrases=expansion_phrases,
        )

    # freshness: share of the sector universe posted within the freshness window
    def freshness_ratio(self, retrieval: RetrievalResult) -> float:
        universe = retrieval.sector_universe
        if not universe:
            return 0.0
        as_of = date.fromisoformat(self.store.as_of)
        fresh = 0
        for p in universe:
            try:
                age = (as_of - date.fromisoformat(p.posting_date)).days
            except ValueError:
                continue
            if age <= SETTINGS.freshness_window_days:
                fresh += 1
        return fresh / len(universe)
