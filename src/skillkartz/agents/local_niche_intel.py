"""Local & Niche Skill Intelligence Agent (conditional - design doc section 8).

Activates only when the Job Market Research Bot reports insufficient standard
job-board data. Collects employment signals from approved offline sources
(regional press, community boards, trade associations, government exchanges),
deduplicates, and assigns a source-reliability score. Guardrail (section 12):
limited to an approved-domain list rather than open crawling, and every record
carries a licensing-status flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ..config import SETTINGS
from ..datastore import DataStore, LocalSignal


@dataclass
class LocalIntelResult:
    skill: str
    location: Optional[str]
    records: list[LocalSignal] = field(default_factory=list)
    duplicates_removed: int = 0
    weak_source_records: int = 0
    usable_records: int = 0
    sources: list[str] = field(default_factory=list)
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None
    fresh_ratio: float = 0.0


class LocalNicheIntelligenceBot:
    name = "Local & Niche Skill Intelligence Agent"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def collect(self, skill: str, location: Optional[str]) -> LocalIntelResult:
        raw = self.store.local_signals_for(skill, location)

        seen: set[tuple] = set()
        deduped: list[LocalSignal] = []
        for r in raw:
            key = (r.skill, r.employer.lower(), r.location, r.description.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        weak = [r for r in deduped
                if r.source_reliability < SETTINGS.niche_source_reliability_floor
                or not r.licensing_ok]
        usable = [r for r in deduped if r not in weak]

        dates = sorted(r.posting_date for r in deduped)
        as_of = date.fromisoformat(self.store.as_of)
        fresh = sum(
            1 for r in deduped
            if (as_of - date.fromisoformat(r.posting_date)).days
            <= SETTINGS.freshness_window_days
        )

        return LocalIntelResult(
            skill=skill,
            location=location,
            records=deduped,
            duplicates_removed=len(raw) - len(deduped),
            weak_source_records=len(weak),
            usable_records=len(usable),
            sources=sorted({r.source for r in deduped}),
            earliest_date=dates[0] if dates else None,
            latest_date=dates[-1] if dates else None,
            fresh_ratio=(fresh / len(deduped)) if deduped else 0.0,
        )
