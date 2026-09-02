"""Niche Skill Governance & Benchmarking Agent (conditional - design doc section 8).

Decides whether a low-data skill is genuinely niche or simply under-covered, and
sets the context-based benchmark that replaces the standard 35% threshold for
that skill. Every benchmark it emits carries its calculation period, evidence
sources, confidence, and review date (design doc section 8).

Guardrail (section 12): a first-time niche benchmark - one with no prior history -
is a high-impact decision that requires human sign-off before it becomes standing
policy. This agent flags that; the Supervisor escalates it.
"""

from __future__ import annotations

from datetime import date, timedelta
from statistics import median
from typing import Optional

from ..config import SETTINGS
from ..datastore import DataStore
from ..models import NicheAssessment
from .local_niche_intel import LocalIntelResult


class NicheSkillGovernanceBot:
    name = "Niche Skill Governance & Benchmarking Agent"

    def __init__(self, store: DataStore, benchmark_history: Optional[dict] = None) -> None:
        self.store = store
        # {skill: benchmark_pct} - empty here means every benchmark is first-time
        self.benchmark_history = benchmark_history or {}

    # ------------------------------------------------------------------ #
    def assess(
        self,
        skill: str,
        target_sector: str,
        location: Optional[str],
        standard_matched: int,
        local: LocalIntelResult,
    ) -> NicheAssessment:
        rationale: list[str] = []
        as_of = date.fromisoformat(self.store.as_of)

        # --- evidence tallies -----------------------------------------
        sectors_present = [
            s for s in self.store.sectors()
            if self.store.sector_skill_pct(skill, s) > 0
        ]
        locations_present = sorted({
            r.location for r in local.records
        } | {
            p.location for p in self.store.postings if skill in p.skills
        })
        employers_present = {
            r.employer for r in local.records
        } | {
            p.employer for p in self.store.postings if skill in p.skills
        }
        growth = self._growth(skill, local)

        rationale.append(
            f"standard job-board matches: {standard_matched}; "
            f"usable local records: {local.usable_records} "
            f"(+{local.weak_source_records} below reliability/licensing floor)"
        )
        rationale.append(
            f"present in {len(sectors_present)}/{len(self.store.sectors())} sectors, "
            f"{len(locations_present)} locations, ~{len(employers_present)} employers"
        )
        rationale.append(f"local posting growth signal: {growth}")

        # --- classification -----------------------------------------
        total_evidence = standard_matched + local.usable_records
        if total_evidence < SETTINGS.niche_min_local_records:
            classification = "Insufficient evidence"
            rationale.append(
                "total usable evidence below the minimum - cannot responsibly "
                "set a benchmark"
            )
        elif len(sectors_present) <= 1:
            classification = "Industry-specific niche skill"
        elif len(locations_present) <= 2:
            classification = "Location-specific niche skill"
        elif growth == "rising":
            classification = "Emerging skill"
        else:
            classification = "Mainstream skill"
        rationale.append(f"classification: {classification}")

        # --- benchmark ---------------------------------------------
        benchmark = self._benchmark(skill, target_sector, classification, rationale)

        first_time = skill not in self.benchmark_history
        if first_time and classification != "Insufficient evidence":
            rationale.append(
                "first-time benchmark for this skill - flagged for human sign-off "
                "before it becomes standing policy"
            )

        confidence = self._confidence(local, total_evidence)
        rationale.append(f"benchmark confidence: {confidence:.2f}")

        return NicheAssessment(
            skill=skill,
            classification=classification,
            benchmark_pct=round(benchmark, 1),
            rationale=rationale,
            local_records_used=local.usable_records,
            weak_source_records=local.weak_source_records,
            review_date=(as_of + timedelta(days=90)).isoformat(),
            evidence_sources=sorted(set(local.sources)),
        )

    # ------------------------------------------------------------------ #
    def _benchmark(
        self, skill: str, sector: str, classification: str, rationale: list[str]
    ) -> float:
        if classification == "Insufficient evidence":
            return SETTINGS.demand_threshold_pct

        meta = self.store.canonical_skills().get(skill, {})
        comparable = [
            s for s in meta.get("related", [])
            if s in self.store.canonical_skills()
        ]
        comp_pcts = [self.store.sector_skill_pct(s, sector) for s in comparable]
        comp_pcts = [p for p in comp_pcts if p > 0]

        if comp_pcts:
            base = median(comp_pcts) * 0.6
            rationale.append(
                f"context benchmark from comparable skills {comparable} in {sector}: "
                f"median {median(comp_pcts):.1f}% * 0.6 = {base:.1f}%"
            )
        else:
            base = 12.0
            rationale.append(
                "no comparable mainstream skills in this sector - falling back to a "
                "12% floor benchmark for niche demand"
            )
        return max(8.0, min(30.0, base))

    def _growth(self, skill: str, local: LocalIntelResult) -> str:
        as_of = date.fromisoformat(self.store.as_of)
        w = SETTINGS.trend_window_days
        recent = prior = 0
        for r in local.records:
            age = (as_of - date.fromisoformat(r.posting_date)).days
            if 0 <= age <= w:
                recent += 1
            elif w < age <= 2 * w:
                prior += 1
        if recent + prior < 4:
            return "unknown"
        if recent > prior:
            return "rising"
        if recent < prior:
            return "declining"
        return "stable"

    def _confidence(self, local: LocalIntelResult, total_evidence: int) -> float:
        volume = min(1.0, total_evidence / 12.0)
        reliability = (
            local.usable_records / len(local.records) if local.records else 0.0
        )
        freshness = local.fresh_ratio
        return round(0.45 * volume + 0.30 * reliability + 0.25 * freshness, 2)
