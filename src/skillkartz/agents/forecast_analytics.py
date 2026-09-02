"""Forecast and Analytics Bot - the deterministic calculation core.

Design doc section 4 / section 7 (chain-of-thought): "works the skill-availability
calculation step by step (skill postings in sector / total sector postings * 100)
rather than producing a number directly. This intermediate reasoning trace is
what keeps the percentage auditable."

No LLM touches this file - every number here is arithmetic on retrieved records.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from ..config import SETTINGS
from ..datastore import DataStore
from ..models import ForecastResult, RetrievalResult, SectorForecast


class ForecastAndAnalyticsBot:
    name = "Forecast and Analytics Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    # ------------------------------------------------------------------ #
    def forecast(
        self,
        retrieval: RetrievalResult,
        target_sector: str,
        exact_skill_names: set[str],
        location: Optional[str],
        freshness_ratio: float,
        benchmark_pct: Optional[float] = None,
        benchmark_source: str = "standard-threshold",
    ) -> ForecastResult:
        benchmark = SETTINGS.demand_threshold_pct if benchmark_pct is None else benchmark_pct
        steps: list[str] = []

        # --- per-sector availability % (chain-of-thought) ---------------
        breakdown = self.store.sector_breakdown(exact_skill_names, location)
        sector_forecasts: list[SectorForecast] = []
        for sector, skill_n, total_n, hit_dates in breakdown:
            pct = round(100.0 * skill_n / total_n, 2) if total_n else 0.0
            reasoning = [
                f"{sector}: {skill_n} postings mention "
                f"{sorted(exact_skill_names)} out of {total_n} total "
                f"=> {skill_n} / {total_n} * 100 = {pct:.2f}%",
            ]
            trend = self._trend(sector, exact_skill_names, location, reasoning)
            sector_forecasts.append(SectorForecast(
                sector=sector,
                skill_postings=skill_n,
                total_postings=total_n,
                availability_pct=pct,
                trend=trend,
                reasoning_steps=reasoning,
            ))

        sector_forecasts.sort(key=lambda s: -s.availability_pct)
        target = next((s for s in sector_forecasts if s.sector == target_sector), None)

        if target is None:
            steps.append(
                f"No postings found for target sector '{target_sector}' "
                f"(location filter: {location or 'any'})."
            )
            target = SectorForecast(target_sector, 0, 0, 0.0, "unknown",
                                    ["no postings in universe"])
            sector_forecasts.append(target)

        steps.append(
            f"Target sector '{target_sector}': {target.skill_postings} / "
            f"{target.total_postings} * 100 = {target.availability_pct:.2f}%"
        )
        steps.append(
            f"Benchmark for comparison: {benchmark:.2f}% ({benchmark_source})"
        )

        below = target.availability_pct < benchmark
        steps.append(
            f"{target.availability_pct:.2f}% {'<' if below else '>='} "
            f"{benchmark:.2f}% => "
            + ("below benchmark, roadmap recommended" if below
               else "at/above benchmark, no roadmap")
        )

        small_sample = (
            target.total_postings < SETTINGS.small_sample_floor
            or target.skill_postings < SETTINGS.small_numerator_floor
        )
        if small_sample:
            steps.append(
                f"Small-sample flag: target sector has "
                f"{target.skill_postings} skill-matching / "
                f"{target.total_postings} total analysed postings "
                f"(floors: numerator {SETTINGS.small_numerator_floor}, "
                f"total {SETTINGS.small_sample_floor})."
            )

        confidence, band, conf_steps = self._confidence(
            retrieval, target, freshness_ratio
        )
        steps.extend(conf_steps)

        return ForecastResult(
            canonical_skill=retrieval.canonical_skill,
            target_sector=target_sector,
            benchmark_pct=round(benchmark, 2),
            benchmark_source=benchmark_source,
            sector_forecasts=sector_forecasts,
            confidence=confidence,
            confidence_band=band,
            below_benchmark=below,
            small_sample=small_sample,
            reasoning_steps=steps,
        )

    # ------------------------------------------------------------------ #
    def _trend(
        self,
        sector: str,
        exact_skill_names: set[str],
        location: Optional[str],
        reasoning: list[str],
    ) -> str:
        as_of = date.fromisoformat(self.store.as_of)
        w = SETTINGS.trend_window_days
        exact_lower = {s.lower() for s in exact_skill_names}
        recent_hit = recent_tot = prior_hit = prior_tot = 0
        for p in self.store.postings:
            if p.sector != sector:
                continue
            if location is not None and p.location != location:
                continue
            try:
                age = (as_of - date.fromisoformat(p.posting_date)).days
            except ValueError:
                continue
            is_hit = bool({s.lower() for s in p.skills} & exact_lower)
            if 0 <= age <= w:
                recent_tot += 1
                recent_hit += is_hit
            elif w < age <= 2 * w:
                prior_tot += 1
                prior_hit += is_hit
        if recent_tot < 5 or prior_tot < 5:
            return "unknown"
        recent_pct = 100.0 * recent_hit / recent_tot
        prior_pct = 100.0 * prior_hit / prior_tot
        delta = recent_pct - prior_pct
        reasoning.append(
            f"  trend: last {w}d {recent_pct:.1f}% vs prior {w}d {prior_pct:.1f}% "
            f"(delta {delta:+.1f}pp)"
        )
        if delta >= SETTINGS.trend_delta_pp:
            return "rising"
        if delta <= -SETTINGS.trend_delta_pp:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------ #
    def _confidence(
        self,
        retrieval: RetrievalResult,
        target: SectorForecast,
        freshness_ratio: float,
    ) -> tuple[float, str, list[str]]:
        s = SETTINGS
        sample_factor = min(1.0, target.total_postings / s.conf_sample_saturation)
        coverage_factor = min(1.0, len(retrieval.sources) / s.conf_sources_saturation)
        freshness_factor = max(0.0, min(1.0, freshness_ratio))

        score = (
            s.conf_weight_sample * sample_factor
            + s.conf_weight_coverage * coverage_factor
            + s.conf_weight_freshness * freshness_factor
        )
        # penalty for a large un-counted semantic-drift surface
        denom = max(target.skill_postings, 1)
        drift_ratio = retrieval.semantic_only_candidates / denom
        drift_penalty = min(0.15, 0.15 * drift_ratio)
        score = round(max(0.0, min(1.0, score - drift_penalty)), 2)

        band = "high" if score >= 0.70 else "medium" if score >= s.confidence_floor else "low"
        steps = [
            f"confidence: sample={sample_factor:.2f}*{s.conf_weight_sample} + "
            f"coverage={coverage_factor:.2f}*{s.conf_weight_coverage} + "
            f"freshness={freshness_factor:.2f}*{s.conf_weight_freshness} "
            f"- drift_penalty={drift_penalty:.2f} => {score:.2f} ({band})",
        ]
        return score, band, steps
