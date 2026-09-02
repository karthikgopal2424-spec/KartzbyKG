"""Career Advisor Bot - intent capture and final response delivery.

Design doc: "collects the user's skill, location, experience, sector, timeframe,
budget, and learning preferences" and "presents the final explainable response".
Two guardrails from section 12 live here: input validation (a clarification turn
rather than a best-guess pass-through) and the template-enforced output contract
(disclaimer + source period + coverage + confidence, always together).
"""

from __future__ import annotations

import re
from typing import Optional

from ..config import NON_GUARANTEE_DISCLAIMER
from ..datastore import DataStore
from ..models import (
    ForecastResult,
    GovernanceReport,
    NicheAssessment,
    RetrievalResult,
    Roadmap,
    UserQuery,
)

_EXPERIENCE_WORDS = {
    "entry": "Entry", "junior": "Entry", "graduate": "Entry", "fresher": "Entry",
    "mid": "Mid", "intermediate": "Mid",
    "senior": "Senior", "lead": "Senior", "principal": "Senior",
}
_FORMAT_WORDS = {
    "online": "online", "in-person": "in-person", "in person": "in-person",
    "classroom": "in-person", "hybrid": "hybrid", "blended": "hybrid",
}


class CareerAdvisorBot:
    name = "Career Advisor Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    # ------------------------------------------------------------------ #
    # Intent capture
    # ------------------------------------------------------------------ #
    def capture(self, raw_text: str, **overrides) -> UserQuery:
        q = UserQuery(raw_text=raw_text)
        low = raw_text.lower()

        q.skill = self._match_skill(low)
        q.sector = self._match_one(low, self.store.sectors())
        q.location = self._match_one(low, self.store.locations())

        for word, canon in _EXPERIENCE_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                q.experience = canon
                break
        for word, canon in _FORMAT_WORDS.items():
            if word in low:
                q.preferred_format = canon
                break

        q.budget_usd = self._match_budget(low)
        q.timeframe_weeks = self._match_timeframe(low)

        for key, value in overrides.items():
            if value is not None and hasattr(q, key):
                setattr(q, key, value)
        return q

    def _match_skill(self, low: str) -> Optional[str]:
        best: Optional[str] = None
        best_len = 0
        for canonical, meta in self.store.canonical_skills().items():
            names = [canonical] + list(meta.get("synonyms", []))
            for name in names:
                if re.search(rf"\b{re.escape(name.lower())}\b", low) and len(name) > best_len:
                    best, best_len = canonical, len(name)
        return best

    @staticmethod
    def _match_one(low: str, options: list[str]) -> Optional[str]:
        for opt in options:
            if re.search(rf"\b{re.escape(opt.lower())}\b", low):
                return opt
        return None

    @staticmethod
    def _match_budget(low: str) -> Optional[float]:
        m = re.search(r"\$\s?([0-9][0-9,]*)", low) or re.search(
            r"(?:budget|under|below|up to|max)\D{0,6}([0-9][0-9,]*)", low
        )
        if m:
            return float(m.group(1).replace(",", ""))
        return None

    @staticmethod
    def _match_timeframe(low: str) -> Optional[int]:
        m = re.search(r"([0-9]+)\s*(week|month|year)s?", low)
        if not m:
            return None
        n = int(m.group(1))
        unit = m.group(2)
        return {"week": n, "month": n * 4, "year": n * 52}[unit]

    # ------------------------------------------------------------------ #
    # Validation (guardrail: clarification instead of best guess)
    # ------------------------------------------------------------------ #
    def clarification_questions(self, q: UserQuery) -> list[str]:
        questions: list[str] = []
        if not q.skill:
            questions.append(
                "Which skill do you want a demand read on? "
                f"(known skills include: {', '.join(list(self.store.canonical_skills())[:8])}...)"
            )
        if not q.sector:
            questions.append(
                "Which sector should I analyse? "
                f"Options: {', '.join(self.store.sectors())}."
            )
        return questions

    # ------------------------------------------------------------------ #
    # Response delivery (template-enforced output contract)
    # ------------------------------------------------------------------ #
    def render(
        self,
        q: UserQuery,
        retrieval: RetrievalResult,
        forecast: ForecastResult,
        roadmap: Optional[Roadmap],
        governance: GovernanceReport,
        niche: Optional[NicheAssessment] = None,
    ) -> str:
        target = forecast.target()
        lines: list[str] = []
        lines.append(
            f"Skill demand read: '{forecast.canonical_skill}' in the "
            f"{forecast.target_sector} sector"
            + (f" ({q.location})" if q.location else "")
        )
        lines.append("=" * 68)

        if target is not None:
            lines.append(
                f"  Observed job-posting share : {target.availability_pct:.1f}%  "
                f"({target.skill_postings} of {target.total_postings} postings)"
            )
        lines.append(
            f"  Benchmark                  : {forecast.benchmark_pct:.1f}%  "
            f"({forecast.benchmark_source})"
        )
        lines.append(
            f"  Demand trend               : {target.trend if target else 'unknown'}"
        )
        lines.append(
            f"  Confidence                 : {forecast.confidence:.2f} "
            f"({forecast.confidence_band})"
        )
        lines.append("")

        if niche is not None:
            lines.append(
                f"  Niche classification       : {niche.classification} "
                f"(benchmark {niche.benchmark_pct:.1f}%, {niche.local_records_used} "
                f"local records, review by {niche.review_date})"
            )
            lines.append("")

        # other sectors, for context
        others = [sf for sf in forecast.sector_forecasts if sf.sector != forecast.target_sector]
        if others:
            lines.append("  For comparison, other sectors:")
            for sf in sorted(others, key=lambda s: -s.availability_pct)[:5]:
                lines.append(
                    f"    - {sf.sector:<14} {sf.availability_pct:5.1f}%  "
                    f"({sf.skill_postings}/{sf.total_postings})"
                )
            lines.append("")

        if forecast.below_benchmark:
            lines.append(
                f"  VERDICT: below the {forecast.benchmark_pct:.0f}% benchmark - "
                "an upskilling roadmap is recommended."
            )
            if roadmap and roadmap.feasible:
                lines.append("")
                lines.append(
                    f"  Staged roadmap (projected share after completion "
                    f"~{roadmap.projected_final_pct:.1f}%, "
                    f"total ${roadmap.total_cost_usd:.0f} / {roadmap.total_weeks} weeks):"
                )
                for step in roadmap.steps:
                    lines.append(
                        f"    Stage {step.stage}: {step.skill} via "
                        f"\"{step.course_title}\" - {step.provider}, "
                        f"${step.cost_usd:.0f}, {step.duration_weeks}w, {step.format} "
                        f"(+{step.projected_uplift_pp:.1f}pp)"
                    )
            elif roadmap is not None:
                lines.append("")
                lines.append("  No roadmap fits the stated constraints:")
                for note in roadmap.notes:
                    lines.append(f"    - {note}")
        else:
            lines.append(
                f"  VERDICT: at or above the {forecast.benchmark_pct:.0f}% benchmark - "
                "no upskilling roadmap needed for this sector."
            )

        if forecast.small_sample:
            lines.append("")
            lines.append(
                "  CAVEAT: small sample - the target sector has very few analysed "
                "postings, so this percentage is indicative only."
            )

        gov_warnings = [i for i in governance.issues if i.severity != "info"]
        if gov_warnings:
            lines.append("")
            lines.append("  Governance notes:")
            for issue in gov_warnings:
                lines.append(f"    [{issue.severity}] {issue.check}: {issue.detail}")

        lines.append("")
        lines.append("  " + "-" * 66)
        lines.append(f"  Data coverage: {retrieval.coverage_note}.")
        lines.append(f"  Disclaimer: {NON_GUARANTEE_DISCLAIMER}")
        return "\n".join(lines)

    def render_escalation(self, q: UserQuery, reason: str) -> str:
        return (
            f"Handing off to a human reviewer.\n"
            f"{'=' * 68}\n"
            f"  Query   : skill='{q.skill}', sector='{q.sector}', "
            f"location='{q.location or 'any'}'\n"
            f"  Reason  : {reason}\n"
            f"  Next    : a human-assisted intake will follow up; ForecastAI will "
            f"not present an automated number for this query.\n"
            f"  Disclaimer: {NON_GUARANTEE_DISCLAIMER}"
        )

    def render_clarification(self, questions: list[str]) -> str:
        body = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions))
        return f"I need a bit more detail before I can answer:\n{body}"
