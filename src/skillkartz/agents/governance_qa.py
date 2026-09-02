"""Governance & QA Bot - the reflection / self-critique pass (design doc section 7).

Runs before any number reaches the user. Re-verifies arithmetic, checks that
every factual claim traces to a record (groundedness), enforces the output
contract, and watches for anomalous drift. When something looks off it either
requests one targeted recalculation or escalates to a human - it never "fixes"
a shaky result silently (design doc section 12).
"""

from __future__ import annotations

from typing import Optional

from ..config import NON_GUARANTEE_DISCLAIMER, SETTINGS
from ..datastore import DataStore
from ..models import (
    ForecastResult,
    GovernanceIssue,
    GovernanceReport,
    NicheAssessment,
    RetrievalResult,
    Roadmap,
)


class GovernanceAndQABot:
    name = "Governance & QA Bot"

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def review(
        self,
        retrieval: RetrievalResult,
        forecast: ForecastResult,
        roadmap: Optional[Roadmap],
        rendered_response: str,
        niche: Optional[NicheAssessment] = None,
        previous_target_pct: Optional[float] = None,
        is_recalculation: bool = False,
    ) -> GovernanceReport:
        issues: list[GovernanceIssue] = []
        recalc = False
        escalate = False
        escalation_reason: Optional[str] = None

        # 1. arithmetic re-verification -------------------------------
        for sf in forecast.sector_forecasts:
            expected = round(100.0 * sf.skill_postings / sf.total_postings, 2) if sf.total_postings else 0.0
            if abs(expected - sf.availability_pct) > 0.01:
                issues.append(GovernanceIssue(
                    "blocker", "arithmetic",
                    f"{sf.sector}: stored {sf.availability_pct}% != recomputed {expected}%",
                ))
                escalate = True
                escalation_reason = "arithmetic mismatch in forecast"
        target = forecast.target()
        if target is not None:
            recomputed_below = target.availability_pct < forecast.benchmark_pct
            if recomputed_below != forecast.below_benchmark:
                issues.append(GovernanceIssue(
                    "blocker", "threshold-logic",
                    "below_benchmark flag disagrees with recomputed comparison",
                ))
                escalate = True
                escalation_reason = "threshold logic inconsistent"

        # 2. groundedness -------------------------------------------
        if roadmap and roadmap.feasible:
            for step in roadmap.steps:
                course = self.store.course_by_id(step.course_id)
                if course is None:
                    issues.append(GovernanceIssue(
                        "blocker", "groundedness",
                        f"roadmap cites course {step.course_id} which is not in the catalog",
                    ))
                    escalate = True
                    escalation_reason = "ungrounded course recommendation"
                elif not course.active:
                    issues.append(GovernanceIssue(
                        "blocker", "groundedness",
                        f"roadmap cites discontinued course {step.course_id}",
                    ))
                    escalate = True
                    escalation_reason = "discontinued course recommendation"
                elif abs(course.cost_usd - step.cost_usd) > 0.01:
                    issues.append(GovernanceIssue(
                        "warning", "groundedness",
                        f"course {step.course_id} cost {step.cost_usd} != catalog {course.cost_usd}",
                    ))

        if forecast.benchmark_source not in {"standard-threshold", "niche-benchmark"}:
            issues.append(GovernanceIssue(
                "warning", "groundedness",
                f"unknown benchmark source '{forecast.benchmark_source}'",
            ))

        # 3. output contract --------------------------------------
        contract_bits = {
            "non-guarantee disclaimer": NON_GUARANTEE_DISCLAIMER[:40] in rendered_response,
            "data coverage line": "Data coverage:" in rendered_response,
            "confidence": "Confidence" in rendered_response or "confidence" in rendered_response,
            "percentage": "%" in rendered_response,
        }
        for label, ok in contract_bits.items():
            if not ok:
                issues.append(GovernanceIssue(
                    "blocker", "output-contract", f"response is missing the {label}",
                ))
                escalate = True
                escalation_reason = "output contract violation"

        if forecast.small_sample and "CAVEAT" not in rendered_response:
            issues.append(GovernanceIssue(
                "blocker", "output-contract",
                "small-sample forecast delivered without the required caveat",
            ))
            escalate = True
            escalation_reason = "missing small-sample caveat"

        # 4. anomaly / drift -------------------------------------
        if previous_target_pct is not None and target is not None:
            delta = abs(target.availability_pct - previous_target_pct)
            if delta > SETTINGS.governance_drift_pp:
                if is_recalculation:
                    issues.append(GovernanceIssue(
                        "warning", "drift",
                        f"target % moved {delta:.1f}pp vs prior run even after recalculation",
                    ))
                else:
                    recalc = True
                    issues.append(GovernanceIssue(
                        "info", "drift",
                        f"target % moved {delta:.1f}pp vs prior run - requesting recalculation",
                    ))

        # 5. semantic-drift surface ----------------------------
        denom = max(len(retrieval.matched), 1)
        if retrieval.semantic_only_candidates > 0.5 * denom:
            issues.append(GovernanceIssue(
                "warning", "semantic-drift",
                f"{retrieval.semantic_only_candidates} related-but-unlisted postings "
                f"were excluded from the count of {len(retrieval.matched)} - "
                f"phrasing coverage may be incomplete",
            ))

        # 6. niche evidence -----------------------------------
        if niche is not None:
            if niche.classification == "Insufficient evidence":
                escalate = True
                escalation_reason = "niche skill classified 'insufficient evidence'"
                issues.append(GovernanceIssue(
                    "blocker", "niche-evidence",
                    "not enough usable evidence to publish a benchmark",
                ))
            if niche.weak_source_records > 0:
                issues.append(GovernanceIssue(
                    "warning", "licensing",
                    f"{niche.weak_source_records} local records were held back "
                    f"(reliability/licensing below floor)",
                ))
            if niche.local_records_used < SETTINGS.niche_min_local_records:
                escalate = True
                escalation_reason = escalation_reason or "thin niche evidence base"
                issues.append(GovernanceIssue(
                    "blocker", "niche-evidence",
                    f"only {niche.local_records_used} usable local records "
                    f"(need {SETTINGS.niche_min_local_records})",
                ))

        # 7. confidence floor --------------------------------
        if forecast.confidence < SETTINGS.confidence_floor:
            if is_recalculation:
                escalate = True
                escalation_reason = escalation_reason or (
                    f"confidence {forecast.confidence:.2f} below floor "
                    f"{SETTINGS.confidence_floor} after recalculation"
                )
                issues.append(GovernanceIssue(
                    "blocker", "confidence-floor",
                    f"confidence {forecast.confidence:.2f} < floor "
                    f"{SETTINGS.confidence_floor} after recalculation",
                ))
            else:
                recalc = True
                issues.append(GovernanceIssue(
                    "info", "confidence-floor",
                    f"confidence {forecast.confidence:.2f} < floor "
                    f"{SETTINGS.confidence_floor} - requesting recalculation",
                ))

        if escalate:
            recalc = False
        passed = not issues or all(i.severity != "blocker" for i in issues)
        return GovernanceReport(
            passed=passed and not escalate,
            issues=issues,
            recalculation_requested=recalc,
            escalate=escalate,
            escalation_reason=escalation_reason,
        )
