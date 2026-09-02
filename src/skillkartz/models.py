"""Typed data structures passed between agents.

These are plain dataclasses so a full pipeline run is a serialisable object the
Governance Bot (and a human reviewer) can audit after the fact - design doc
section 12 "Runtime monitoring ... written to the audit log for every response".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
@dataclass
class UserQuery:
    raw_text: str
    skill: Optional[str] = None
    location: Optional[str] = None
    sector: Optional[str] = None
    experience: Optional[str] = None
    budget_usd: Optional[float] = None
    timeframe_weeks: Optional[int] = None
    preferred_format: Optional[str] = None
    goal: Optional[str] = None

    def missing_required(self) -> list[str]:
        missing = []
        if not self.skill:
            missing.append("skill")
        if not self.sector:
            missing.append("sector")
        return missing


# --------------------------------------------------------------------------- #
# Query rewrite (input normalisation, before intent capture)
# --------------------------------------------------------------------------- #
@dataclass
class RewriteCorrection:
    span: str          # the original token/phrase as the user typed it
    replacement: str   # the known term it was snapped to
    category: str      # "skill" | "sector" | "location"
    similarity: float  # difflib ratio, 0..1


@dataclass
class QueryRewrite:
    original_text: str
    normalized_text: str
    corrections: list["RewriteCorrection"] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.normalized_text != self.original_text


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class JobPosting:
    id: str
    title: str
    employer: str
    sector: str
    location: str
    experience_level: str
    posting_date: str
    source: str
    skills: list[str]
    description: str


@dataclass
class RetrievalResult:
    query_terms: list[str]
    canonical_skill: str
    sector: Optional[str]
    location: Optional[str]
    matched: list[JobPosting]
    sector_universe: list[JobPosting]
    duplicates_removed: int
    semantic_only_candidates: int
    sources: list[str]
    earliest_date: Optional[str]
    latest_date: Optional[str]
    insufficient_data: bool

    @property
    def coverage_note(self) -> str:
        span = f"{self.earliest_date} to {self.latest_date}" if self.earliest_date else "n/a"
        return (
            f"{len(self.sector_universe)} postings analysed for sector "
            f"'{self.sector}', {len(self.sources)} distinct sources, "
            f"posting dates {span}, {self.duplicates_removed} duplicates removed"
        )


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
@dataclass
class SectorForecast:
    sector: str
    skill_postings: int
    total_postings: int
    availability_pct: float
    trend: str  # "rising" | "stable" | "declining" | "unknown"
    reasoning_steps: list[str] = field(default_factory=list)


@dataclass
class ForecastResult:
    canonical_skill: str
    target_sector: str
    benchmark_pct: float
    benchmark_source: str  # "standard-threshold" | "niche-benchmark"
    sector_forecasts: list[SectorForecast]
    confidence: float
    confidence_band: str
    below_benchmark: bool
    small_sample: bool
    reasoning_steps: list[str] = field(default_factory=list)

    def target(self) -> Optional[SectorForecast]:
        for sf in self.sector_forecasts:
            if sf.sector == self.target_sector:
                return sf
        return None


# --------------------------------------------------------------------------- #
# Niche extension
# --------------------------------------------------------------------------- #
@dataclass
class NicheAssessment:
    skill: str
    classification: str  # Mainstream | Emerging | Location-specific niche | Industry-specific niche | Insufficient evidence
    benchmark_pct: float
    rationale: list[str]
    local_records_used: int
    weak_source_records: int
    review_date: str
    evidence_sources: list[str]


# --------------------------------------------------------------------------- #
# Roadmap (Tree-of-Thought output)
# --------------------------------------------------------------------------- #
@dataclass
class RoadmapStep:
    stage: int
    skill: str
    course_id: str
    course_title: str
    provider: str
    cost_usd: float
    duration_weeks: int
    format: str
    projected_uplift_pp: float
    score: float


@dataclass
class Roadmap:
    feasible: bool
    steps: list[RoadmapStep]
    total_cost_usd: float
    total_weeks: int
    projected_final_pct: float
    demand_gap_pp: float
    notes: list[str] = field(default_factory=list)
    search_stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #
@dataclass
class GovernanceIssue:
    severity: str  # "info" | "warning" | "blocker"
    check: str
    detail: str


@dataclass
class GovernanceReport:
    passed: bool
    issues: list[GovernanceIssue]
    recalculation_requested: bool
    escalate: bool
    escalation_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class AgentTrace:
    agent: str
    summary: str
    detail: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: UserQuery
    status: str  # "answered" | "clarification_needed" | "escalated"
    clarification_questions: list[str] = field(default_factory=list)
    rewrite: Optional[QueryRewrite] = None
    retrieval: Optional[RetrievalResult] = None
    niche: Optional[NicheAssessment] = None
    forecast: Optional[ForecastResult] = None
    roadmap: Optional[Roadmap] = None
    governance: Optional[GovernanceReport] = None
    response_text: str = ""
    trace: list[AgentTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        def convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(self)
