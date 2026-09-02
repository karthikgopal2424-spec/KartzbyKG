"""Supervisor / Orchestrator Bot - the hierarchical backbone (design doc section 11).

Star topology: the Supervisor holds the plan and the session state, dispatches to
specialists, runs the parallel fan-out (Skill Intelligence + Job Market Research),
takes the conditional branch into the niche pair when data is thin, runs the
Governance reflection loop, and applies at most one targeted recalculation before
escalating. It performs no heavy computation itself.
"""

from __future__ import annotations

from typing import Optional

from ..datastore import DataStore
from ..llm import LLMClient
from ..models import (
    AgentTrace,
    GovernanceReport,
    PipelineResult,
    QueryRewrite,
    UserQuery,
)
from .career_advisor import CareerAdvisorBot
from .forecast_analytics import ForecastAndAnalyticsBot
from .governance_qa import GovernanceAndQABot
from .job_market_research import JobMarketResearchBot
from .local_niche_intel import LocalNicheIntelligenceBot
from .niche_governance import NicheSkillGovernanceBot
from .query_rewriter import QueryRewriterBot
from .skill_intelligence import SkillIntelligenceBot
from .upskilling import UpskillingBot


class SupervisorBot:
    name = "Supervisor / Orchestrator Bot"

    def __init__(self, store: DataStore, llm: Optional[LLMClient] = None) -> None:
        self.store = store
        self.llm = llm or LLMClient()
        self.rewriter = QueryRewriterBot(store)
        self.advisor = CareerAdvisorBot(store)
        self.skill_intel = SkillIntelligenceBot(store)
        self.research = JobMarketResearchBot(store)
        self.forecaster = ForecastAndAnalyticsBot(store)
        self.governance = GovernanceAndQABot(store)
        self.local_intel = LocalNicheIntelligenceBot(store)
        self.niche_gov = NicheSkillGovernanceBot(store)
        # --- session memory (design doc section 5 "audit logs ... accepted and
        #     rejected recommendations can improve future ranking rules") ------
        self.memory: dict = {
            "results": {},          # query_key -> last target %
            "accepted": [],
            "rejected": [],
            "provider_bias": {},    # provider -> score nudge for the Upskilling Bot
        }

    # ------------------------------------------------------------------ #
    def record_feedback(self, query: UserQuery, kind: str, provider: Optional[str] = None) -> None:
        entry = {"skill": query.skill, "sector": query.sector, "provider": provider}
        if kind == "accepted":
            self.memory["accepted"].append(entry)
            if provider:
                self.memory["provider_bias"][provider] = min(
                    0.1, self.memory["provider_bias"].get(provider, 0.0) + 0.05)
        elif kind == "rejected":
            self.memory["rejected"].append(entry)
            if provider:
                self.memory["provider_bias"][provider] = max(
                    -0.1, self.memory["provider_bias"].get(provider, 0.0) - 0.05)

    @staticmethod
    def _query_key(q: UserQuery) -> str:
        return f"{(q.skill or '').lower()}|{(q.sector or '').lower()}|{(q.location or '').lower()}"

    # ------------------------------------------------------------------ #
    def run(self, query: UserQuery, rewrite: Optional[QueryRewrite] = None) -> PipelineResult:
        result = PipelineResult(query=query, status="answered")
        result.rewrite = rewrite
        trace = result.trace

        # --- plan ----------------------------------------------------
        trace.append(AgentTrace(
            self.name, "planned run",
            {"steps": ["rewrite", "skill-intel", "job-research", "forecast",
                       "threshold", "upskilling?", "governance", "deliver"],
             "llm": self.llm.status}))

        # --- input normalisation (Query Rewriter Bot) --------------
        if rewrite is not None:
            trace.append(AgentTrace(
                self.rewriter.name,
                ("normalised input: " + "; ".join(rewrite.notes))
                if rewrite.changed else "no changes - input already clean",
                {"original": rewrite.original_text,
                 "normalized": rewrite.normalized_text,
                 "corrections": [c.__dict__ for c in rewrite.corrections]}))

        # --- intent validation (Career Advisor guardrail) ----------
        questions = self.advisor.clarification_questions(query)
        if questions:
            result.status = "clarification_needed"
            result.clarification_questions = questions
            result.response_text = self.advisor.render_clarification(questions)
            trace.append(AgentTrace(self.advisor.name, "requested clarification",
                                    {"questions": questions}))
            return result

        # --- parallel fan-out: skill intel + job market research ----
        resolution = self.skill_intel.resolve(query.skill)
        trace.append(AgentTrace(self.skill_intel.name,
                                f"resolved '{query.skill}' via {resolution.matched_via}",
                                {"canonical": resolution.canonical,
                                 "query_terms": resolution.query_terms}))
        if not resolution.resolved:
            result.status = "clarification_needed"
            q = (f"I couldn't map '{query.skill}' to a known skill. "
                 f"Did you mean one of: {', '.join(list(self.store.canonical_skills())[:10])}?")
            result.clarification_questions = [q]
            result.response_text = self.advisor.render_clarification([q])
            return result

        canonical = resolution.canonical
        retrieval = self.research.research(
            resolution.query_terms, canonical, query.sector, query.location,
            resolution.exact_skill_names, resolution.related_terms)
        freshness = self.research.freshness_ratio(retrieval)
        result.retrieval = retrieval
        trace.append(AgentTrace(self.research.name, retrieval.coverage_note,
                                {"matched": len(retrieval.matched),
                                 "insufficient_data": retrieval.insufficient_data,
                                 "semantic_only_excluded": retrieval.semantic_only_candidates}))

        # --- conditional branch: niche extension pair -------------
        niche = None
        benchmark_pct = None
        benchmark_source = "standard-threshold"
        if retrieval.insufficient_data:
            trace.append(AgentTrace(self.name, "insufficient standard data - "
                                    "activating niche extension pair"))
            local = self.local_intel.collect(canonical, query.location)
            trace.append(AgentTrace(self.local_intel.name,
                                    f"{local.usable_records} usable local records "
                                    f"({local.weak_source_records} held back)",
                                    {"sources": local.sources,
                                     "duplicates_removed": local.duplicates_removed}))
            niche = self.niche_gov.assess(
                canonical, query.sector, query.location,
                len(retrieval.matched), local)
            benchmark_pct = niche.benchmark_pct
            benchmark_source = "niche-benchmark"
            result.niche = niche
            trace.append(AgentTrace(self.niche_gov.name,
                                    f"{niche.classification}, benchmark {niche.benchmark_pct}%",
                                    {"rationale": niche.rationale}))

        # --- forecast (+ one Governance-triggered recalculation) --
        prev_pct = self.memory["results"].get(self._query_key(query))
        forecast, governance, roadmap, draft = self._forecast_and_review(
            query, retrieval, canonical, resolution.exact_skill_names, freshness,
            benchmark_pct, benchmark_source, niche, prev_pct, is_recalc=False)
        trace.append(AgentTrace(self.forecaster.name,
                                f"target {forecast.target_sector} = "
                                f"{forecast.target().availability_pct:.2f}% "
                                f"(benchmark {forecast.benchmark_pct}%, "
                                f"confidence {forecast.confidence})",
                                {"reasoning": forecast.reasoning_steps}))

        if governance.recalculation_requested:
            trace.append(AgentTrace(self.governance.name,
                                    "anomaly/low-confidence - requesting recalculation",
                                    {"issues": [i.__dict__ for i in governance.issues]}))
            forecast, governance, roadmap, draft = self._forecast_and_review(
                query, retrieval, canonical, resolution.exact_skill_names, freshness,
                benchmark_pct, benchmark_source, niche, prev_pct, is_recalc=True)

        result.forecast = forecast
        result.roadmap = roadmap
        result.governance = governance
        trace.append(AgentTrace(self.governance.name,
                                "passed" if governance.passed else
                                ("escalate" if governance.escalate else "passed with warnings"),
                                {"issues": [i.__dict__ for i in governance.issues],
                                 "escalate": governance.escalate}))

        # --- deliver ---------------------------------------------
        self.memory["results"][self._query_key(query)] = forecast.target().availability_pct
        if governance.escalate:
            result.status = "escalated"
            result.response_text = self.advisor.render_escalation(
                query, governance.escalation_reason or "governance escalation")
            trace.append(AgentTrace(self.advisor.name, "delivered escalation notice"))
            return result

        result.response_text = self._maybe_narrate(draft, result)
        trace.append(AgentTrace(self.advisor.name, "delivered explainable response"))
        return result

    # ------------------------------------------------------------------ #
    def _forecast_and_review(
        self, query, retrieval, canonical, exact_names, freshness,
        benchmark_pct, benchmark_source, niche, prev_pct, is_recalc,
    ):
        forecast = self.forecaster.forecast(
            retrieval, query.sector, exact_names, query.location, freshness,
            benchmark_pct, benchmark_source)

        roadmap = None
        if forecast.below_benchmark and forecast.target().total_postings > 0:
            bot = UpskillingBot(self.store, self.memory["provider_bias"])
            roadmap = bot.build_roadmap(
                target_skill=canonical,
                target_sector=query.sector,
                current_pct=forecast.target().availability_pct,
                benchmark_pct=forecast.benchmark_pct,
                budget_usd=query.budget_usd,
                timeframe_weeks=query.timeframe_weeks,
                preferred_format=query.preferred_format)

        draft = self.advisor.render(
            query, retrieval, forecast, roadmap,
            GovernanceReport(True, [], False, False), niche)
        governance = self.governance.review(
            retrieval, forecast, roadmap, draft, niche, prev_pct, is_recalc)
        # re-render with the real governance notes now that we have them
        draft = self.advisor.render(query, retrieval, forecast, roadmap,
                                    governance, niche)
        return forecast, governance, roadmap, draft

    # ------------------------------------------------------------------ #
    def _maybe_narrate(self, draft: str, result: PipelineResult) -> str:
        prose = self.llm.narrate(
            system=(
                "You are a careful career advisor. Rephrase the analysis below into "
                "2-4 short paragraphs for the user. Do NOT change any number, "
                "percentage, course name, provider, or the non-guarantee disclaimer. "
                "Keep it plain and honest; do not imply a job guarantee."
            ),
            prompt=draft,
        )
        if not prose:
            return draft
        return prose + "\n\n--- structured detail ---\n" + draft
