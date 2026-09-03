"""Optional LangGraph orchestration backend.

The hand-rolled Supervisor in ``agents/supervisor.py`` is already a state graph:
plan -> validate -> fan-out -> conditional niche branch -> forecast -> a
Governance loop with one recalculation -> deliver or escalate. This module
expresses that exact shape as a LangGraph ``StateGraph`` instead.

    ForecastAIPipeline(backend="graph")          # uses this module
    ForecastAIPipeline(backend="native")  (default)

Nothing about the analysis changes: every node delegates to the *same* agent
instances (via a ``SupervisorBot``), reuses its ``_forecast_and_review`` /
``_maybe_narrate`` helpers and its session-memory dict, and the graph returns the
same ``PipelineResult``. What LangGraph adds is an explicit, inspectable graph
(``GraphOrchestrator.mermaid()``), a checkpointer for per-thread state, and a
standard place for the human-escalation interrupt.

Needs ``skillkartz[graph]`` (``pip install "skillkartz[graph]"``).
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Optional, TypedDict

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - guarded by the extra
    raise ImportError(
        "the LangGraph backend needs 'langgraph' (pip install \"skillkartz[graph]\")"
    ) from exc

from .agents.supervisor import SupervisorBot
from .models import AgentTrace, PipelineResult, UserQuery


class _State(TypedDict, total=False):
    # inputs
    query_text: str
    overrides: dict
    # progressively filled
    rewrite: Any
    query: UserQuery
    clar: list          # pending clarification questions
    resolution: Any
    retrieval: Any
    freshness: float
    niche: Any
    benchmark_pct: Optional[float]
    benchmark_source: str
    prev_pct: Optional[float]
    attempts: int
    forecast: Any
    governance: Any
    roadmap: Any
    draft: str
    status: str
    response_text: str
    trace: Annotated[list, operator.add]


class GraphOrchestrator:
    """LangGraph equivalent of ``SupervisorBot.run``."""

    def __init__(self, supervisor: SupervisorBot) -> None:
        self.sup = supervisor
        self._app = self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        g = StateGraph(_State)

        g.add_node("rewrite", self._n_rewrite)
        g.add_node("capture", self._n_capture)
        g.add_node("resolve", self._n_resolve)
        g.add_node("research", self._n_research)
        g.add_node("niche", self._n_niche)
        g.add_node("forecast", self._n_forecast)
        g.add_node("clarify", self._n_clarify)
        g.add_node("escalate", self._n_escalate)
        g.add_node("deliver", self._n_deliver)

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "capture")
        g.add_conditional_edges("capture", self._route_after_capture,
                                {"clarify": "clarify", "resolve": "resolve"})
        g.add_conditional_edges("resolve", self._route_after_resolve,
                                {"clarify": "clarify", "research": "research"})
        g.add_conditional_edges("research", self._route_after_research,
                                {"niche": "niche", "forecast": "forecast"})
        g.add_edge("niche", "forecast")
        g.add_conditional_edges("forecast", self._route_after_forecast,
                                {"forecast": "forecast",   # Governance recalculation loop
                                 "escalate": "escalate",
                                 "deliver": "deliver"})
        g.add_edge("clarify", END)
        g.add_edge("escalate", END)
        g.add_edge("deliver", END)

        return g.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------ #
    # nodes  (each delegates to the real agent on self.sup)
    # ------------------------------------------------------------------ #
    def _n_rewrite(self, state: _State) -> dict:
        rw = self.sup.rewriter.rewrite(state["query_text"])
        summary = ("normalised input: " + "; ".join(rw.notes)) if rw.changed \
            else "no changes - input already clean"
        return {"rewrite": rw,
                "trace": [AgentTrace(self.sup.rewriter.name, summary,
                                     {"normalized": rw.normalized_text})]}

    def _n_capture(self, state: _State) -> dict:
        q = self.sup.advisor.capture(state["rewrite"].normalized_text,
                                     **state.get("overrides", {}))
        q.raw_text = state["query_text"]
        questions = self.sup.advisor.clarification_questions(q)
        return {"query": q, "clar": questions,
                "attempts": 0, "benchmark_source": "standard-threshold",
                "prev_pct": self.sup.memory["results"].get(self.sup._query_key(q)),
                "trace": [AgentTrace(self.sup.advisor.name, "captured intent",
                                     {"skill": q.skill, "sector": q.sector,
                                      "location": q.location})]}

    def _n_resolve(self, state: _State) -> dict:
        res = self.sup.skill_intel.resolve(state["query"].skill)
        out: dict = {"resolution": res,
                     "trace": [AgentTrace(self.sup.skill_intel.name,
                                          f"resolved '{state['query'].skill}' via "
                                          f"{res.matched_via}",
                                          {"canonical": res.canonical})]}
        if not res.resolved:
            out["clar"] = [
                f"I couldn't map '{state['query'].skill}' to a known skill. "
                f"Did you mean one of: "
                f"{', '.join(list(self.sup.store.canonical_skills())[:10])}?"
            ]
        return out

    def _n_research(self, state: _State) -> dict:
        res = state["resolution"]
        q = state["query"]
        retr = self.sup.research.research(
            res.query_terms, res.canonical, q.sector, q.location,
            res.exact_skill_names, res.related_terms)
        return {"retrieval": retr,
                "freshness": self.sup.research.freshness_ratio(retr),
                "trace": [AgentTrace(self.sup.research.name, retr.coverage_note,
                                     {"matched": len(retr.matched),
                                      "insufficient_data": retr.insufficient_data})]}

    def _n_niche(self, state: _State) -> dict:
        res, q, retr = state["resolution"], state["query"], state["retrieval"]
        local = self.sup.local_intel.collect(res.canonical, q.location)
        niche = self.sup.niche_gov.assess(res.canonical, q.sector, q.location,
                                          len(retr.matched), local)
        return {"niche": niche,
                "benchmark_pct": niche.benchmark_pct,
                "benchmark_source": "niche-benchmark",
                "trace": [
                    AgentTrace(self.sup.name, "insufficient standard data - "
                               "activating niche extension pair"),
                    AgentTrace(self.sup.niche_gov.name,
                               f"{niche.classification}, benchmark {niche.benchmark_pct}%",
                               {"rationale": niche.rationale})]}

    def _n_forecast(self, state: _State) -> dict:
        attempts = state.get("attempts", 0) + 1
        is_recalc = attempts > 1
        res, q = state["resolution"], state["query"]
        forecast, governance, roadmap, draft = self.sup._forecast_and_review(
            q, state["retrieval"], res.canonical, res.exact_skill_names,
            state["freshness"], state.get("benchmark_pct"),
            state.get("benchmark_source", "standard-threshold"),
            state.get("niche"), state.get("prev_pct"), is_recalc)
        verb = "recalculated" if is_recalc else "forecast"
        return {"forecast": forecast, "governance": governance,
                "roadmap": roadmap, "draft": draft, "attempts": attempts,
                "trace": [AgentTrace(self.sup.forecaster.name,
                                     f"{verb} {forecast.target_sector} = "
                                     f"{forecast.target().availability_pct:.2f}% "
                                     f"(confidence {forecast.confidence})",
                                     {"reasoning": forecast.reasoning_steps}),
                          AgentTrace(self.sup.governance.name,
                                     "passed" if governance.passed else
                                     ("escalate" if governance.escalate
                                      else "passed with warnings"),
                                     {"issues": [i.__dict__ for i in governance.issues],
                                      "recalc": governance.recalculation_requested})]}

    def _n_clarify(self, state: _State) -> dict:
        qs = state.get("clar") or []
        return {"status": "clarification_needed", "clar": qs,
                "response_text": self.sup.advisor.render_clarification(qs),
                "trace": [AgentTrace(self.sup.advisor.name, "requested clarification",
                                     {"questions": qs})]}

    def _n_escalate(self, state: _State) -> dict:
        g = state["governance"]
        return {"status": "escalated",
                "response_text": self.sup.advisor.render_escalation(
                    state["query"], g.escalation_reason or "governance escalation"),
                "trace": [AgentTrace(self.sup.advisor.name, "delivered escalation notice")]}

    def _n_deliver(self, state: _State) -> dict:
        forecast = state["forecast"]
        key = self.sup._query_key(state["query"])
        self.sup.memory["results"][key] = forecast.target().availability_pct
        result = PipelineResult(query=state["query"], status="answered")
        text = self.sup._maybe_narrate(state["draft"], result)
        return {"status": "answered", "response_text": text,
                "trace": [AgentTrace(self.sup.advisor.name,
                                     "delivered explainable response")]}

    # ------------------------------------------------------------------ #
    # routers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _route_after_capture(state: _State) -> str:
        return "clarify" if state.get("clar") else "resolve"

    @staticmethod
    def _route_after_resolve(state: _State) -> str:
        return "research" if state["resolution"].resolved else "clarify"

    @staticmethod
    def _route_after_research(state: _State) -> str:
        return "niche" if state["retrieval"].insufficient_data else "forecast"

    @staticmethod
    def _route_after_forecast(state: _State) -> str:
        g = state["governance"]
        if g.recalculation_requested and state.get("attempts", 0) < 2:
            return "forecast"
        if g.escalate:
            return "escalate"
        return "deliver"

    # ------------------------------------------------------------------ #
    def run(self, query_text: str, overrides: Optional[dict] = None,
            thread_id: Optional[str] = None) -> PipelineResult:
        config = {"configurable": {"thread_id": thread_id or f"run-{uuid.uuid4().hex}"}}
        final = self._app.invoke(
            {"query_text": query_text, "overrides": overrides or {}, "trace": []},
            config=config,
        )
        return self._assemble(final)

    @staticmethod
    def _assemble(state: _State) -> PipelineResult:
        result = PipelineResult(
            query=state["query"],
            status=state.get("status", "answered"),
            clarification_questions=state.get("clar") or [] if
            state.get("status") == "clarification_needed" else [],
            rewrite=state.get("rewrite"),
            retrieval=state.get("retrieval"),
            niche=state.get("niche"),
            forecast=state.get("forecast"),
            roadmap=state.get("roadmap"),
            governance=state.get("governance"),
            response_text=state.get("response_text", ""),
            trace=list(state.get("trace", [])),
        )
        return result

    # ------------------------------------------------------------------ #
    def mermaid(self) -> str:
        """The compiled graph as a Mermaid diagram (handy for the README)."""
        return self._app.get_graph().draw_mermaid()
