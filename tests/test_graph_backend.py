"""The LangGraph orchestration backend must be a behavioural no-op: same agents,
same PipelineResult as the hand-rolled Supervisor."""

import pytest

pytest.importorskip("langgraph")

from skillkartz.llm import LLMClient  # noqa: E402
from skillkartz.pipeline import ForecastAIPipeline  # noqa: E402

CASES = [
    ("Is Python in demand for banking jobs in Chicago?", {}),
    ("How in demand is SQL for banking jobs?", {"sector": "Banking"}),
    ("Is Python in demand for education jobs?",
     {"sector": "Education", "budget_usd": 600, "timeframe_weeks": 24}),
    ("Is Python worth learning?", {}),                                  # clarification
    ("demand for flux capacitor design", {"sector": "Technology"}),     # unknown skill
    ("HVAC roadmap", {"sector": "Manufacturing", "location": "Chicago"}),   # niche benchmark
    ("HVAC roadmap", {"sector": "Manufacturing", "location": "Bangalore"}),  # escalation
]


@pytest.fixture(scope="module")
def native():
    return ForecastAIPipeline(llm=LLMClient(enabled=False), backend="native")


@pytest.fixture(scope="module")
def graph():
    return ForecastAIPipeline(llm=LLMClient(enabled=False), backend="graph")


@pytest.mark.parametrize("query,overrides", CASES)
def test_graph_matches_native(native, graph, query, overrides):
    a = native.run(query, **overrides)
    b = graph.run(query, **overrides)
    assert a.status == b.status
    assert a.response_text == b.response_text
    if a.forecast is not None:
        assert b.forecast is not None
        assert a.forecast.target().availability_pct == b.forecast.target().availability_pct
        assert a.forecast.below_benchmark == b.forecast.below_benchmark
    if a.roadmap is not None and a.roadmap.feasible:
        assert [s.course_id for s in a.roadmap.steps] == [s.course_id for s in b.roadmap.steps]


def test_graph_trace_covers_core_stages(graph):
    r = graph.run("Python demand in banking", sector="Banking")
    agents = {t.agent for t in r.trace}
    assert {"Query Rewriter Bot", "Skill Intelligence Bot",
            "Job Market Research Bot", "Forecast and Analytics Bot",
            "Governance & QA Bot"} <= agents


def test_graph_exposes_a_mermaid_diagram(graph):
    m = graph._graph.mermaid()
    assert "forecast" in m and "escalate" in m and "clarify" in m
