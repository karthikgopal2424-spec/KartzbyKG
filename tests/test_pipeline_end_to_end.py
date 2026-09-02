from skillkartz.config import NON_GUARANTEE_DISCLAIMER
from skillkartz.pipeline import ForecastAIPipeline


def test_below_threshold_query_produces_roadmap(pipeline: ForecastAIPipeline):
    r = pipeline.run("How in demand is Python for education jobs?",
                     sector="Education")
    assert r.status == "answered"
    assert r.forecast.below_benchmark
    assert r.roadmap is not None and r.roadmap.feasible
    assert r.roadmap.steps


def test_above_threshold_query_has_no_roadmap(pipeline: ForecastAIPipeline):
    r = pipeline.run("SQL demand in banking", sector="Banking")
    assert r.status == "answered"
    assert not r.forecast.below_benchmark
    assert r.roadmap is None


def test_response_always_carries_the_output_contract(pipeline: ForecastAIPipeline):
    r = pipeline.run("Tableau demand in retail", sector="Retail")
    assert NON_GUARANTEE_DISCLAIMER[:40] in r.response_text
    assert "Data coverage:" in r.response_text
    assert "Confidence" in r.response_text
    assert "%" in r.response_text


def test_missing_sector_asks_for_clarification(pipeline: ForecastAIPipeline):
    r = pipeline.run("Is Python worth learning?")
    assert r.status == "clarification_needed"
    assert r.clarification_questions


def test_unknown_skill_asks_for_clarification(pipeline: ForecastAIPipeline):
    r = pipeline.run("demand for flux capacitor design", sector="Technology")
    assert r.status == "clarification_needed"


def test_niche_skill_uses_niche_benchmark(pipeline: ForecastAIPipeline):
    r = pipeline.run("HVAC roadmap", sector="Manufacturing", location="Chicago")
    assert r.niche is not None
    assert r.forecast.benchmark_source == "niche-benchmark"
    assert r.forecast.benchmark_pct != 35.0


def test_niche_skill_with_no_local_evidence_escalates(pipeline: ForecastAIPipeline):
    r = pipeline.run("HVAC roadmap", sector="Manufacturing", location="Bangalore")
    assert r.status == "escalated"
    assert "escalation" in r.response_text.lower() or "human" in r.response_text.lower()


def test_trace_covers_every_core_stage(pipeline: ForecastAIPipeline):
    r = pipeline.run("Python demand in banking", sector="Banking")
    agents = {t.agent for t in r.trace}
    assert "Skill Intelligence Bot" in agents
    assert "Job Market Research Bot" in agents
    assert "Forecast and Analytics Bot" in agents
    assert "Governance & QA Bot" in agents


def test_result_serialises_to_json_dict(pipeline: ForecastAIPipeline):
    import json
    r = pipeline.run("Python demand in banking", sector="Banking")
    json.dumps(r.to_dict(), default=str)  # must not raise


def test_session_memory_records_feedback(pipeline: ForecastAIPipeline):
    r = pipeline.run("Python demand in retail", sector="Retail")
    pipeline.record_feedback(r.query, "accepted", provider="OpenLearn")
    assert pipeline.memory["accepted"]
    assert pipeline.memory["provider_bias"]["OpenLearn"] > 0
