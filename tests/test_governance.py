import dataclasses

from skillkartz.agents.forecast_analytics import ForecastAndAnalyticsBot
from skillkartz.agents.governance_qa import GovernanceAndQABot
from skillkartz.agents.job_market_research import JobMarketResearchBot
from skillkartz.agents.skill_intelligence import SkillIntelligenceBot
from skillkartz.datastore import DataStore
from skillkartz.models import Roadmap, RoadmapStep


def _pack(store, skill="SQL", sector="Banking"):
    si, jm, fc = (SkillIntelligenceBot(store), JobMarketResearchBot(store),
                  ForecastAndAnalyticsBot(store))
    res = si.resolve(skill)
    retr = jm.research(res.query_terms, res.canonical, sector, None,
                       res.exact_skill_names, res.related_terms)
    fcast = fc.forecast(retr, sector, res.exact_skill_names, None,
                        jm.freshness_ratio(retr))
    return retr, fcast


GOOD_RESPONSE = (
    "Observed job-posting share : 57.7%\nConfidence : 0.9\n"
    "Data coverage: ... .\n"
    "Disclaimer: This figure is the observed share of analysed job postings "
    "that mention the skill."
)


def test_clean_forecast_passes(store: DataStore):
    retr, fcast = _pack(store)
    rep = GovernanceAndQABot(store).review(retr, fcast, None, GOOD_RESPONSE)
    assert rep.passed and not rep.escalate


def test_arithmetic_tamper_is_caught(store: DataStore):
    retr, fcast = _pack(store)
    tampered = dataclasses.replace(fcast)
    tampered.sector_forecasts[0].availability_pct += 20.0
    rep = GovernanceAndQABot(store).review(retr, tampered, None, GOOD_RESPONSE)
    assert rep.escalate
    assert any(i.check == "arithmetic" for i in rep.issues)


def test_missing_disclaimer_is_blocked(store: DataStore):
    retr, fcast = _pack(store)
    rep = GovernanceAndQABot(store).review(retr, fcast, None, "57.7% and nothing else")
    assert rep.escalate
    assert any(i.check == "output-contract" for i in rep.issues)


def test_discontinued_course_recommendation_is_caught(store: DataStore):
    retr, fcast = _pack(store)
    dead = next(c for c in store.courses if not c.active)
    rm = Roadmap(
        feasible=True,
        steps=[RoadmapStep(1, dead.skill, dead.id, dead.title, dead.provider,
                           dead.cost_usd, dead.duration_weeks, dead.format, 5.0, 0.5)],
        total_cost_usd=dead.cost_usd, total_weeks=dead.duration_weeks,
        projected_final_pct=40.0, demand_gap_pp=10.0,
    )
    rep = GovernanceAndQABot(store).review(retr, fcast, rm, GOOD_RESPONSE)
    assert rep.escalate
    assert any(i.check == "groundedness" for i in rep.issues)


def test_drift_requests_recalculation_once(store: DataStore):
    retr, fcast = _pack(store)
    target_pct = fcast.target().availability_pct
    rep = GovernanceAndQABot(store).review(
        retr, fcast, None, GOOD_RESPONSE, previous_target_pct=target_pct + 40.0)
    assert rep.recalculation_requested and not rep.escalate
    # ... but on the recalculation pass it must not loop forever
    rep2 = GovernanceAndQABot(store).review(
        retr, fcast, None, GOOD_RESPONSE, previous_target_pct=target_pct + 40.0,
        is_recalculation=True)
    assert not rep2.recalculation_requested
