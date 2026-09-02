from skillkartz.agents.forecast_analytics import ForecastAndAnalyticsBot
from skillkartz.agents.job_market_research import JobMarketResearchBot
from skillkartz.agents.skill_intelligence import SkillIntelligenceBot
from skillkartz.datastore import DataStore


def _forecast(store, skill, sector, location=None):
    si = SkillIntelligenceBot(store)
    jm = JobMarketResearchBot(store)
    fc = ForecastAndAnalyticsBot(store)
    res = si.resolve(skill)
    retr = jm.research(res.query_terms, res.canonical, sector, location,
                       res.exact_skill_names, res.related_terms)
    fresh = jm.freshness_ratio(retr)
    return retr, fc.forecast(retr, sector, res.exact_skill_names, location, fresh)


def test_percentage_is_exactly_count_over_total(store: DataStore):
    retr, fcast = _forecast(store, "SQL", "Banking")
    t = fcast.target()
    assert t.availability_pct == round(100.0 * t.skill_postings / t.total_postings, 2)


def test_below_benchmark_triggers_when_under_threshold(store: DataStore):
    _, fcast = _forecast(store, "Python", "Education")
    assert fcast.below_benchmark is True
    assert fcast.benchmark_pct == 35.0


def test_above_benchmark_when_over_threshold(store: DataStore):
    _, fcast = _forecast(store, "SQL", "Banking")
    assert fcast.below_benchmark is False


def test_sectors_are_ranked_by_availability(store: DataStore):
    _, fcast = _forecast(store, "Python", "Technology")
    pcts = [s.availability_pct for s in fcast.sector_forecasts]
    assert pcts == sorted(pcts, reverse=True)


def test_small_numerator_sets_caveat_flag(store: DataStore):
    # Welding barely appears in online Banking postings
    _, fcast = _forecast(store, "Welding", "Banking")
    assert fcast.small_sample is True


def test_confidence_between_zero_and_one_with_band(store: DataStore):
    _, fcast = _forecast(store, "SQL", "Banking")
    assert 0.0 <= fcast.confidence <= 1.0
    assert fcast.confidence_band in {"low", "medium", "high"}


def test_reasoning_trace_is_populated(store: DataStore):
    _, fcast = _forecast(store, "SQL", "Banking")
    joined = " ".join(fcast.reasoning_steps)
    assert "* 100 =" in joined and "confidence:" in joined
