from skillkartz.agents.upskilling import UpskillingBot
from skillkartz.config import SETTINGS
from skillkartz.datastore import DataStore


def _bot(store):
    return UpskillingBot(store)


def test_roadmap_is_staged_and_ordered(store: DataStore):
    rm = _bot(store).build_roadmap(
        "SQL", "Retail", current_pct=22.0, benchmark_pct=35.0,
        budget_usd=2000, timeframe_weeks=40, preferred_format=None,
    )
    assert rm.feasible
    assert [s.stage for s in rm.steps] == list(range(1, len(rm.steps) + 1))
    assert len(rm.steps) <= SETTINGS.tot_max_depth


def test_hard_budget_constraint_is_respected(store: DataStore):
    rm = _bot(store).build_roadmap(
        "SQL", "Retail", current_pct=10.0, benchmark_pct=35.0,
        budget_usd=120, timeframe_weeks=60, preferred_format=None,
    )
    if rm.feasible:
        assert rm.total_cost_usd <= 120
    else:
        assert any("infeasible" in n or "budget" in n for n in rm.notes)


def test_infeasible_reports_rather_than_fabricates(store: DataStore):
    rm = _bot(store).build_roadmap(
        "SQL", "Retail", current_pct=10.0, benchmark_pct=35.0,
        budget_usd=1.0, timeframe_weeks=1, preferred_format=None,
    )
    assert rm.feasible is False
    assert rm.steps == []
    assert rm.notes


def test_search_is_deterministic(store: DataStore):
    kw = dict(target_skill="Python", target_sector="Banking", current_pct=15.0,
              benchmark_pct=35.0, budget_usd=1500, timeframe_weeks=40,
              preferred_format=None)
    a = _bot(store).build_roadmap(**kw)
    b = _bot(store).build_roadmap(**kw)
    assert [(s.skill, s.course_id) for s in a.steps] == \
           [(s.skill, s.course_id) for s in b.steps]


def test_beam_search_stats_recorded(store: DataStore):
    rm = _bot(store).build_roadmap(
        "Python", "Banking", current_pct=15.0, benchmark_pct=35.0,
        budget_usd=3000, timeframe_weeks=52, preferred_format=None,
    )
    assert rm.search_stats.get("beam_width") == SETTINGS.tot_beam_width
    assert rm.search_stats.get("nodes_expanded", 0) > 0
