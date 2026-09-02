from skillkartz.agents.query_rewriter import QueryRewriterBot
from skillkartz.datastore import DataStore
from skillkartz.pipeline import ForecastAIPipeline


def test_clean_input_is_left_untouched(store: DataStore):
    bot = QueryRewriterBot(store)
    rw = bot.rewrite("Is Python in demand for banking jobs in Chicago?")
    assert not rw.changed
    assert rw.corrections == []


def test_single_token_typo_snaps_to_sector_and_location(store: DataStore):
    bot = QueryRewriterBot(store)
    rw = bot.rewrite("is python in demand for bankng jobs in chicgo?")
    assert rw.normalized_text == "is python in demand for Banking jobs in Chicago?"
    cats = {c.category for c in rw.corrections}
    assert cats == {"sector", "location"}
    assert all(c.similarity >= 0.82 for c in rw.corrections)


def test_multi_word_skill_typo_is_corrected(store: DataStore):
    bot = QueryRewriterBot(store)
    rw = bot.rewrite("data analsis demand in retial")
    assert "Data Analysis" in rw.normalized_text
    assert "Retail" in rw.normalized_text


def test_short_and_common_words_are_never_corrected(store: DataStore):
    bot = QueryRewriterBot(store)
    rw = bot.rewrite("is it in demand for jobs and work")
    assert not rw.changed


def test_spaceless_multiword_alias_resolves(store: DataStore):
    # a single mistyped token for a two-word term still snaps to the
    # spaced canonical form the downstream matcher expects
    bot = QueryRewriterBot(store)
    assert bot.rewrite("is python in demand in newyork").normalized_text == \
        "is python in demand in New York"


def test_wildly_wrong_token_is_left_for_clarification(store: DataStore):
    # 'pyhtin' is only ~0.67 similar to 'python' - below the floor, so the
    # rewriter must NOT guess; the clarification guardrail owns this case.
    bot = QueryRewriterBot(store)
    rw = bot.rewrite("is pyhtin in demand for banking jobs?")
    assert "pyhtin" in rw.normalized_text
    assert all(c.category != "skill" for c in rw.corrections)


def test_near_miss_skill_typo_reaches_an_answer(pipeline: ForecastAIPipeline):
    r = pipeline.run("pythin demand in banking", sector="Banking")
    assert r.rewrite.changed
    assert r.query.skill == "Python"
    assert r.status == "answered"


def test_rewrite_is_recorded_on_the_result_and_trace(pipeline: ForecastAIPipeline):
    r = pipeline.run("tablaeu demand in retial", sector="Retail")
    assert r.rewrite is not None and r.rewrite.changed
    assert any(t.agent == "Query Rewriter Bot" for t in r.trace)
    # the user's original wording is preserved for the audit log
    assert r.query.raw_text == "tablaeu demand in retial"


def test_rewrite_survives_json_serialisation(pipeline: ForecastAIPipeline):
    import json
    r = pipeline.run("pythin demand in bankng", sector="Banking")
    blob = json.dumps(r.to_dict(), default=str)
    assert "Query Rewriter Bot" in blob
