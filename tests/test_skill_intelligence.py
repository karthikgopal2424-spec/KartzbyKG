from skillkartz.agents.skill_intelligence import SkillIntelligenceBot
from skillkartz.datastore import DataStore


def test_synonym_resolves_to_canonical(store: DataStore):
    bot = SkillIntelligenceBot(store)
    res = bot.resolve("ML")
    assert res.canonical == "Machine Learning"
    assert res.matched_via == "synonym"


def test_case_insensitive_canonical(store: DataStore):
    bot = SkillIntelligenceBot(store)
    assert bot.resolve("python").canonical == "Python"


def test_fuzzy_match(store: DataStore):
    bot = SkillIntelligenceBot(store)
    res = bot.resolve("power-bi")
    assert res.canonical == "Power BI"


def test_unresolved_skill(store: DataStore):
    bot = SkillIntelligenceBot(store)
    res = bot.resolve("underwater basket weaving")
    assert not res.resolved
    assert res.matched_via == "unresolved"


def test_query_terms_expand_but_exact_names_do_not(store: DataStore):
    bot = SkillIntelligenceBot(store)
    res = bot.resolve("Machine Learning")
    assert "scikit-learn" in res.query_terms          # expansion for retrieval
    assert res.exact_skill_names == {"Machine Learning"}  # counting stays strict
