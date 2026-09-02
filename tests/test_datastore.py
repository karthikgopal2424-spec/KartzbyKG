from skillkartz.datastore import DataStore


def test_corpora_load(store: DataStore):
    assert len(store.postings) > 400
    assert store.courses and store.local_signals
    assert "Banking" in store.sectors()
    assert "Chicago" in store.locations()


def test_retrieve_returns_full_matching_set_not_topk(store: DataStore):
    r = store.retrieve(
        query_terms=["SQL"], canonical_skill="SQL",
        sector="Banking", location=None, exact_skill_names={"SQL"},
    )
    # every matched posting really lists the skill (exact-match counting rule)
    assert all("SQL" in p.skills for p in r.matched)
    # and the matched set is the *full* set, not a 3-5 slice
    assert len(r.matched) > 20
    assert len(r.matched) <= len(r.sector_universe)


def test_dedupe_removes_injected_duplicates(store: DataStore):
    r = store.retrieve(
        query_terms=["Python"], canonical_skill="Python",
        sector="Technology", location=None, exact_skill_names={"Python"},
    )
    assert r.duplicates_removed > 0


def test_percentage_matches_ground_truth_direction(store: DataStore):
    # Technology has a much higher Python propensity than Education by construction
    tech = store.sector_skill_pct("Python", "Technology")
    edu = store.sector_skill_pct("Python", "Education")
    assert tech > 40 > edu


def test_thin_universe_flags_insufficient_data(store: DataStore):
    # A niche/trade skill with almost no online postings
    r = store.retrieve(
        query_terms=["Welding"], canonical_skill="Welding",
        sector="Banking", location=None, exact_skill_names={"Welding"},
    )
    assert r.insufficient_data
