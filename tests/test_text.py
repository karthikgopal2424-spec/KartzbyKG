from skillkartz.text import TfidfIndex, tokenize


def test_tokenize_keeps_tech_tokens():
    assert tokenize("C++, CI/CD and Node.js") == ["c++", "ci/cd", "and", "node.js"]


def test_similarity_ranks_related_docs_higher():
    idx = TfidfIndex().build([
        ("a", "python pandas data pipeline engineer"),
        ("b", "senior java spring backend developer"),
        ("c", "python scripting automation numpy"),
    ])
    q = idx.query_vector(["python", "pandas", "numpy"])
    sa, sb, sc = idx.score("a", q), idx.score("b", q), idx.score("c", q)
    assert sa > sb and sc > sb
    assert 0.0 <= sb < 0.2
