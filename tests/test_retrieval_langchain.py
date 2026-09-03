"""The LangChain retrieval backend swaps the semantic index only. The headline
percentage (an exact-match count over structured skill lists) must be identical
to the TF-IDF backend."""

import pytest

pytest.importorskip("langchain_core")

from skillkartz.datastore import DataStore  # noqa: E402
from skillkartz.llm import LLMClient  # noqa: E402
from skillkartz.pipeline import ForecastAIPipeline  # noqa: E402

CASES = [
    ("Is Python in demand for banking jobs in Chicago?", {}),
    ("React demand", {"sector": "Technology"}),
    ("Tableau demand in retail", {"sector": "Retail"}),
    ("How in demand is Machine Learning for banking?", {"sector": "Banking"}),
]


@pytest.fixture(scope="module")
def tfidf():
    return ForecastAIPipeline(llm=LLMClient(enabled=False), retrieval="tfidf")


@pytest.fixture(scope="module")
def langchain():
    return ForecastAIPipeline(llm=LLMClient(enabled=False), retrieval="langchain")


def test_backend_selection():
    from skillkartz.retrieval_langchain import LangChainSemanticIndex
    from skillkartz.text import TfidfIndex
    assert isinstance(DataStore(retrieval_backend="tfidf")._index, TfidfIndex)
    assert isinstance(DataStore(retrieval_backend="langchain")._index, LangChainSemanticIndex)


@pytest.mark.parametrize("query,overrides", CASES)
def test_headline_percentage_is_identical(tfidf, langchain, query, overrides):
    a = tfidf.run(query, **overrides)
    b = langchain.run(query, **overrides)
    assert a.status == b.status
    assert a.forecast.target().availability_pct == b.forecast.target().availability_pct
    assert a.forecast.below_benchmark == b.forecast.below_benchmark


def test_hashing_embeddings_are_deterministic():
    from skillkartz.retrieval_langchain import HashingEmbeddings
    e = HashingEmbeddings(dim=64)
    v1 = e.embed_query("python pandas scripting")
    v2 = e.embed_query("python pandas scripting")
    assert v1 == v2
    assert len(v1) == 64
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9  # L2-normalised


def test_exposes_a_real_langchain_retriever():
    idx = DataStore(retrieval_backend="langchain")._index
    retriever = idx.as_retriever(search_kwargs={"k": 3})
    from langchain_core.retrievers import BaseRetriever
    assert isinstance(retriever, BaseRetriever)
    docs = retriever.invoke("python pandas data pipeline")
    assert len(docs) == 3
    assert all("id" in d.metadata for d in docs)
