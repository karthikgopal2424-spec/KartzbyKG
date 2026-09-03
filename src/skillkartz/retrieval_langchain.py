"""Optional LangChain-backed semantic retrieval (the design's "vector database").

`text.py` ships a dependency-free TF-IDF index. This module offers a drop-in
alternative built on LangChain's retrieval abstractions - a real
``Embeddings`` implementation feeding a ``VectorStore`` / ``VectorStoreRetriever``
- so the semantic layer can be swapped without touching a single agent.

    DataStore(retrieval_backend="langchain")          # uses this module
    DataStore(retrieval_backend="tfidf")   (default)  # uses text.py

The default embedding here (`HashingEmbeddings`) is deterministic, offline and
dependency-free on purpose: it keeps the capstone's "runs anywhere, same answer
every time" property while still exercising the LangChain code path. Point it at
a real model by passing any ``langchain_core.embeddings.Embeddings`` instance -
e.g. ``langchain_huggingface.HuggingFaceEmbeddings(model_name=
"sentence-transformers/all-MiniLM-L6-v2")`` (install ``skillkartz[embeddings-hf]``).

Only the *semantic-drift surface* in ``DataStore.retrieve`` consults this index;
the headline percentage is an exact-match count over structured skill lists and
is therefore identical whichever backend is active.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Optional

try:
    from langchain_core.embeddings import Embeddings
except ImportError as exc:  # pragma: no cover - guarded by the extra
    raise ImportError(
        "the LangChain retrieval backend needs 'langchain-core' "
        "(pip install \"skillkartz[embeddings]\")"
    ) from exc

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#/-]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbeddings(Embeddings):
    """Deterministic, offline embedding: signed token hashing into a fixed-width
    dense vector (the classic 'hashing trick', cf. ``sklearn`` HashingVectorizer),
    then L2-normalised so a dot product is a cosine similarity.

    Not as expressive as a trained model, but zero-dependency, reproducible, and
    enough to demonstrate the retriever plumbing. Swap in a real ``Embeddings``
    for production-grade semantics.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big")
            sign = 1.0 if (digest[8] & 1) else -1.0
            vec[h % self.dim] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LangChainSemanticIndex:
    """Duck-type-compatible with ``text.TfidfIndex`` - exposes ``build``,
    ``query_vector`` and ``score`` so ``DataStore.retrieve`` needs no changes.

    Also exposes ``as_retriever()`` returning a genuine LangChain
    ``VectorStoreRetriever`` for anyone wanting to compose it into an LCEL chain.
    """

    def __init__(self, embeddings: Optional[Embeddings] = None) -> None:
        self.embeddings: Embeddings = embeddings or HashingEmbeddings()
        self._vectors: dict[str, list[float]] = {}
        self._texts: dict[str, str] = {}
        self._store = None  # lazily built LangChain InMemoryVectorStore

    # -- TfidfIndex-compatible surface -------------------------------- #
    def build(self, documents: Iterable[tuple[str, str]]) -> "LangChainSemanticIndex":
        ids: list[str] = []
        texts: list[str] = []
        for doc_id, text in documents:
            ids.append(doc_id)
            texts.append(text)
        for doc_id, text, vec in zip(ids, texts, self.embeddings.embed_documents(texts)):
            self._vectors[doc_id] = vec
            self._texts[doc_id] = text
        return self

    def query_vector(self, terms: Iterable[str]) -> list[float]:
        return self.embeddings.embed_query(" ".join(terms))

    def score(self, doc_id: str, query_vec: list[float]) -> float:
        doc_vec = self._vectors.get(doc_id)
        if not doc_vec:
            return 0.0
        # both sides are L2-normalised, so the dot product is cosine similarity
        return sum(a * b for a, b in zip(doc_vec, query_vec))

    def __contains__(self, doc_id: str) -> bool:  # pragma: no cover - convenience
        return doc_id in self._vectors

    # -- genuine LangChain retriever --------------------------------- #
    def as_retriever(self, **kwargs):
        """A LangChain ``VectorStoreRetriever`` over the same postings."""
        if self._store is None:
            from langchain_core.documents import Document
            from langchain_core.vectorstores import InMemoryVectorStore

            self._store = InMemoryVectorStore(self.embeddings)
            self._store.add_documents(
                [Document(page_content=self._texts[i], metadata={"id": i})
                 for i in self._texts],
                ids=list(self._texts),
            )
        return self._store.as_retriever(**kwargs)
