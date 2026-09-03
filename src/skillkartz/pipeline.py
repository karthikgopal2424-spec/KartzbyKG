"""Top-level entry point: text query in, explainable answer out.

    from skillkartz import ForecastAIPipeline
    pipe = ForecastAIPipeline()
    result = pipe.run("Is Python in demand for banking jobs in Chicago?")
    print(result.response_text)

Two things are pluggable, both defaulting to the dependency-free path:

    ForecastAIPipeline(backend="graph")        # LangGraph orchestration
    ForecastAIPipeline(retrieval="langchain")  # LangChain Embeddings + VectorStore

The agents, the arithmetic and the returned ``PipelineResult`` are identical
whichever backend is selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agents.supervisor import SupervisorBot
from .config import SETTINGS
from .datastore import DataStore
from .llm import LLMClient
from .models import PipelineResult, UserQuery

__all__ = ["ForecastAIPipeline", "PipelineResult"]


class ForecastAIPipeline:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        llm: Optional[LLMClient] = None,
        backend: Optional[str] = None,
        retrieval: Optional[str] = None,
    ) -> None:
        self.backend = (backend or SETTINGS.orchestrator).lower()
        self.store = DataStore(data_dir, retrieval_backend=retrieval)
        self.llm = llm or LLMClient()
        self.supervisor = SupervisorBot(self.store, self.llm)
        self._graph = None
        if self.backend == "graph":
            from .graph_pipeline import GraphOrchestrator
            self._graph = GraphOrchestrator(self.supervisor)
        elif self.backend != "native":
            raise ValueError(
                f"unknown backend {self.backend!r} (use 'native' or 'graph')"
            )

    # ------------------------------------------------------------------ #
    def run(self, query_text: str, **overrides) -> PipelineResult:
        """`overrides` may pin any UserQuery field (skill, sector, location,
        experience, budget_usd, timeframe_weeks, preferred_format, goal)."""
        if self._graph is not None:
            return self._graph.run(query_text, overrides)

        # Normalise the free text (typo -> known term) before intent capture,
        # then keep the user's original wording on the query for the audit log.
        rewrite = self.supervisor.rewriter.rewrite(query_text)
        query: UserQuery = self.supervisor.advisor.capture(rewrite.normalized_text, **overrides)
        query.raw_text = query_text
        return self.supervisor.run(query, rewrite=rewrite)

    # ------------------------------------------------------------------ #
    def record_feedback(self, query: UserQuery, kind: str, provider: Optional[str] = None) -> None:
        """kind: 'accepted' | 'rejected'. Feeds the Supervisor's session memory."""
        self.supervisor.record_feedback(query, kind, provider)

    @property
    def memory(self) -> dict:
        return self.supervisor.memory
