from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skillkartz.datastore import DataStore  # noqa: E402
from skillkartz.pipeline import ForecastAIPipeline  # noqa: E402


@pytest.fixture(scope="session")
def store() -> DataStore:
    return DataStore()


@pytest.fixture()
def pipeline() -> ForecastAIPipeline:
    # LLM disabled explicitly so tests are deterministic and offline.
    from skillkartz.llm import LLMClient
    return ForecastAIPipeline(llm=LLMClient(enabled=False))
