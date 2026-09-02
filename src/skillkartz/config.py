"""Central configuration - every tunable threshold from the design doc lives here.

Keeping these in one place is deliberate: the Safety & Intervention Plan (design
doc section 12) treats the demand threshold, the confidence floor, and the
escalation triggers as governed policy values, not magic numbers scattered
through the agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    # --- Forecast / threshold logic (design doc sections 1, 4, 9) ------------
    demand_threshold_pct: float = 35.0
    """Below this % share of postings in the target sector, the Upskilling Bot fires."""

    small_sample_floor: int = 8
    """Fewer total postings than this in the target sector => mandatory caveat."""

    small_numerator_floor: int = 5
    """Fewer than this many skill-matching postings => the %% is not meaningful."""

    insufficient_data_postings: int = 6
    """At or below this many sector postings for the skill query => niche path."""

    trend_window_days: int = 120
    trend_delta_pp: float = 5.0
    """+/- percentage points between the two trailing windows to call rising/declining."""

    # --- Confidence scoring (design doc section 4 "assigns a confidence score") --
    confidence_floor: float = 0.45
    """Below this after one Governance-triggered recalculation => escalate to a human."""

    conf_weight_sample: float = 0.45
    conf_weight_coverage: float = 0.30
    conf_weight_freshness: float = 0.25
    conf_sample_saturation: int = 30
    conf_sources_saturation: int = 4
    freshness_window_days: int = 180

    # --- Tree-of-Thought roadmap search (design doc section 10) --------------
    tot_beam_width: int = 3
    tot_branching_factor: int = 4
    tot_max_depth: int = 4
    tot_uplift_damping: float = 0.30
    """Projected demand-gap closed per added complementary skill = its sector %% * damping."""

    tot_score_weights: dict = field(default_factory=lambda: {
        "demand_relevance": 0.35,
        "constraint_fit": 0.20,
        "quality": 0.20,
        "prerequisite": 0.15,
        "format_fit": 0.10,
    })

    # --- Niche benchmarking (design doc section 8) --------------------------
    niche_min_local_records: int = 4
    niche_source_reliability_floor: float = 0.50

    # --- Governance (design doc section 12 "Guardrails") -------------------
    governance_drift_pp: float = 25.0
    """Same-query result moving more than this vs a prior run => recalculation request."""

    # --- Retrieval (design doc section 9) ---------------------------------
    retrieval_similarity_floor: float = 0.08
    """Semantic-match score a posting must clear to enter the candidate set."""

    # --- Query Rewriter Bot (input normalisation, runs before intent capture) --
    rewriter_similarity_floor: float = 0.82
    """difflib ratio a mistyped token must reach to be auto-corrected to a
    known skill / sector / location term. Deliberately high: the design's
    'ask rather than guess' rule still owns anything below this."""

    rewriter_min_token_len: int = 4
    """Tokens shorter than this are never spell-corrected (too many false hits
    on short function words)."""

    # --- Optional LLM narration (design doc: "LLM supports conversation") --
    llm_model: str = os.environ.get("SKILLKARTZ_MODEL", "claude-sonnet-5")
    llm_enabled: bool = os.environ.get("SKILLKARTZ_LLM", "0") == "1"

    data_dir: Path = DEFAULT_DATA_DIR


SETTINGS = Settings()

NON_GUARANTEE_DISCLAIMER = (
    "This figure is the observed share of analysed job postings that mention the "
    "skill. It is not a prediction of your individual chance of being hired."
)
