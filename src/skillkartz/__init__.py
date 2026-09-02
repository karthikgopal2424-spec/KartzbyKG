"""SKILLKARTZ / ForecastAI - supervisor-led multi-bot career forecasting agent.

This package is the runnable capstone implementation of the system described in
``SKILLKARTZ_AI_Agent_updated_6.docx``. See ``docs/DESIGN_MAP.md`` for the
section-by-section mapping between the design document and the code.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .pipeline import ForecastAIPipeline, PipelineResult

__all__ = ["ForecastAIPipeline", "PipelineResult", "__version__"]
