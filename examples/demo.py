"""Runs a spread of scenarios that each exercise a different path through the
pipeline. No arguments, no network, no API key needed:

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skillkartz.pipeline import ForecastAIPipeline  # noqa: E402

SCENARIOS = [
    ("Query Rewriter - typo'd skill/sector still reaches an answer",
     "how in demand is pythin for bankng jobs?", {}),
    ("Above benchmark - no roadmap",
     "How in demand is SQL for banking jobs?", {"sector": "Banking"}),
    ("Below benchmark - Tree-of-Thought roadmap",
     "Is Python in demand for education jobs?",
     {"sector": "Education", "budget_usd": 600, "timeframe_weeks": 24}),
    ("Below benchmark - tight budget forces constraint handling",
     "SQL demand in retail",
     {"sector": "Retail", "budget_usd": 150, "timeframe_weeks": 12,
      "preferred_format": "online"}),
    ("Niche extension - offline benchmark",
     "HVAC roadmap for manufacturing",
     {"sector": "Manufacturing", "location": "Chicago"}),
    ("Escalation - niche skill, no local evidence",
     "HVAC roadmap for manufacturing",
     {"sector": "Manufacturing", "location": "Bangalore"}),
    ("Clarification - missing sector",
     "Is Python worth learning?", {}),
]


def main() -> None:
    pipe = ForecastAIPipeline()
    print(f"LLM: {pipe.llm.status}\n")
    for label, query, overrides in SCENARIOS:
        print("#" * 76)
        print(f"# {label}")
        print(f"# query: {query!r}  overrides: {overrides}")
        print("#" * 76)
        result = pipe.run(query, **overrides)
        if result.rewrite and result.rewrite.changed:
            print("Note: " + "; ".join(result.rewrite.notes) + "\n")
        print(result.response_text)
        print(f"\n[status: {result.status}]\n")


if __name__ == "__main__":
    main()
