"""Command-line interface for ForecastAI / SKILLKARTZ.

    python -m skillkartz "Is Python in demand for banking jobs in Chicago?"
    python -m skillkartz "welding roadmap" --sector Manufacturing --location Chicago
    python -m skillkartz "React demand" --sector Technology --json
    python -m skillkartz "ML in banking" --trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .llm import LLMClient
from .pipeline import ForecastAIPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillkartz",
        description="Supervisor-led multi-bot career forecasting and upskilling agent.",
    )
    p.add_argument("query", nargs="?", default="",
                   help="free-text question, e.g. 'Is SQL in demand for retail jobs?'")
    p.add_argument("--skill", help="pin the target skill (overrides parsing)")
    p.add_argument("--sector", help="pin the target sector")
    p.add_argument("--location", help="pin the location filter")
    p.add_argument("--experience", help="Entry | Mid | Senior")
    p.add_argument("--budget", type=float, dest="budget_usd",
                   help="learning budget in USD (for the roadmap)")
    p.add_argument("--timeframe-weeks", type=int, dest="timeframe_weeks",
                   help="available time for upskilling, in weeks")
    p.add_argument("--format", dest="preferred_format",
                   help="online | in-person | hybrid")
    p.add_argument("--data-dir", type=Path, help="override the data directory")
    p.add_argument("--json", action="store_true", help="emit the full result as JSON")
    p.add_argument("--trace", action="store_true", help="print the agent trace")
    p.add_argument("--llm", action="store_true",
                   help="enable Claude narration (needs ANTHROPIC_API_KEY)")
    p.add_argument("--version", action="version", version=f"skillkartz {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.query and not args.skill:
        build_parser().print_help()
        return 2

    llm = LLMClient(enabled=True) if args.llm else None
    pipe = ForecastAIPipeline(data_dir=args.data_dir, llm=llm)

    overrides = {
        k: getattr(args, k)
        for k in ("skill", "sector", "location", "experience",
                  "budget_usd", "timeframe_weeks", "preferred_format")
        if getattr(args, k) is not None
    }
    result = pipe.run(args.query, **overrides)

    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if result.rewrite and result.rewrite.changed:
        print("Note: " + "; ".join(result.rewrite.notes) + "\n")

    print(result.response_text)

    if args.trace:
        print("\n" + "=" * 68)
        print("AGENT TRACE")
        print("=" * 68)
        for i, t in enumerate(result.trace, 1):
            print(f"{i:>2}. [{t.agent}] {t.summary}")
            for key, val in t.detail.items():
                text = json.dumps(val, default=str)
                if len(text) > 300:
                    text = text[:297] + "..."
                print(f"      {key}: {text}")

    return {"answered": 0, "clarification_needed": 3, "escalated": 4}.get(result.status, 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
