# SKILLKARTZ / ForecastAI — capstone implementation

A runnable Python implementation of the system specified in
`SKILLKARTZ_AI_Agent_updated_6.docx`: a **supervisor-led multi-bot agent** that
tells a user what share of job postings in a chosen sector and location mention a
target skill, and — when that share is below benchmark — builds a staged
upskilling roadmap using **Tree-of-Thought** search.

It runs **fully offline** on a seeded synthetic corpus. No API key is required.
An optional Claude integration only rephrases the final answer; it never produces
a number.

---

## Use case

> *"What percentage of banking jobs in Chicago ask for Python, and if it's low,
> what should I learn?"*

Job portals list vacancies; labour reports give broad statistics. Neither gives a
personalised, current, **auditable** answer to that question. ForecastAI does:

1. **Captures intent** — skill, sector, location, experience, budget, timeframe,
   learning-format preference — and asks a clarifying question rather than
   guessing when something essential is missing.
2. **Retrieves** the full set of matching postings (semantic + metadata filter),
   removes duplicates, records source coverage and freshness.
3. **Calculates** the demand percentage *step by step* —
   `skill postings ÷ total sector postings × 100` — ranks sectors, detects the
   demand trend, and attaches a confidence score.
4. **Branches to a niche path** when standard job-board data is too thin:
   pulls approved offline signals, classifies the skill (niche vs.
   insufficient-evidence) and sets a context-based benchmark to use instead of
   the flat 35%.
5. **Builds a roadmap** with Tree-of-Thought beam search when demand is below
   benchmark — comparing candidate `(skill, course)` sequences against budget,
   timeframe, prerequisite order and forecasted demand — or reports that *no
   roadmap fits* rather than inventing one.
6. **Governs** every answer: re-verifies the arithmetic, checks that every
   claim traces to a record, enforces the disclaimer/coverage/confidence
   contract, watches for drift, and **escalates to a human** when confidence
   stays below the floor or the evidence is too weak.

---

## Quickstart

```bash
cd skillkartz
python3 -m pytest -q                       # 48 tests core, ~0.3s, no deps beyond pytest
python3 examples/demo.py                    # seven scenarios, one per pipeline path
```

(`pip install -e ".[dev]"` adds the optional-backend packages and runs 64 tests.)

Run a single query (no install needed — just set the path):

```bash
PYTHONPATH=src python3 -m skillkartz "Is Python in demand for banking jobs in Chicago?"
PYTHONPATH=src python3 -m skillkartz --skill SQL --sector Retail --budget 800 --timeframe-weeks 24 --format online
PYTHONPATH=src python3 -m skillkartz --skill HVAC --sector Manufacturing --location Chicago --trace
PYTHONPATH=src python3 -m skillkartz "React demand" --sector Technology --json
```

Or install it and use the `skillkartz` command / the library (needs a modern
`pip` — use a virtualenv if your system `pip` is old):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
skillkartz "How in demand is Machine Learning for banking?"
```

```python
from skillkartz import ForecastAIPipeline

pipe = ForecastAIPipeline()
result = pipe.run("Is Python in demand for education jobs?", sector="Education")
print(result.response_text)
print(result.forecast.confidence, result.roadmap.steps)
```

### Optional: Claude narration

```bash
pip install -e ".[llm]"
export SKILLKARTZ_LLM=1
export ANTHROPIC_API_KEY=sk-ant-...          # or run `ant auth login`
# optional: export SKILLKARTZ_MODEL=claude-opus-5   (default: claude-sonnet-5)
skillkartz "SQL demand in retail" --llm
```

The LLM is handed the finished, governed analysis and asked to rephrase it
without changing any number, course, provider or the disclaimer. With the flag
off (the default) the same content is emitted from a deterministic template.

---

## Architecture

```
                         ┌──────────────────────────┐
      free-text query ──▶ │  Query Rewriter Bot      │  typo → taxonomy term (difflib)
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │  Career Advisor Bot      │  intent capture + validation
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │  Supervisor / Orchestrator│  plan · delegate · session memory
                         └──┬───────────────┬────────┘
              parallel fan-out              │
        ┌────────────────┐  ┌───────────────────────┐
        │ Skill          │  │ Job Market Research   │  retrieve · dedupe · coverage
        │ Intelligence   │  │  (shared TF-IDF index)│
        └───────┬────────┘  └───────────┬───────────┘
                └──────────┬────────────┘
                           ▼                        insufficient data?
                ┌──────────────────────┐   ┌─────────────────────────────────┐
                │ Forecast & Analytics │   │ (conditional) Local & Niche      │
                │  % = a/b·100  (CoT)  │◀──│  Skill Intelligence  +           │
                │  trend · confidence  │   │  Niche Governance & Benchmarking │
                └──────────┬───────────┘   └─────────────────────────────────┘
             below benchmark │
                             ▼
                ┌──────────────────────┐
                │ Upskilling Bot       │  Tree-of-Thought beam search
                │  (beam=3, depth=4)   │  hard-constraint prune + rubric score
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐   reflection: arithmetic · groundedness ·
                │ Governance & QA Bot  │   output contract · drift · confidence floor
                └──────────┬───────────┘   → pass · recalculate once · escalate
                           ▼
                ┌──────────────────────┐
                │ Career Advisor Bot   │  explainable response (or escalation notice)
                └──────────────────────┘
```

**8 core agents** (always run) + **2 conditional agents** (only on thin data).
The Query Rewriter Bot is the newest core agent: it runs first and snaps mistyped
skill / sector / location tokens onto the nearest taxonomy term (deterministic
`difflib` character similarity, floor 0.82) so a small typo reaches an answer
instead of a clarification turn, while anything genuinely ambiguous is still
handed to the "ask rather than guess" guardrail.
Coordination is deliberately hybrid: a hierarchical Supervisor backbone, a
parallel fan-out for the two independent retrieval bots, a conditional graph
branch for the niche pair, and one targeted feedback loop from Governance.

See [`docs/DESIGN_MAP.md`](docs/DESIGN_MAP.md) for a section-by-section mapping
back to the design document.

---

## Pluggable backends (optional)

The default build is dependency-free and deterministic. Two seams can be swapped
for their framework equivalents without touching a single agent — same
`PipelineResult`, same numbers:

| Seam | Default | Framework option | Install |
|---|---|---|---|
| Orchestration | hand-rolled `SupervisorBot` | **LangGraph** `StateGraph` (`graph_pipeline.py`) — conditional edges, the Governance recalculation cycle as a real loop, a checkpointer, a `draw_mermaid()` diagram | `pip install -e ".[graph]"` |
| Semantic retrieval | TF-IDF in `text.py` | **LangChain** `Embeddings` + `InMemoryVectorStore` + `VectorStoreRetriever` (`retrieval_langchain.py`); default embedding is a deterministic offline token-hash, swappable for a real model via `[embeddings-hf]` | `pip install -e ".[embeddings]"` |

```bash
python3 -m skillkartz "Is Python in demand for banking jobs?" --backend graph
python3 -m skillkartz "React demand" --sector Technology --retrieval langchain
# or: export SKILLKARTZ_BACKEND=graph  SKILLKARTZ_RETRIEVAL=langchain
```

```python
ForecastAIPipeline(backend="graph", retrieval="langchain")
```

The point of the exercise: the number-producing and decision-making paths stay
pure Python and deterministic; only the plumbing changes. Parity is enforced by
`tests/test_graph_backend.py` and `tests/test_retrieval_langchain.py`.

---

## Project layout

```
skillkartz/
├── data/
│   ├── generate_dataset.py     seeded synthetic-corpus generator
│   ├── job_postings.json        ~900 postings, per-sector skill % known by construction
│   ├── skill_taxonomy.json      canonical skills, synonyms, phrasing variants, complements
│   ├── course_catalog.json      courses (a few deliberately "discontinued")
│   └── local_signals.json       approved offline signals for niche/trade skills
├── src/skillkartz/
│   ├── config.py               every governed threshold in one place
│   ├── models.py               typed dataclasses passed between agents
│   ├── text.py                 dependency-free TF-IDF + cosine similarity
│   ├── graph_pipeline.py       optional LangGraph orchestration backend
│   ├── retrieval_langchain.py  optional LangChain Embeddings + VectorStore backend
│   ├── datastore.py            corpus loading, retrieval, dedupe, co-occurrence mining
│   ├── llm.py                  optional Claude narration (guarded, offline fallback)
│   ├── pipeline.py             ForecastAIPipeline — the public entry point
│   ├── cli.py                  `python -m skillkartz`
│   └── agents/                 the ten bots + the supervisor
├── tests/                      48 pytest tests, one file per component
├── examples/demo.py            seven end-to-end scenarios
└── docs/DESIGN_MAP.md          design-doc → code cross-reference
```

---

## How the design doc's guarantees show up in code

| Guarantee | Mechanism |
|---|---|
| Numbers are auditable, not generated | every `%` is `count/total*100` on retrieved rows; `ForecastResult.reasoning_steps` shows the working; `--json` dumps the whole run |
| Retrieval returns the full matching set | `DataStore.retrieve` has no top-k cap (design doc calls this "retrieval-for-computation") |
| Semantic drift is contained | counting is exact-match on the canonical skill; related-but-unlisted postings are counted separately as a drift surface and flagged by Governance |
| A percentage never ships without its confidence | `CareerAdvisorBot.render` template + Governance output-contract check (a blocker if violated) |
| Small samples are labelled | `< 5` matching or `< 8` total postings → mandatory caveat, enforced by Governance |
| Tree-of-Thought only where it belongs | beam search is inside `UpskillingBot` only; every other step has one correct derivation and uses chain-of-thought |
| The system fails loudly | low confidence → one recalculation → human escalation; thin niche evidence → escalation; discontinued/absent course → blocker |
| First-time niche benchmarks need sign-off | `NicheSkillGovernanceBot` flags any benchmark with no history for human review |

---

## Known limitations (it is a capstone demonstrator)

- **Synthetic data.** The corpus is generated, not scraped. Percentages are
  realistic *by construction* so the arithmetic is checkable against ground
  truth — they are not real labour-market figures.
- **Toy uplift model.** The roadmap's "projected % after completion" uses a flat
  `complementary-skill sector-share × 0.30` damping factor. It is enough to drive
  the Tree-of-Thought search and demonstrate constraint handling, pruning and
  determinism; it is not a validated outcome model, so a roadmap may suggest a
  broadly useful complementary skill (e.g. Excel) rather than a domain-perfect
  one.
- **TF-IDF, not embeddings.** `text.py` stands in for the design's vector
  database. It demonstrates the *behaviour* (phrasing variants, drift risk)
  without a model dependency.
- **Least privilege is by module boundary + documentation**, not an enforced
  capability system.
- **The LLM path is intentionally minimal** — narration only, single call, no
  tools.
