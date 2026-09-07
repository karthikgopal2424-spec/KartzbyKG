# Implementation overview

*How SKILLKARTZ / ForecastAI is built — frameworks, tools, models, APIs and
libraries, and how each supports the design.*

---

## 1. Summary

ForecastAI is a **supervisor-led multi-agent system** written in **Python 3.9+**,
~3,100 lines of source across 21 modules. It answers "what share of job postings
in a sector/location mention a skill, and if it is below benchmark, what should I
learn?" and it does so **auditably**: every number is `count / total × 100` over
retrieved records, every claim traces to a row, and the system escalates to a
human rather than guess when the evidence is thin.

The guiding constraint drove almost every implementation choice:

> **The number-producing and decision-making paths must be deterministic,
> inspectable, and runnable offline.** A language model may polish the wording;
> it may never produce or alter a figure.

Consequently the **default build has zero third-party runtime dependencies** —
only the Python standard library. Frameworks (LangGraph, LangChain, the Anthropic
SDK) are wired in as *optional, swappable seams* behind feature flags, and a
parity test suite proves they do not change a single output.

---

## 2. Repository layout

```
skillkartz/
├── data/                         seeded synthetic corpus (JSON, ~570 KB)
│   ├── job_postings.json          ~900 postings; per-sector skill % known by construction
│   ├── skill_taxonomy.json        canonical skills, synonyms, complements, sectors, locations
│   ├── course_catalog.json        ~60 courses (a few deliberately "discontinued")
│   ├── local_signals.json         approved offline signals for niche/trade skills
│   └── generate_dataset.py        deterministic (seeded) corpus generator
├── src/skillkartz/
│   ├── pipeline.py                ForecastAIPipeline — public entry point + backend switch
│   ├── config.py                  every governed threshold / env toggle in one place
│   ├── models.py                  frozen dataclasses passed between agents (the data contract)
│   ├── datastore.py               corpus load, retrieval, dedupe, co-occurrence mining
│   ├── text.py                    dependency-free TF-IDF + cosine similarity (default index)
│   ├── llm.py                     optional Claude narration (guarded, offline fallback)
│   ├── cli.py                     `python -m skillkartz`
│   ├── graph_pipeline.py          optional LangGraph orchestration backend
│   ├── retrieval_langchain.py     optional LangChain Embeddings + VectorStore backend
│   └── agents/                    10 bots + the Supervisor
│       ├── supervisor.py                 hierarchical orchestrator + session memory
│       ├── query_rewriter.py             typo → taxonomy term (difflib)
│       ├── career_advisor.py             intent capture, validation, response rendering
│       ├── skill_intelligence.py         synonym / phrasing / fuzzy resolution
│       ├── job_market_research.py        retrieval, dedupe, coverage, freshness
│       ├── forecast_analytics.py         the % (chain-of-thought), trend, confidence
│       ├── upskilling.py                 Tree-of-Thought beam search for the roadmap
│       ├── governance_qa.py              reflection: re-verify, ground, contract, drift
│       ├── local_niche_intel.py          (conditional) approved offline signal collection
│       └── niche_governance.py           (conditional) niche classification + context benchmark
├── serve.py                       browser UI — stdlib-only HTTP server
├── tests/                         64 pytest tests, one file per component
├── examples/demo.py               seven end-to-end scenarios
└── docs/                          DESIGN_MAP.md, this file
```

---

## 3. Language and standard-library building blocks

The default pipeline uses **only the Python standard library**. The pieces doing
real work:

| Stdlib module | Where | Role in the design |
|---|---|---|
| `dataclasses` | `models.py`, throughout | The inter-agent **data contract**. Every message is a typed dataclass, so a full run serialises to plain JSON (`PipelineResult.to_dict()`) for the audit log. |
| `difflib.SequenceMatcher` | `agents/query_rewriter.py` | **Deterministic fuzzy string match** for typo correction, gated at a 0.82 similarity floor from `config.py`. No model needed; reproducible. |
| `re` | `agents/career_advisor.py`, `text.py` | Rule-based **intent / slot extraction** (skill, sector, budget `$800`, `24 weeks`) and tokenisation. |
| `math`, `collections.Counter` | `text.py` | **TF-IDF vectors + cosine similarity** — a dependency-free stand-in for the design's vector database. |
| `statistics.median` | `agents/niche_governance.py` | Context-benchmark computation for niche skills. |
| `datetime.date` | forecast trend, freshness, review dates | Trailing-window **trend detection** and data-freshness scoring. |
| `json`, `pathlib` | `datastore.py` | Loads the four JSON corpora into dataclasses. |
| `argparse` | `cli.py` | The `skillkartz` command and its flags (`--json`, `--trace`, `--backend`, `--retrieval`, `--llm`). |
| `http.server`, `urllib.parse`, `threading` | `serve.py` | The **browser UI** — a single-file HTTP server, no web framework, no build step. |
| `hashlib` | `retrieval_langchain.py` | The default offline embedding (signed token hashing). |

**Why:** the standard library keeps the artefact reproducible on any machine with
Python, makes `python3 -m pytest` a ~0.3 s no-install check, and guarantees the
arithmetic path has no hidden framework behaviour to reason about.

---

## 4. Frameworks and libraries

All third-party packages are **optional extras** in `pyproject.toml`. The default
install pulls none of them.

| Package | Version tested | Extra | Where used | How it supports the design |
|---|---|---|---|---|
| **pytest** | 8.4.2 | `[dev]` | `tests/` (64 tests) | Enforces the design's guarantees as executable checks: arithmetic re-verification, groundedness, the output contract, and **backend parity** (`test_graph_backend.py`, `test_retrieval_langchain.py`). |
| **LangGraph** | 0.6.11 | `[graph]` | `graph_pipeline.py` | Optional orchestration backend. The Supervisor's plan→branch→loop→escalate is already a state graph; LangGraph expresses it as a `StateGraph` with conditional edges, the Governance recalculation as a real **cycle**, an `Annotated[list, operator.add]` reducer for the trace, a `MemorySaver` checkpointer for per-thread state, and `draw_mermaid()` for a generated architecture diagram. Nodes delegate to the *same* agent objects, so behaviour is unchanged. |
| **langchain-core** | 0.3.86 | `[embeddings]` | `retrieval_langchain.py` | Optional retrieval backend. Provides the `Embeddings` interface, `InMemoryVectorStore`, and a real `VectorStoreRetriever`, so the semantic layer can be swapped for embeddings + a vector store without touching an agent. Default embedding (`HashingEmbeddings`) is deterministic and offline; a trained model drops in via `[embeddings-hf]`. |
| **numpy** | 2.0.2 | `[embeddings]` | (transitive) | Required by `langchain-core`'s cosine-similarity utility when the LangChain `VectorStoreRetriever` object is used directly. |
| **anthropic** | ≥ 0.40 (declared) | `[llm]` | `llm.py` | The official Anthropic SDK. One tool-less `messages.create` call to rephrase the finished, governed analysis. Guarded: absent SDK / key ⇒ deterministic template. |
| **langgraph-checkpoint**, **pydantic**, **orjson**, **tenacity**, … | — | (transitive of `[graph]`) | — | Pulled in by LangGraph; not used directly. |

**Deliberately *not* used:** no web framework (the UI is `http.server`), no ORM
(the corpus is read-only JSON), no ML framework in the default path, and — by
design — **no agent framework in the number-producing or decision-making path**.
`docs/DESIGN_MAP.md` records that LangChain / CrewAI / MCP were considered for the
Tree-of-Thought step and consciously left out to preserve auditability; LangGraph
and LangChain were then added back only as *alternative plumbing*.

---

## 5. Models

| Model | Where | Purpose | Notes |
|---|---|---|---|
| **`claude-sonnet-5`** (default; override with `SKILLKARTZ_MODEL`, e.g. `claude-opus-5`) | `llm.py`, invoked once from `agents/supervisor.py::_maybe_narrate` | **Narration only** — rephrase the completed analysis into 2–4 paragraphs. | System prompt forbids changing any number, course, provider, or the disclaimer. Output is `prose + "\n\n--- structured detail ---\n" + deterministic_template`. Runs only when `SKILLKARTZ_LLM=1` / `--llm` **and** the SDK and credentials are present; otherwise the identical content comes from a template. It is never in the retrieval, forecasting, Tree-of-Thought, or governance path. |
| **TF-IDF vector model** (bespoke, `text.py`) | default semantic index | Phrasing-variant matching and the semantic-drift surface. | Not a learned model; a classic weighted bag-of-words with cosine similarity. |
| **`HashingEmbeddings`** (bespoke, `retrieval_langchain.py`) | default embedding for the LangChain backend | Deterministic offline vectors via signed token hashing. | Swap for `sentence-transformers/all-MiniLM-L6-v2` via `pip install -e ".[embeddings-hf]"`. |

The design's premise is that a standalone LLM is insufficient for an auditable
labour-market answer; the model is therefore confined to a presentation role.

---

## 6. APIs

**Consumed**

- **Anthropic Messages API** — `client.messages.create(model, max_tokens=700,
  system, messages)` in `llm.py`. One call per answer, no tools, no streaming.
  All failures fall back to the deterministic template. Credentials via
  `ANTHROPIC_API_KEY` or `ant auth login`.

**Exposed** (`serve.py`, stdlib HTTP)

| Route | Returns |
|---|---|
| `GET /` | the single-page chat UI |
| `POST /` | no-JS fallback: runs the pipeline, renders the answer server-side |
| `GET` / `POST /api/query` | JSON result (same fields as the CLI flags) |
| `GET /api/resources` | corpus + index statistics, session run counters |
| `GET /api/memory` | the Supervisor's session memory |

**Public Python API** — `from skillkartz import ForecastAIPipeline`;
`pipe.run(text, **overrides) -> PipelineResult`; `pipe.record_feedback(...)`;
`pipe.memory`. Backends selected via `ForecastAIPipeline(backend=…, retrieval=…)`
or `SKILLKARTZ_BACKEND` / `SKILLKARTZ_RETRIEVAL`.

---

## 7. Techniques implemented in code (not external libraries)

These are "tools" in the design-document sense — capabilities the agents share
through the single `DataStore`, all deterministic Python:

| Technique | Location |
|---|---|
| Retrieval-for-computation (full matching set, no top-k) | `datastore.py::retrieve` |
| Metadata filtering + content-hash **deduplication** | `datastore.py::_dedupe` |
| **Semantic-drift** accounting (related-but-unlisted postings counted separately) | `datastore.py::retrieve` + `governance_qa.py` |
| **Co-occurrence mining** for roadmap candidate skills | `datastore.py::cooccurring_skills` |
| **Chain-of-thought** derivation traces | `forecast_analytics.py` (`reasoning_steps`) |
| **Confidence scoring** (weighted sample + coverage + freshness − drift penalty) | `forecast_analytics.py::_confidence` |
| Trailing-window **trend detection** | `forecast_analytics.py::_trend` |
| **Tree-of-Thought beam search** (beam 3, depth 4) with hard-constraint pruning, a weighted rubric, and one-shot constraint relaxation | `upskilling.py` |
| **Reflection / self-verification** (recompute, ground, contract, drift) | `governance_qa.py::review` |
| **Human-in-the-loop escalation** and one targeted **recalculation loop** | `supervisor.py` + `governance_qa.py` |
| **Session memory** + lightweight preference learning (accepted/rejected → provider ranking bias) | `supervisor.py` (`memory`, `record_feedback`) |

---

## 8. How the toolchain supports the design goals

| Design goal | Supporting choices |
|---|---|
| **Auditable, not generated** | Stdlib arithmetic only; `dataclasses` → JSON audit log; `pytest` re-verifies every `%`; the LLM SDK is fenced off from numbers. |
| **Deterministic** | No randomness in the default path; `difflib`/TF-IDF/beam search are pure functions; the seeded `generate_dataset.py` fixes the corpus; `HashingEmbeddings` is reproducible. |
| **Runs offline, anywhere** | Zero default dependencies; JSON corpus in-repo; `http.server` UI; LLM optional with template fallback. |
| **Fails loudly** | `governance_qa.py` blockers + escalation; `Roadmap(feasible=False)` instead of a fabricated plan; `niche_governance.py` flags first-time benchmarks. |
| **Extensible without risk** | Frameworks isolated behind `pyproject.toml` extras and `backend=` / `retrieval=` flags; parity tests guarantee identical results; the agent classes are reused unchanged by the LangGraph nodes. |

---

## 9. Build, test and run

```bash
# default (no third-party runtime deps)
python3 -m pytest -q                         # 48 core tests, ~0.3 s
python3 -m skillkartz "Is Python in demand for banking jobs in Chicago?"
python3 serve.py                             # browser UI at http://127.0.0.1:8000

# full install (adds optional backends; 64 tests)
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest -q

# optional framework backends
python3 -m skillkartz "React demand" --sector Technology --backend graph --retrieval langchain
```

**Dev tooling:** `pytest` (test runner, config in `pyproject.toml`), `git` +
GitHub (`gh` CLI) for version control, `setuptools` build backend, editable
install via `pip install -e`. Target runtime: CPython 3.9+ (developed on 3.9.6).
