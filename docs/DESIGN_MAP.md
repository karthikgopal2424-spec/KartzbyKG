# Design-document → code map

Every row ties a section of `SKILLKARTZ_AI_Agent_updated_6.docx` to the file that
implements it. Use this when grading the capstone against the design.

| Design doc section | Concept | Where it lives |
|---|---|---|
| 1. Problem & intended users | % of postings mentioning a skill; 35% threshold | `config.py` (`demand_threshold_pct`), `agents/forecast_analytics.py` |
| 2. Why a standalone LLM is insufficient | LLM does conversation only; numbers are deterministic | `llm.py` (optional, narration-only), all `agents/*` are pure Python |
| 3. Operating environment | job descriptions, taxonomy, course catalog, labour data, tools | `data/*.json`, `datastore.py` |
| 4. Supervisor-led multi-bot solution | 7 core bots + orchestration | `agents/supervisor.py` + the 6 specialist modules |
| 4. "calculates: skill % = skill postings ÷ total × 100" | the formula | `agents/forecast_analytics.py::forecast` |
| 4. "assigns a confidence score" | sample + coverage + freshness − drift | `agents/forecast_analytics.py::_confidence` |
| 5. Actions & feedback across steps | clarify → retrieve → calc → threshold → recommend → validate → present | `agents/supervisor.py::run` |
| 5. "correction only to the relevant specialist" | targeted recalculation, not full restart | `agents/supervisor.py` (single recalculation branch) |
| 5. session memory / accepted-rejected recs | ranking feedback | `agents/supervisor.py` (`self.memory`, `record_feedback`) |
| 7. Memory store (Supervisor) | short-term session context + long-term rec history | `agents/supervisor.py::memory` |
| 7. Chain-of-thought (Forecast output path) | step-by-step calculation trace | `SectorForecast.reasoning_steps`, `ForecastResult.reasoning_steps` |
| 7. Reflection (Governance) | self-critique before the answer ships | `agents/governance_qa.py::review` |
| 7. Tools & data infrastructure (shared) | one index every bot draws on | `datastore.py` (`TfidfIndex`, single `DataStore`) |
| 8. Local & Niche Skill Intelligence Agent | approved offline sources, dedupe, reliability score | `agents/local_niche_intel.py`, `data/local_signals.json` |
| 8. Niche Skill Governance & Benchmarking Agent | niche classification + context benchmark | `agents/niche_governance.py` |
| 8. Integration workflow (9 steps) | trigger → classify → collect → normalise → benchmark → forecast → QA | `agents/supervisor.py::run` (the `insufficient_data` branch) |
| 8. "Every benchmark must include period, sources, confidence, review date" | `NicheAssessment` fields | `models.py::NicheAssessment` |
| 9. Retrieval is required | grounding every number in current postings | `datastore.py::retrieve` |
| 9. "retrieval-for-computation, not top-k" | full matching set returned | `datastore.py::retrieve` (no `k` cap) |
| 9. Semantic resolution ("ML" ↔ "Machine Learning") | synonym / phrasing expansion | `agents/skill_intelligence.py` |
| 9. Chunking: one posting = one chunk | index construction | `datastore.py::_load` |
| 9. Semantic-drift failure mode + hybrid mitigation | exact-match counting + drift surface recorded + Governance check | `datastore.py::retrieve` (`semantic_only_candidates`), `agents/governance_qa.py` (`semantic-drift` check) |
| 10. Tree-of-Thought — only in the Upskilling Bot | beam search over partial roadmaps | `agents/upskilling.py` |
| 10.2 thought / node / branch / depth | `_Node`, branching, `tot_max_depth` | `agents/upskilling.py`, `config.py` |
| 10.3 beam search + rubric + hard-constraint prune + one relaxation | `_beam_search`, `_score_step` | `agents/upskilling.py` |
| 10.3 "reports that no roadmap fits rather than fabricating" | infeasible result | `agents/upskilling.py` (`Roadmap.feasible = False`) |
| 10.4 ToT roles → tools (LangChain / CrewAI / MCP) | default build is dependency-free; optional LangGraph + LangChain backends are wired as drop-in seams | `graph_pipeline.py` (LangGraph `StateGraph`), `retrieval_langchain.py` (LangChain `Embeddings` + `VectorStore`), selected via `ForecastAIPipeline(backend=…, retrieval=…)` |
| 11. Multi-agent architecture — 7 core + 2 conditional | agent roster | `agents/__init__.py`, `agents/supervisor.py` |
| 11. Hybrid coordination (star + fan-out + conditional branch + feedback loop) | orchestration shape | `agents/supervisor.py::run` |
| 12. Safety & Intervention Plan — risk table | per-stage risks | mirrored by the checks in `agents/governance_qa.py` |
| 12. Guardrails: input checks | typo-tolerant normalisation *then* clarification instead of best guess | `agents/query_rewriter.py` (deterministic difflib snap to taxonomy terms), `agents/career_advisor.py::clarification_questions` |
| 12. Guardrails: output constraints | disclaimer + period + coverage + confidence, always | `agents/career_advisor.py::render` + `governance_qa.py` output-contract check |
| 12. Guardrails: source verification | reliability floor, licensing flag | `agents/local_niche_intel.py`, `config.py` |
| 12. Guardrails: tool access limits (least privilege) | read-only research bots, no cross-writes | documented per agent; enforced by module boundaries |
| 12. Guardrails: escalation rules | confidence floor → recalculation → human | `agents/supervisor.py` + `agents/governance_qa.py` |
| 12. Guardrails: runtime monitoring / audit log | full serialisable trace | `models.py::PipelineResult.to_dict`, `--json` / `--trace` |
| 12. Evaluation metrics (correctness, groundedness, calibration, safety, …) | the checks + the test suite | `agents/governance_qa.py`, `tests/` |
| 12. Conditions requiring human intervention | ambiguity, low confidence, first-time benchmark, policy sensitivity, conflicting evidence, user dispute | `agents/governance_qa.py` (escalation branches), `agents/niche_governance.py` (first-time sign-off flag) |
