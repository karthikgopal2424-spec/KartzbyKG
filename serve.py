"""Browser UI for ForecastAI / SKILLKARTZ - a chat-style front end.

A dependency-free HTTP wrapper around ``ForecastAIPipeline``. The left column is
a conversation (your questions + short replies); the full governed analysis is
published in the right column. Optional parameters sharpen the answer and each
one carries a short note explaining how.

    python3 serve.py                 # http://127.0.0.1:8000
    python3 serve.py --port 9000
    python3 serve.py --host 0.0.0.0  # expose on the LAN (be deliberate)

Routes
    GET  /              the app (single page)
    POST /              no-JS fallback: run the pipeline, render the answer
    GET  /api/query     JSON API (query params mirror the CLI flags)
    POST /api/query     JSON API (form-encoded or JSON body)  <- used by the page
                        response carries `meta` (timing + corpus/index stats +
                        chain-of-thought is in `forecast.reasoning_steps`)
    GET  /api/resources corpus + index stats and session run counters
    GET  /api/memory    the Supervisor's session memory (remembered results,
                        accepted/rejected feedback, provider ranking nudges)

It stays true to the project: fully offline, no API key, one pipeline instance
loading the seeded corpus once. Claude narration is left off.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from skillkartz import ForecastAIPipeline

SECTORS = [
    "", "Banking", "Education", "Healthcare", "Logistics",
    "Manufacturing", "Retail", "Technology",
]
FORMATS = ["", "online", "in-person", "hybrid"]

# One pipeline for the process; the corpus + TF-IDF index load once. The
# Supervisor keeps session memory, so serialise runs behind a lock.
_PIPELINE = ForecastAIPipeline()
_LOCK = threading.Lock()

# Fields accepted from the form / query string, mapped to UserQuery overrides.
_OVERRIDE_FIELDS = (
    "skill", "sector", "location", "experience",
    "budget_usd", "timeframe_weeks", "preferred_format",
)


def _collect_overrides(params: dict) -> dict:
    """params: {name: str}. Cast the numeric ones, drop blanks."""
    out: dict = {}
    for key in _OVERRIDE_FIELDS:
        val = (params.get(key) or "").strip()
        if not val:
            continue
        try:
            if key == "budget_usd":
                out[key] = float(val)
            elif key == "timeframe_weeks":
                out[key] = int(val)
            else:
                out[key] = val
        except ValueError:
            continue  # ignore an unparseable number rather than 500
    return out


_STATS = {"runs": 0, "total_ms": 0.0}


def _query_key(q) -> str:
    """Same shape the Supervisor uses to key its session memory."""
    return f"{(q.skill or '').lower()}|{(q.sector or '').lower()}|{(q.location or '').lower()}"


def _run(query_text: str, overrides: dict):
    """Run the pipeline under the lock; also return wall-time (ms) and whether
    this exact (skill, sector, location) was already in session memory before
    the run (i.e. the Governance drift check had a prior value to compare)."""
    with _LOCK:
        prior_keys = set(_PIPELINE.memory["results"])
        t0 = time.perf_counter()
        result = _PIPELINE.run(query_text, **overrides)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _STATS["runs"] += 1
        _STATS["total_ms"] += elapsed_ms
        seen_before = _query_key(result.query) in prior_keys
    return result, elapsed_ms, seen_before


def _corpus_stats() -> dict:
    """Static picture of the data + index the pipeline draws on (design doc
    section 7: one shared 'Tools & Data Infrastructure' layer)."""
    store = _PIPELINE.store
    postings = store.postings
    dates = sorted(p.posting_date for p in postings)
    idx = getattr(store, "_index", None)
    files = []
    for name in ("job_postings.json", "skill_taxonomy.json",
                 "course_catalog.json", "local_signals.json"):
        path = store.data_dir / name
        if path.exists():
            files.append({"file": name, "bytes": path.stat().st_size})
    skills = store.canonical_skills()
    return {
        "data_dir": str(store.data_dir),
        "corpus_as_of": store.as_of,
        "job_postings": len(postings),
        "distinct_sectors": len({p.sector for p in postings}),
        "distinct_sources": sorted({p.source for p in postings}),
        "distinct_employers": len({p.employer for p in postings}),
        "posting_date_span": [dates[0], dates[-1]] if dates else None,
        "canonical_skills": len(skills),
        "skill_synonyms": sum(len(m.get("synonyms", [])) for m in skills.values()),
        "locations": store.locations(),
        "courses": len(store.courses),
        "courses_discontinued": sum(1 for c in store.courses if not c.active),
        "local_signal_records": len(store.local_signals),
        "tfidf_documents": len(getattr(idx, "_doc_ids", []) or []),
        "tfidf_vocab_terms": len(getattr(idx, "_idf", {}) or {}),
        "data_files": files,
        "llm": _PIPELINE.llm.status,
        "python": sys.version.split()[0],
        "server_pid": os.getpid(),
    }


_CORPUS = _corpus_stats()


def _memory_snapshot(result=None, seen_before: bool = False) -> dict:
    """The Supervisor's session memory (design doc sections 5 & 7): short-term
    per-query results + the accepted/rejected feedback that nudges ranking."""
    mem = _PIPELINE.memory
    snap = {
        "remembered_results": dict(mem.get("results", {})),
        "accepted": len(mem.get("accepted", [])),
        "rejected": len(mem.get("rejected", [])),
        "provider_bias": dict(mem.get("provider_bias", {})),
    }
    if result is not None:
        key = _query_key(result.query)
        snap["current_query_key"] = key
        snap["this_query_seen_before"] = seen_before
        snap["drift_check"] = (
            "compared against the previous result for this exact query"
            if seen_before else
            "first time this exact query ran this session — nothing to compare yet"
        )
    return snap


def _meta(elapsed_ms: float, result=None, seen_before: bool = False) -> dict:
    runs = _STATS["runs"] or 1
    return {
        "elapsed_ms": round(elapsed_ms, 1),
        "run_number": _STATS["runs"],
        "session_runs": _STATS["runs"],
        "session_avg_ms": round(_STATS["total_ms"] / runs, 1),
        "corpus": _CORPUS,
        "memory": _memory_snapshot(result, seen_before),
    }


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForecastAI / SKILLKARTZ</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #eef1f4; --panel: #ffffff; --ink: #1c1e21; --muted: #667085;
    --line: #dfe1e5; --accent: #2563eb; --accent-ink: #ffffff;
    --bubble-user: #2563eb; --bubble-user-ink: #ffffff;
    --bubble-bot: #f1f3f5; --bubble-bot-ink: #1c1e21;
    --field: #ffffff; --code: #f5f6f8;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101215; --panel: #1a1d21; --ink: #e7e9ea;
      --muted: #97a1ac; --line: #2c3036; --accent: #3b82f6;
      --bubble-user: #2563eb; --bubble-user-ink: #ffffff;
      --bubble-bot: #24282e; --bubble-bot-ink: #e7e9ea;
      --field: #14171b; --code: #14171b;
    }
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  a { color: var(--accent); }
  code { background: rgba(127,127,127,.16); padding: 1px 5px; border-radius: 4px;
         font-size: .92em; }

  .app {
    display: grid;
    grid-template-columns: minmax(340px, 430px) minmax(0, 1fr);
    gap: 20px; width: 97vw; max-width: 1800px; margin: 0 auto; padding: 20px;
    height: 100vh;
  }
  .col { background: var(--panel); border: 1px solid var(--line);
         border-radius: 14px; display: flex; flex-direction: column;
         min-height: 0; overflow: hidden; }
  .col > header { padding: 14px 18px; border-bottom: 1px solid var(--line); }
  .col > header h1 { font-size: 16px; margin: 0; }
  .col > header p { margin: 3px 0 0; color: var(--muted); font-size: 12.5px; }

  /* ---- left: conversation ---- */
  #thread { flex: 1; overflow-y: auto; padding: 18px; display: flex;
            flex-direction: column; gap: 12px; }
  .msg { max-width: 88%; padding: 10px 13px; border-radius: 14px;
         white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }
  .msg.user { align-self: flex-end; background: var(--bubble-user);
              color: var(--bubble-user-ink); border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: var(--bubble-bot);
             color: var(--bubble-bot-ink); border-bottom-left-radius: 4px; }
  .msg.bot.thinking { color: var(--muted); font-style: italic; }
  .msg .chips { margin-top: 7px; display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { font-size: 11px; padding: 2px 7px; border-radius: 999px;
          background: rgba(255,255,255,.22); }
  .msg.bot .chip { background: rgba(127,127,127,.18); }
  .msg .ql { margin: 6px 0 0; padding-left: 18px; }
  .msg .actions { margin-top: 9px; display: flex; flex-direction: column;
                  align-items: flex-start; gap: 6px; }
  button.example {
    background: transparent; color: var(--accent); border: 1px solid var(--line);
    border-radius: 999px; padding: 5px 11px; font-size: 12.5px; font-weight: 500;
    text-align: left; white-space: normal; cursor: pointer;
  }
  button.example:hover { border-color: var(--accent);
                         background: rgba(37,99,235,.08); filter: none; }

  .composer { border-top: 1px solid var(--line); padding: 12px; }
  .composer .row { display: flex; gap: 8px; align-items: flex-end; }
  textarea#q {
    flex: 1; resize: none; min-height: 44px; max-height: 140px; padding: 11px 12px;
    border: 1px solid var(--line); border-radius: 10px; font: inherit;
    background: var(--field); color: inherit;
  }
  button {
    appearance: none; border: 0; border-radius: 10px; padding: 11px 18px;
    font: inherit; font-weight: 600; background: var(--accent);
    color: var(--accent-ink); cursor: pointer; white-space: nowrap;
  }
  button:hover { filter: brightness(1.05); }
  button:disabled { opacity: .55; cursor: default; }

  details.refine { margin-top: 10px; border: 1px solid var(--line);
                   border-radius: 10px; background: var(--field); }
  details.refine > summary { cursor: pointer; padding: 9px 12px; font-weight: 600;
                             font-size: 13px; list-style: none; }
  details.refine > summary::-webkit-details-marker { display: none; }
  details.refine > summary::before { content: "▸ "; color: var(--muted); }
  details.refine[open] > summary::before { content: "▾ "; }
  .refine .body { padding: 4px 12px 12px; }
  .disclaimer { font-size: 12px; color: var(--muted); margin: 2px 0 12px;
                padding: 9px 11px; border-left: 3px solid var(--accent);
                background: rgba(37,99,235,.06); border-radius: 0 8px 8px 0; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; }
  .grid .full { grid-column: 1 / -1; }
  .field label { display: block; font-size: 11.5px; font-weight: 600;
                 letter-spacing: .02em; margin-bottom: 3px; }
  .field input, .field select {
    width: 100%; padding: 7px 9px; border: 1px solid var(--line);
    border-radius: 8px; font: inherit; background: var(--panel); color: inherit;
  }
  .field .why { display: block; font-size: 11px; color: var(--muted);
                margin-top: 3px; }

  /* ---- right: published answer ---- */
  #right > header { background: rgba(37,99,235,.05); }
  #answer { flex: 1; overflow-y: auto; padding: 22px 26px; }
  #answer .badge { font-size: 12px; }
  pre.analysis { font-size: 13.5px; max-width: 78ch; }
  .empty { color: var(--muted); font-size: 14px; }
  .empty ul { padding-left: 18px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 999px;
           font-size: 11.5px; font-weight: 700; text-transform: uppercase;
           letter-spacing: .03em; }
  .badge.answered { background: #dcfce7; color: #166534; }
  .badge.clarification_needed { background: #fef9c3; color: #854d0e; }
  .badge.escalated { background: #fee2e2; color: #991b1b; }
  .rewrite { font-size: 12.5px; color: var(--muted); margin: 10px 0 0; }
  pre.analysis, pre.mono {
    background: var(--code); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;
    margin: 12px 0 0; font-size: 13px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  #answer details { margin-top: 12px; }
  #answer summary { cursor: pointer; font-weight: 600; font-size: 13px; }
  details.proc { border: 1px solid var(--line); border-radius: 10px;
                 padding: 10px 12px; background: var(--field); }
  details.proc h4 { margin: 12px 0 6px; font-size: 12px; text-transform: uppercase;
                    letter-spacing: .04em; color: var(--muted); }
  ol.cot { margin: 8px 0 2px; padding-left: 22px; font-size: 13px; }
  ol.cot li { padding: 3px 0; }
  ol.timeline { list-style: none; margin: 10px 0 2px; padding: 0; }
  ol.timeline li { display: flex; gap: 10px; padding: 7px 0;
                   border-top: 1px solid var(--line); font-size: 13px; }
  ol.timeline li:first-child { border-top: 0; }
  .tl-n { flex: none; width: 22px; height: 22px; border-radius: 50%;
          background: var(--accent); color: #fff; font-size: 12px;
          font-weight: 700; display: flex; align-items: center;
          justify-content: center; }
  .tl-agent { font-weight: 700; }
  .tl-detail { margin-top: 3px; font-size: 11.5px; color: var(--muted);
               font-family: ui-monospace, Menlo, Consolas, monospace;
               word-break: break-word; }
  table.kv { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.kv th { text-align: left; font-weight: 600; color: var(--muted);
                padding: 4px 10px 4px 0; vertical-align: top; white-space: nowrap; }
  table.kv td { padding: 4px 0; vertical-align: top; }
  table.kv tr + tr th, table.kv tr + tr td { border-top: 1px solid var(--line); }
  .foot { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line);
          font-size: 12px; color: var(--muted); }

  @media (max-width: 900px) {
    .app { grid-template-columns: 1fr; width: 100%; height: auto;
           min-height: 100vh; }
    .col { min-height: 58vh; }
    #thread { min-height: 300px; }
  }
</style>
</head>
<body>
<div class="app">

  <!-- ============ LEFT: conversation ============ -->
  <section class="col" id="left">
    <header>
      <h1>ForecastAI &mdash; skill-demand chat</h1>
      <p>Ask about a skill in a sector. Fully offline, seeded corpus, no API key.</p>
    </header>

    <div id="thread" aria-live="polite">
      <noscript><div class="msg bot">Ask about a skill's demand in a sector, e.g.
"Is Python in demand for banking jobs in Chicago?". The full analysis appears on the right.</div></noscript>
    </div>

    <form class="composer" id="form" method="post" action="/">
      <div class="row">
        <textarea id="q" name="query" placeholder="Ask about a skill's demand in a sector&hellip;" autocomplete="off"></textarea>
        <button type="submit" id="send">Send</button>
      </div>

      <details class="refine">
        <summary>Refine your question (optional)</summary>
        <div class="body">
          <p class="disclaimer">
            These are optional. ForecastAI reports the <strong>observed share of analysed
            job postings that mention the skill</strong> &mdash; not your personal chance of
            being hired. The fields below narrow what is counted and feed the
            upskilling roadmap; the more you set, the more specific (and, for
            small slices, the more caveated) the answer.
          </p>
          <div class="grid">
            <div class="field full">
              <label for="skill">Skill override</label>
              <input id="skill" name="skill" autocomplete="off">
              <span class="why">The exact capability measured. Set it to skip free-text parsing / typo-correction when the auto-detected skill is wrong.</span>
            </div>
            <div class="field">
              <label for="sector">Sector</label>
              <select id="sector" name="sector">__SECTOR_OPTIONS__</select>
              <span class="why">Which industry's postings are the denominator. This moves the number the most &mdash; e.g. Python ~15% in Banking vs ~46% in Technology.</span>
            </div>
            <div class="field">
              <label for="location">Location</label>
              <input id="location" name="location" autocomplete="off" placeholder="e.g. Chicago">
              <span class="why">Restricts postings to one place. Narrows the sample &mdash; can trigger the small-sample caveat or the niche-benchmark path.</span>
            </div>
            <div class="field">
              <label for="experience">Experience</label>
              <select id="experience" name="experience">
                <option value="">— any —</option>
                <option>Entry</option><option>Mid</option><option>Senior</option>
              </select>
              <span class="why">Context for the read. It does not change the core percentage today.</span>
            </div>
            <div class="field">
              <label for="preferred_format">Learning format</label>
              <select id="preferred_format" name="preferred_format">__FORMAT_OPTIONS__</select>
              <span class="why">online / in-person / hybrid. Biases course pick so the staged roadmap matches how you learn.</span>
            </div>
            <div class="field">
              <label for="budget_usd">Budget (USD)</label>
              <input id="budget_usd" name="budget_usd" type="number" min="0" step="1" placeholder="e.g. 800">
              <span class="why">Hard constraint on the roadmap. Too low and the planner reports "no roadmap fits" rather than inventing one.</span>
            </div>
            <div class="field">
              <label for="timeframe_weeks">Timeframe (weeks)</label>
              <input id="timeframe_weeks" name="timeframe_weeks" type="number" min="1" step="1" placeholder="e.g. 24">
              <span class="why">Hard cap on total course duration. Shorter timeframes prune longer course sequences.</span>
            </div>
          </div>
        </div>
      </details>
    </form>
  </section>

  <!-- ============ RIGHT: published answer ============ -->
  <section class="col" id="right">
    <header>
      <h1>Answer</h1>
      <p>The latest governed analysis, in full.</p>
    </header>
    <div id="answer">__RESULT__</div>
  </section>

</div>

<script>var KNOWN = __KNOWN__;</script>
<script>
(function () {
  "use strict";
  var form = document.getElementById("form");
  var qEl = document.getElementById("q");
  var sendBtn = document.getElementById("send");
  var thread = document.getElementById("thread");
  var answer = document.getElementById("answer");
  var STORE = "skillkartz-chat-v3";

  // Conversation context: the last resolved skill / sector / location, so a
  // short follow-up ("New York", "what about retail?") is read as a delta on
  // the previous turn rather than a fresh query with everything missing.
  var ctx = { skill: null, sector: null, location: null };
  function noSpace(s) { return String(s || "").toLowerCase().replace(/\s+/g, ""); }

  // difflib-style similarity (2*matched / total, multiset char overlap) so a
  // typo'd term still counts as "the user named this dimension".
  function qratio(a, b) {
    a = a.toLowerCase(); b = b.toLowerCase();
    if (!a.length || !b.length) return 0;
    var count = {}, i, m = 0;
    for (i = 0; i < a.length; i++) count[a[i]] = (count[a[i]] || 0) + 1;
    for (i = 0; i < b.length; i++) {
      if (count[b[i]] > 0) { count[b[i]]--; m++; }
    }
    return 2 * m / (a.length + b.length);
  }
  // does `text` name one of `terms` — exact substring, spaceless, or a word
  // that's ≥ 0.8 similar to a term (or a term's first word)?
  function namesAny(text, terms) {
    var lc = " " + text.toLowerCase() + " ", tn = noSpace(text);
    var words = text.toLowerCase().match(/[a-z][a-z0-9+.#-]{2,}/g) || [];
    return (terms || []).some(function (term) {
      var lo = String(term).toLowerCase();
      if (lc.indexOf(" " + lo + " ") >= 0 || lc.indexOf(lo + " ") >= 0) return true;
      if (lo.indexOf(" ") >= 0 && tn.indexOf(noSpace(lo)) >= 0) return true;
      var head = lo.split(" ")[0];
      return words.some(function (w) {
        return Math.abs(w.length - head.length) <= 2 && qratio(w, head) >= 0.8;
      });
    });
  }
  function carryContext(text) {
    var carry = {};
    var mSkill = namesAny(text, KNOWN.skills);
    var mSector = namesAny(text, KNOWN.sectors);
    var mLoc = namesAny(text, KNOWN.locations);
    if (mSkill && mSector) return carry;  // a whole new question — start clean
    if (ctx.skill && !mSkill) carry.skill = ctx.skill;
    if (ctx.sector && !mSector) carry.sector = ctx.sector;
    // carry location only on a pure refinement (no skill/sector named either)
    if (ctx.location && !mLoc && !mSkill && !mSector) carry.location = ctx.location;
    return carry;
  }
  function updateContext(d) {
    if (d && d.query) {
      if (d.query.skill) ctx.skill = d.query.skill;
      if (d.query.sector) ctx.sector = d.query.sector;
      if (d.query.location) ctx.location = d.query.location;
    }
  }

  var PARAM_FIELDS = ["skill", "sector", "location", "experience",
                      "preferred_format", "budget_usd", "timeframe_weeks"];
  var PARAM_LABEL = {
    skill: "skill", sector: "sector", location: "location",
    experience: "experience", preferred_format: "format",
    budget_usd: "budget $", timeframe_weeks: "weeks"
  };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function collectParams() {
    var p = {};
    PARAM_FIELDS.forEach(function (k) {
      var el = document.getElementById(k);
      if (el && el.value.trim()) p[k] = el.value.trim();
    });
    return p;
  }

  function addMsg(kind, text, opts) {
    opts = opts || {};
    var el = document.createElement("div");
    el.className = "msg " + kind + (opts.extra ? " " + opts.extra : "");
    el.textContent = text;
    if (opts.chips && opts.chips.length) {
      var wrap = document.createElement("div");
      wrap.className = "chips";
      opts.chips.forEach(function (c) {
        var s = document.createElement("span");
        s.className = "chip";
        s.textContent = c;
        wrap.appendChild(s);
      });
      el.appendChild(wrap);
    }
    if (opts.list && opts.list.length) {
      var ol = document.createElement("ol");
      ol.className = "ql";
      opts.list.forEach(function (q) {
        var li = document.createElement("li");
        li.textContent = q;
        ol.appendChild(li);
      });
      el.appendChild(ol);
    }
    if (opts.actions && opts.actions.length) {
      var act = document.createElement("div");
      act.className = "actions";
      opts.actions.forEach(function (a) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "example";
        b.textContent = a.label;
        b.addEventListener("click", function () {
          qEl.value = a.prompt;
          form.requestSubmit ? form.requestSubmit()
            : form.dispatchEvent(new Event("submit", { cancelable: true }));
        });
        act.appendChild(b);
      });
      el.appendChild(act);
    }
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
  }

  // --- conversational "meta" replies (offline, no pipeline call) --------
  var META_RE = new RegExp("^(hi|hey|hello|yo|help|menu|start|\\?+|" +
    "how (can|do) you help|what( can| do) you( do| help)|who are you|" +
    "what (is|are) (this|you)|how (does|do) (this|it|you) work|" +
    "what do you do|explain( yourself)?|how (to|do i) use|capabilities|" +
    "examples?|give (me )?an? example|options)\\b", "i");

  function botHelp() {
    addMsg("bot",
      "I'm ForecastAI. I read a seeded, offline corpus of ~900 job postings and " +
      "tell you what share of postings in a sector mention a skill — computed as " +
      "skill postings ÷ sector postings × 100, not a guess. If that share is below " +
      "the benchmark I also build a staged upskilling roadmap with a Tree-of-Thought " +
      "search.\n\n" +
      "Ask me in one line: a skill + a sector (a location is optional). " +
      "I'll ask for anything essential that's missing rather than guess.\n\n" +
      "Open “Refine your question” below to pin the skill/sector or add budget, " +
      "timeframe and format — those feed the roadmap. Every reply is published in " +
      "full on the right, with the agent steps it ran and the data it used.",
      { actions: [
        { label: "Is Python in demand for banking jobs in Chicago?",
          prompt: "Is Python in demand for banking jobs in Chicago?" },
        { label: "SQL demand in retail",
          prompt: "SQL demand in retail" },
        { label: "Machine Learning for healthcare",
          prompt: "How in demand is Machine Learning for healthcare jobs?" }
      ] });
  }

  function runChips(d) {
    var out = [];
    if (d.trace) out.push(d.trace.length + " agents");
    if (d.meta && d.meta.elapsed_ms != null) out.push(d.meta.elapsed_ms + " ms");
    if (d.retrieval) out.push((d.retrieval.sector_universe || []).length + " postings");
    return out;
  }

  // Say out loud which sector / location was inferred (typo-corrected or kept
  // from an earlier turn) so a wrong guess is easy to catch and correct.
  function assumptionNote(d, carry) {
    var q = d.query || {};
    var rw = (d.rewrite && d.rewrite.corrections) || [];
    function why(cat) {
      var c = rw.filter(function (x) { return x.category === cat; })[0];
      if (c) return 'read from "' + c.span + '"';
      if (carry && carry[cat]) return "kept from your earlier question";
      return null;
    }
    var parts = [];
    var ws = why("sector");
    if (q.sector && ws) parts.push("sector " + q.sector + " (" + ws + ")");
    var wl = why("location");
    if (q.location && wl) parts.push("location " + q.location + " (" + wl + ")");
    if (!parts.length) return "";
    return "Assuming " + parts.join(" and ") +
      " — reply with a different sector or location to change it.";
  }

  function headline(d) {
    try {
      var f = d.forecast;
      if (f) {
        var t = (f.sector_forecasts || []).filter(function (s) {
          return s.sector === f.target_sector;
        })[0];
        if (t) {
          return f.canonical_skill + " in " + f.target_sector + ": " +
            t.availability_pct.toFixed(1) + "%  (benchmark " +
            f.benchmark_pct.toFixed(0) + "%, " +
            (f.below_benchmark ? "below — roadmap suggested" : "at/above") + ")";
        }
      }
    } catch (e) {}
    return null;
  }

  function fmtBytes(n) {
    if (n == null) return "";
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  function kv(rows) {
    return '<table class="kv"><tbody>' + rows.map(function (r) {
      return "<tr><th>" + esc(r[0]) + "</th><td>" + r[1] + "</td></tr>";
    }).join("") + "</tbody></table>";
  }

  function renderProcess(d) {
    if (!d.trace || !d.trace.length) return "";
    var items = d.trace.map(function (t, i) {
      var detail = "";
      if (t.detail && Object.keys(t.detail).length) {
        var s = JSON.stringify(t.detail);
        if (s.length > 420) s = s.slice(0, 417) + "…";
        detail = '<div class="tl-detail">' + esc(s) + "</div>";
      }
      return '<li><span class="tl-n">' + (i + 1) + '</span>' +
        '<div><span class="tl-agent">' + esc(t.agent) + "</span> — " +
        esc(t.summary) + detail + "</div></li>";
    }).join("");
    return '<details class="proc" open><summary>Process — ' + d.trace.length +
      " agent steps, in order</summary><ol class=\"timeline\">" + items +
      "</ol></details>";
  }

  function renderCoT(d) {
    var f = d.forecast;
    if (!f || !(f.reasoning_steps || []).length) return "";
    var main = f.reasoning_steps.map(function (s, i) {
      return '<li>' + esc(s) + "</li>";
    }).join("");
    // the target sector's own derivation (count / total * 100, trend)
    var tgt = (f.sector_forecasts || []).filter(function (s) {
      return s.sector === f.target_sector;
    })[0];
    var sub = "";
    if (tgt && (tgt.reasoning_steps || []).length) {
      sub = '<h4>Target sector derivation</h4><ol class="cot">' +
        tgt.reasoning_steps.map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") +
        "</ol>";
    }
    return '<details class="proc" open><summary>Chain of thought — how the ' +
      'number was derived</summary><ol class="cot">' + main + "</ol>" + sub +
      '<p class="foot" style="margin-top:8px">Only the Upskilling Bot uses ' +
      'Tree-of-Thought search; every other step has one correct derivation and ' +
      'shows this step-by-step (chain-of-thought) working.</p></details>';
  }

  function renderMemory(d) {
    var mem = (d.meta || {}).memory;
    if (!mem) return "";
    var results = mem.remembered_results || {};
    var keys = Object.keys(results);
    var rows = [
      ["this query", esc(mem.current_query_key || "—")],
      ["seen before this session?", mem.this_query_seen_before ? "yes" : "no"],
      ["Governance drift check", esc(mem.drift_check || "—")],
      ["queries remembered", keys.length],
      ["accepted / rejected feedback", mem.accepted + " / " + mem.rejected]
    ];
    var pb = mem.provider_bias || {};
    var pbKeys = Object.keys(pb);
    if (pbKeys.length) {
      rows.push(["provider ranking nudges", pbKeys.map(function (k) {
        return esc(k) + " " + (pb[k] >= 0 ? "+" : "") + pb[k].toFixed(2);
      }).join(", ")]);
    }
    var recall = "";
    if (keys.length) {
      recall = '<h4>Remembered results (skill|sector|location → last %)</h4>' +
        '<ol class="cot">' + keys.map(function (k) {
          return "<li>" + esc(k) + " → " + Number(results[k]).toFixed(2) + "%</li>";
        }).join("") + "</ol>";
    }
    return '<details class="proc"><summary>Session memory</summary>' +
      kv(rows) + recall +
      '<p class="foot" style="margin-top:8px">The Supervisor keeps this for the ' +
      'life of the server process: re-asking the same query lets Governance ' +
      'compare the two results for drift, and accepted / rejected roadmaps nudge ' +
      'future course ranking.</p></details>';
  }

  function renderResources(d) {
    var m = d.meta || {};
    var c = m.corpus || {};
    var runRows = [
      ["pipeline time", (m.elapsed_ms != null ? m.elapsed_ms + " ms" : "—")],
      ["run # this session", (m.run_number != null ? m.run_number : "—")],
      ["session avg", (m.session_avg_ms != null ? m.session_avg_ms + " ms" : "—")],
      ["LLM narration", esc(c.llm || "—")]
    ];
    if (d.retrieval) {
      var r = d.retrieval;
      runRows.push(["postings analysed (sector universe)", (r.sector_universe || []).length]);
      runRows.push(["postings matched (skill)", (r.matched || []).length]);
      runRows.push(["duplicates removed", r.duplicates_removed]);
      runRows.push(["semantic-only, excluded from the count", r.semantic_only_candidates]);
      runRows.push(["sources used", esc((r.sources || []).join(", "))]);
      if (r.earliest_date) runRows.push(["posting date span", esc(r.earliest_date + " → " + r.latest_date)]);
    }
    var files = (c.data_files || []).map(function (f) {
      return esc(f.file) + " (" + fmtBytes(f.bytes) + ")";
    }).join("<br>");
    var corpusRows = [
      ["job postings indexed", c.job_postings],
      ["TF-IDF documents / vocab terms", (c.tfidf_documents + " / " + c.tfidf_vocab_terms)],
      ["sectors / locations", (c.distinct_sectors + " / " + (c.locations || []).length)],
      ["canonical skills / synonyms", (c.canonical_skills + " / " + c.skill_synonyms)],
      ["courses (discontinued)", (c.courses + " (" + c.courses_discontinued + ")")],
      ["local niche-signal records", c.local_signal_records],
      ["corpus as-of", esc(c.corpus_as_of || "—")],
      ["data files", files || "—"],
      ["runtime", "Python " + esc(c.python || "?") + " · pid " + esc(String(c.server_pid || "?"))]
    ];
    return '<details class="proc"><summary>Resources &amp; this run</summary>' +
      '<h4>This run</h4>' + kv(runRows) +
      '<h4>Corpus &amp; index (shared, loaded once)</h4>' + kv(corpusRows) +
      "</details>";
  }

  function renderAnswer(d) {
    if (d.error) {
      answer.innerHTML = '<span class="badge escalated">error</span>' +
        '<pre class="analysis">' + esc(d.error) + "</pre>";
      return;
    }
    var status = d.status || "answered";
    var html = '<span class="badge ' + esc(status) + '">' +
      esc(status.replace(/_/g, " ")) + "</span>";

    if (d.rewrite && d.rewrite.normalized_text !== d.rewrite.original_text) {
      html += '<p class="rewrite"><strong>Rewriter:</strong> ' +
        esc((d.rewrite.notes || []).join("; ")) + "</p>";
    }
    html += '<pre class="analysis">' + esc(d.response_text || "") + "</pre>";

    html += renderCoT(d);
    html += renderProcess(d);
    html += renderMemory(d);
    html += renderResources(d);
    html += "<details><summary>Full result JSON</summary><pre class='mono'>" +
      esc(JSON.stringify(d, null, 2)) + "</pre></details>";
    html += '<p class="foot">Every % is <code>skill postings ÷ sector postings × 100</code> ' +
      'on the retrieved rows. It is the observed share of analysed postings, not a hiring prediction.</p>';
    answer.innerHTML = html;
    answer.scrollTop = 0;
  }

  function persist(d) {
    try {
      sessionStorage.setItem(STORE, JSON.stringify({
        thread: thread.innerHTML, answer: answer.innerHTML
      }));
    } catch (e) {}
  }
  function restore() {
    try {
      var raw = sessionStorage.getItem(STORE);
      if (!raw) return;
      var s = JSON.parse(raw);
      if (s.thread) thread.innerHTML = s.thread;
      if (s.answer && s.answer.trim()) answer.innerHTML = s.answer;
      thread.scrollTop = thread.scrollHeight;
    } catch (e) {}
  }

  function ask(text) {
    var params = collectParams();
    // carry the previous turn's skill/sector/location unless this message,
    // or an explicit Refine field, overrides that dimension
    var carry = carryContext(text);
    Object.keys(carry).forEach(function (k) {
      if (!params[k]) params[k] = carry[k];
    });
    var explicit = collectParams();
    var chips = Object.keys(params).map(function (k) {
      var tag = PARAM_LABEL[k] + ": " + params[k];
      return (carry[k] && !explicit[k]) ? "↩ " + tag : tag;  // ↩ = from context
    });
    addMsg("user", text || "(using the parameters below)", { chips: chips });
    var thinking = addMsg("bot",
      "Running the pipeline:", { extra: "thinking", list: [
        "Query Rewriter — normalise the wording",
        "Career Advisor — capture intent, validate",
        "Skill Intelligence + Job Market Research — resolve & retrieve",
        "Forecast & Analytics — compute the share, trend, confidence",
        "Upskilling (if below benchmark) — Tree-of-Thought roadmap",
        "Governance & QA — re-check, then deliver"
      ] });
    sendBtn.disabled = true;

    var body = JSON.stringify(Object.assign({ query: text }, params));
    fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body
    }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      thinking.remove();
      var d = res.d;
      if (!res.ok || d.error) {
        addMsg("bot", "Something went wrong: " + (d.error || "request failed"));
        renderAnswer(d);
      } else if (d.status === "clarification_needed") {
        updateContext(d);
        addMsg("bot", "I need a bit more detail:", { list: d.clarification_questions || [] });
        renderAnswer(d);
      } else if (d.status === "escalated") {
        updateContext(d);
        addMsg("bot", "This one goes to a human reviewer — see the panel on the right.");
        renderAnswer(d);
      } else {
        updateContext(d);
        var note = assumptionNote(d, carry);
        addMsg("bot",
               (headline(d) || "Done — full analysis on the right.") +
               (note ? "\n\n" + note : ""),
               { chips: runChips(d) });
        renderAnswer(d);
      }
      persist();
    }).catch(function (e) {
      thinking.remove();
      addMsg("bot", "Network error: " + e);
    }).finally(function () {
      sendBtn.disabled = false;
      qEl.focus();
    });
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var text = qEl.value.trim();
    var params = collectParams();
    var hasParams = Object.keys(params).length > 0;
    if (!text && !hasParams) {
      qEl.focus();
      addMsg("bot", "Type a question, or open “Refine” and set at least a skill and sector.");
      return;
    }
    // Meta / greeting / "how can you help" — answer conversationally, no run.
    // (A short word count guards against matching a real query that happens
    //  to start with "help".)
    if (text && !params.skill && text.split(/\s+/).length <= 7 && META_RE.test(text)) {
      addMsg("user", text);
      qEl.value = "";
      qEl.style.height = "auto";
      botHelp();
      persist();
      return;
    }
    qEl.value = "";
    qEl.style.height = "auto";
    ask(text);
  });

  qEl.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
  qEl.addEventListener("input", function () {
    qEl.style.height = "auto";
    qEl.style.height = Math.min(qEl.scrollHeight, 140) + "px";
  });

  function showResourcesOnEmpty() {
    fetch("/api/resources").then(function (r) { return r.json(); })
      .then(function (j) {
        renderAnswer({
          status: "answered",
          response_text:
            "No analysis yet — ask a question on the left.\n\n" +
            "Below: the data + index every run draws on (loaded once at startup).",
          trace: [],
          meta: { corpus: j.corpus }
        });
        var proc = answer.querySelector("details.proc");
        if (proc) proc.open = true;  // keep Resources expanded on the empty state
        persist();
      }).catch(function () {});
  }

  restore();
  if (!sessionStorage.getItem(STORE)) {
    showResourcesOnEmpty();
    botHelp();  // greet with a self-description + clickable examples
    persist();
  }
  qEl.focus();
})();
</script>
</body>
</html>
"""

EMPTY_ANSWER = (
    '<div class="empty">'
    "<p>No analysis yet. Ask a question on the left and the full governed "
    "read is published here:</p><ul>"
    "<li>the observed job-posting share, benchmark, trend and confidence</li>"
    "<li>a staged Tree-of-Thought upskilling roadmap when demand is below benchmark</li>"
    "<li>the agent trace and the complete result JSON</li>"
    "</ul></div>"
)


def _options(values, selected):
    out = []
    for v in values:
        label = v or "— any —"
        sel = " selected" if v == selected else ""
        out.append(f'<option value="{html.escape(v)}"{sel}>{html.escape(label)}</option>')
    return "".join(out)


def _result_html(result, elapsed_ms: float | None = None,
                 seen_before: bool = False) -> str:
    """Server-rendered answer panel (used by the no-JS POST fallback)."""
    status = html.escape(result.status)
    body = html.escape(result.response_text)
    d = result.to_dict()
    d["meta"] = _meta(elapsed_ms if elapsed_ms is not None else 0.0,
                      result, seen_before)
    payload = html.escape(json.dumps(d, indent=2, default=str))

    proc = "".join(
        f'<li><span class="tl-n">{i}</span><div>'
        f'<span class="tl-agent">{html.escape(t.agent)}</span> — {html.escape(t.summary)}'
        f'</div></li>'
        for i, t in enumerate(result.trace, 1)
    )
    rewrite = ""
    if getattr(result, "rewrite", None) and result.rewrite.changed:
        rewrite = (f'<p class="rewrite"><strong>Rewriter:</strong> '
                   f'{html.escape("; ".join(result.rewrite.notes))}</p>')

    rr = result.retrieval
    run_rows = [("pipeline time", f"{d['meta']['elapsed_ms']} ms"),
                ("run # this session", d["meta"]["run_number"]),
                ("LLM narration", html.escape(_CORPUS["llm"]))]
    if rr is not None:
        run_rows += [
            ("postings analysed (sector universe)", len(rr.sector_universe)),
            ("postings matched (skill)", len(rr.matched)),
            ("duplicates removed", rr.duplicates_removed),
            ("semantic-only, excluded from count", rr.semantic_only_candidates),
            ("sources used", html.escape(", ".join(rr.sources))),
        ]
    c = _CORPUS
    corpus_rows = [
        ("job postings indexed", c["job_postings"]),
        ("TF-IDF documents / vocab terms",
         f'{c["tfidf_documents"]} / {c["tfidf_vocab_terms"]}'),
        ("sectors / locations",
         f'{c["distinct_sectors"]} / {len(c["locations"])}'),
        ("canonical skills / synonyms",
         f'{c["canonical_skills"]} / {c["skill_synonyms"]}'),
        ("courses (discontinued)",
         f'{c["courses"]} ({c["courses_discontinued"]})'),
        ("local niche-signal records", c["local_signal_records"]),
        ("corpus as-of", html.escape(c["corpus_as_of"])),
        ("data files", "<br>".join(
            f'{html.escape(f["file"])} ({f["bytes"] / 1024:.1f} KB)'
            for f in c["data_files"])),
        ("runtime", f'Python {html.escape(c["python"])} · pid {c["server_pid"]}'),
    ]

    def _kv(rows):
        return ('<table class="kv"><tbody>' + "".join(
            f'<tr><th>{html.escape(str(k))}</th><td>{v}</td></tr>'
            for k, v in rows) + "</tbody></table>")

    # --- chain of thought (forecast derivation) --------------------------
    cot = ""
    fc = result.forecast
    if fc is not None and fc.reasoning_steps:
        main = "".join(f"<li>{html.escape(s)}</li>" for s in fc.reasoning_steps)
        tgt = next((s for s in fc.sector_forecasts if s.sector == fc.target_sector), None)
        sub = ""
        if tgt is not None and tgt.reasoning_steps:
            sub = ('<h4>Target sector derivation</h4><ol class="cot">'
                   + "".join(f"<li>{html.escape(s)}</li>" for s in tgt.reasoning_steps)
                   + "</ol>")
        cot = (f'<details class="proc" open><summary>Chain of thought — how the '
               f'number was derived</summary><ol class="cot">{main}</ol>{sub}</details>')

    # --- session memory ------------------------------------------------
    mem = d["meta"]["memory"]
    mem_rows = [
        ("this query", html.escape(mem.get("current_query_key", "—"))),
        ("seen before this session?", "yes" if mem.get("this_query_seen_before") else "no"),
        ("Governance drift check", html.escape(mem.get("drift_check", "—"))),
        ("queries remembered", len(mem.get("remembered_results", {}))),
        ("accepted / rejected feedback", f'{mem["accepted"]} / {mem["rejected"]}'),
    ]
    recall = ""
    if mem.get("remembered_results"):
        recall = ('<h4>Remembered results (skill|sector|location → last %)</h4>'
                  '<ol class="cot">' + "".join(
                      f"<li>{html.escape(k)} → {v:.2f}%</li>"
                      for k, v in mem["remembered_results"].items())
                  + "</ol>")
    memory = (f'<details class="proc"><summary>Session memory</summary>'
              f'{_kv(mem_rows)}{recall}</details>')

    return (
        f'<span class="badge {status}">{status.replace("_", " ")}</span>'
        f'{rewrite}'
        f'<pre class="analysis">{body}</pre>'
        f'{cot}'
        f'<details class="proc" open><summary>Process — {len(result.trace)} '
        f'agent steps, in order</summary><ol class="timeline">{proc}</ol></details>'
        f'{memory}'
        f'<details class="proc"><summary>Resources &amp; this run</summary>'
        f'<h4>This run</h4>{_kv(run_rows)}'
        f'<h4>Corpus &amp; index (shared, loaded once)</h4>{_kv(corpus_rows)}</details>'
        f'<details><summary>Full result JSON</summary>'
        f'<pre class="mono">{payload}</pre></details>'
    )


def _render(result=None, note: str = "", elapsed_ms: float | None = None,
            seen_before: bool = False) -> bytes:
    if result is not None:
        panel = _result_html(result, elapsed_ms, seen_before)
    elif note:
        panel = f'<div class="empty"><p>{html.escape(note)}</p></div>'
    else:
        panel = EMPTY_ANSWER
    skills = _PIPELINE.store.canonical_skills()
    known = {
        "sectors": _PIPELINE.store.sectors(),
        "locations": _PIPELINE.store.locations(),
        "skills": sorted(set(list(skills) + [
            syn for meta in skills.values() for syn in meta.get("synonyms", [])
        ])),
    }
    page = (PAGE
            .replace("__SECTOR_OPTIONS__", _options(SECTORS, ""))
            .replace("__FORMAT_OPTIONS__", _options(FORMATS, ""))
            .replace("__KNOWN__", json.dumps(known))
            .replace("__RESULT__", panel))
    return page.encode("utf-8")


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "SkillkartzWeb/0.3"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_params(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype == "application/json" and raw:
            data = json.loads(raw)
            return {k: ("" if v is None else str(v)) for k, v in data.items()}
        return {k: v[0] for k, v in parse_qs(raw).items()}

    # -- GET ----------------------------------------------------------- #
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, _render(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/query":
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            self._api(params)
            return
        if parsed.path == "/api/resources":
            payload = json.dumps(
                {"corpus": _CORPUS, "session": _STATS}, indent=2, default=str
            ).encode()
            self._send(200, payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/memory":
            payload = json.dumps(
                {"memory": _memory_snapshot(), "session": _STATS},
                indent=2, default=str,
            ).encode()
            self._send(200, payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/healthz":
            self._send(200, b'{"ok": true}', "application/json")
            return
        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    do_HEAD = do_GET

    # -- POST -------------------------------------------------------- #
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            params = self._read_params()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, f"bad request: {exc}\n".encode(), "text/plain; charset=utf-8")
            return

        if parsed.path == "/":  # no-JS fallback
            has_input = bool((params.get("query") or "").strip()
                             or (params.get("skill") or "").strip())
            if not has_input:
                self._send(200, _render(note="Type a question, or set at least a skill and sector."),
                           "text/html; charset=utf-8")
                return
            try:
                result, elapsed, seen_before = _run(
                    params.get("query", ""), _collect_overrides(params))
            except Exception as exc:
                self._send(500, _render(note=f"Pipeline error: {exc}"),
                           "text/html; charset=utf-8")
                return
            self._send(200, _render(result, elapsed_ms=elapsed, seen_before=seen_before),
                       "text/html; charset=utf-8")
            return

        if parsed.path == "/api/query":
            self._api(params)
            return

        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    # -- shared JSON path ------------------------------------------ #
    def _api(self, params: dict) -> None:
        query_text = (params.get("query") or "").strip()
        if not query_text and not (params.get("skill") or "").strip():
            self._send(400, b'{"error": "query or skill is required"}',
                       "application/json")
            return
        try:
            result, elapsed, seen_before = _run(query_text, _collect_overrides(params))
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}).encode(),
                       "application/json")
            return
        d = result.to_dict()
        d["meta"] = _meta(elapsed, result, seen_before)
        payload = json.dumps(d, indent=2, default=str).encode()
        self._send(200, payload, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Browser UI for ForecastAI / SKILLKARTZ.")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    args = ap.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"ForecastAI web UI on {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
