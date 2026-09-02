"""Upskilling Bot - staged roadmap via Tree-of-Thought beam search.

This is the one place in the system where Tree-of-Thought is justified (design doc
section 10): roadmap generation is a combinatorial planning problem with several
valid complementary-skill orderings, hard budget/timeframe constraints, and a
real risk of committing early to a mediocre plan.

Structure (matches design doc section 10.2-10.3):
  * thought  = a partial roadmap: a sequence of (skill, course) steps + remaining gap
  * branch   = an alternative next step (different skill, or different course)
  * depth    = roadmap stage (capped at SETTINGS.tot_max_depth)
  * search   = beam search, width SETTINGS.tot_beam_width
  * evaluate = deterministic hard-constraint prune, then a weighted rubric score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import SETTINGS
from ..datastore import Course, DataStore
from ..models import Roadmap, RoadmapStep


@dataclass
class _Node:
    steps: list[RoadmapStep] = field(default_factory=list)
    skills_used: set = field(default_factory=set)
    cost: float = 0.0
    weeks: int = 0
    projected_pct: float = 0.0
    score_sum: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.depth if self.steps else 0.0


class UpskillingBot:
    name = "Upskilling Bot"

    def __init__(self, store: DataStore, provider_bias: Optional[dict] = None) -> None:
        self.store = store
        self.provider_bias = provider_bias or {}

    # ------------------------------------------------------------------ #
    def build_roadmap(
        self,
        target_skill: str,
        target_sector: str,
        current_pct: float,
        benchmark_pct: float,
        budget_usd: Optional[float],
        timeframe_weeks: Optional[int],
        preferred_format: Optional[str],
    ) -> Roadmap:
        gap = round(benchmark_pct - current_pct, 2)
        notes: list[str] = []
        candidates = self._candidate_skills(target_skill, target_sector)
        if not candidates:
            return Roadmap(False, [], 0.0, 0, current_pct, gap,
                           ["no complementary skills available for this sector"],
                           {"candidate_skills": 0})

        result = self._beam_search(
            candidates, target_sector, current_pct, benchmark_pct,
            budget_usd, timeframe_weeks, preferred_format,
        )

        # one-shot constraint relaxation if nothing feasible (design doc s.10.3)
        if result is None and timeframe_weeks is not None:
            relaxed = int(timeframe_weeks * 1.5)
            notes.append(
                f"no roadmap fit {timeframe_weeks} weeks; retried with a relaxed "
                f"{relaxed}-week window"
            )
            result = self._beam_search(
                candidates, target_sector, current_pct, benchmark_pct,
                budget_usd, relaxed, preferred_format,
            )

        if result is None:
            notes.append(
                "no combination of the available courses stays within the stated "
                "budget/timeframe - reporting infeasible rather than fabricating a plan"
            )
            return Roadmap(False, [], 0.0, 0, current_pct, gap, notes,
                           {"candidate_skills": len(candidates)})

        best, stats = result
        if preferred_format:
            off = [s.skill for s in best.steps if s.format != preferred_format]
            if off:
                notes.append(
                    f"format preference '{preferred_format}' not met for: "
                    f"{', '.join(off)} (format is a soft preference, not a hard "
                    f"constraint - design doc s.10.3)"
                )
        if best.projected_pct < benchmark_pct:
            notes.append(
                f"roadmap closes most of the gap but projected share "
                f"{best.projected_pct:.1f}% is still under the "
                f"{benchmark_pct:.0f}% benchmark within the depth limit"
            )
        return Roadmap(
            feasible=True,
            steps=best.steps,
            total_cost_usd=round(best.cost, 2),
            total_weeks=best.weeks,
            projected_final_pct=round(best.projected_pct, 2),
            demand_gap_pp=gap,
            notes=notes,
            search_stats=stats,
        )

    # ------------------------------------------------------------------ #
    def _candidate_skills(self, target_skill: str, sector: str) -> list[str]:
        meta = self.store.canonical_skills().get(target_skill, {})
        complementary = [
            s for s in meta.get("complementary", [])
            if s in self.store.canonical_skills()
        ]
        cooc = self.store.cooccurring_skills(target_skill, sector, top_n=10)

        def _usable(seq):
            out = []
            for s in seq:
                if s != target_skill and s not in out and self.store.courses_for(s):
                    out.append(s)
            return out

        # Curated complementary skills lead (they are the domain-sensible next
        # steps); co-occurrence-mined skills follow. Within each group, rank by
        # demand relevance in the target sector (design doc s.10.3).
        curated = sorted(_usable(complementary),
                         key=lambda s: (-self.store.sector_skill_pct(s, sector), s))
        mined = sorted((s for s in _usable(cooc) if s not in curated),
                       key=lambda s: (-self.store.sector_skill_pct(s, sector), s))
        return curated + mined

    def _candidate_courses(self, skill: str) -> list[Course]:
        courses = sorted(
            self.store.courses_for(skill, active_only=True),
            key=lambda c: (-c.rating, c.id),
        )
        return courses[:2]

    # ------------------------------------------------------------------ #
    def _beam_search(
        self,
        candidates: list[str],
        sector: str,
        current_pct: float,
        benchmark_pct: float,
        budget: Optional[float],
        timeframe: Optional[int],
        preferred_format: Optional[str],
    ) -> Optional[tuple[_Node, dict]]:
        beam: list[_Node] = [_Node(projected_pct=current_pct)]
        completed: list[_Node] = []
        nodes_expanded = 0
        nodes_pruned = 0

        for _ in range(SETTINGS.tot_max_depth):
            next_beam: list[_Node] = []
            for node in beam:
                if node.projected_pct >= benchmark_pct and node.steps:
                    completed.append(node)
                    continue
                expansions = 0
                for skill in candidates:
                    if skill in node.skills_used:
                        continue
                    if expansions >= SETTINGS.tot_branching_factor:
                        break
                    for course in self._candidate_courses(skill):
                        new_cost = node.cost + course.cost_usd
                        new_weeks = node.weeks + course.duration_weeks
                        if budget is not None and new_cost > budget:
                            nodes_pruned += 1
                            continue
                        if timeframe is not None and new_weeks > timeframe:
                            nodes_pruned += 1
                            continue
                        step_score, uplift = self._score_step(
                            skill, course, sector, node, budget, timeframe,
                            benchmark_pct, preferred_format,
                        )
                        child = _Node(
                            steps=node.steps + [RoadmapStep(
                                stage=node.depth + 1,
                                skill=skill,
                                course_id=course.id,
                                course_title=course.title,
                                provider=course.provider,
                                cost_usd=course.cost_usd,
                                duration_weeks=course.duration_weeks,
                                format=course.format,
                                projected_uplift_pp=round(uplift, 2),
                                score=round(step_score, 3),
                            )],
                            skills_used=node.skills_used | {skill},
                            cost=new_cost,
                            weeks=new_weeks,
                            projected_pct=round(node.projected_pct + uplift, 2),
                            score_sum=node.score_sum + step_score,
                        )
                        next_beam.append(child)
                        nodes_expanded += 1
                    expansions += 1

            if not next_beam:
                break
            next_beam.sort(key=lambda n: (-n.mean_score, -n.projected_pct, n.weeks))
            beam = next_beam[:SETTINGS.tot_beam_width]
            for node in beam:
                if node.projected_pct >= benchmark_pct:
                    completed.append(node)

        pool = completed or beam
        pool = [n for n in pool if n.steps]
        if not pool:
            return None
        pool.sort(key=lambda n: (-n.score_sum, -n.projected_pct, n.weeks))
        stats = {
            "candidate_skills": len(candidates),
            "nodes_expanded": nodes_expanded,
            "nodes_pruned": nodes_pruned,
            "completed_paths": len(completed),
            "beam_width": SETTINGS.tot_beam_width,
            "max_depth": SETTINGS.tot_max_depth,
        }
        return pool[0], stats

    # ------------------------------------------------------------------ #
    def _score_step(
        self,
        skill: str,
        course: Course,
        sector: str,
        node: _Node,
        budget: Optional[float],
        timeframe: Optional[int],
        benchmark_pct: float,
        preferred_format: Optional[str],
    ) -> tuple[float, float]:
        w = SETTINGS.tot_score_weights
        sector_pct = self.store.sector_skill_pct(skill, sector)
        uplift = sector_pct * SETTINGS.tot_uplift_damping

        # normalised so a ~50%-of-postings skill scores ~1.0 and demand
        # differences between candidates stay visible (not all clamped to 1)
        demand_relevance = min(1.0, sector_pct / 50.0)

        if budget:
            budget_head = max(0.0, 1.0 - (node.cost + course.cost_usd) / budget)
        else:
            budget_head = 1.0
        if timeframe:
            time_head = max(0.0, 1.0 - (node.weeks + course.duration_weeks) / timeframe)
        else:
            time_head = 1.0
        constraint_fit = 0.5 * (budget_head + time_head)

        quality = course.rating / 5.0 + self.provider_bias.get(course.provider, 0.0)
        quality = max(0.0, min(1.0, quality))

        stage = node.depth + 1
        prereq = {
            "Beginner": 1.0,
            "Intermediate": 0.8 if stage >= 2 else 0.6,
            "Advanced": 0.8 if stage >= 3 else 0.4,
        }.get(course.level, 0.6)

        if not preferred_format or course.format == preferred_format:
            format_fit = 1.0
        else:
            format_fit = 0.5

        score = (
            w["demand_relevance"] * demand_relevance
            + w["constraint_fit"] * constraint_fit
            + w["quality"] * quality
            + w["prerequisite"] * prereq
            + w["format_fit"] * format_fit
        )
        return score, uplift
