"""The specialist bots from design doc sections 4, 8 and 11.

Eight core agents (the Query Rewriter Bot normalises input ahead of intent
capture) + two conditional extension agents, each with a distinct tool profile
and failure mode (design doc section 11 "role-separation test").
"""

from .career_advisor import CareerAdvisorBot
from .query_rewriter import QueryRewriterBot
from .skill_intelligence import SkillIntelligenceBot
from .job_market_research import JobMarketResearchBot
from .forecast_analytics import ForecastAndAnalyticsBot
from .upskilling import UpskillingBot
from .governance_qa import GovernanceAndQABot
from .niche_governance import NicheSkillGovernanceBot
from .local_niche_intel import LocalNicheIntelligenceBot
from .supervisor import SupervisorBot

__all__ = [
    "CareerAdvisorBot",
    "QueryRewriterBot",
    "SkillIntelligenceBot",
    "JobMarketResearchBot",
    "ForecastAndAnalyticsBot",
    "UpskillingBot",
    "GovernanceAndQABot",
    "NicheSkillGovernanceBot",
    "LocalNicheIntelligenceBot",
    "SupervisorBot",
]
