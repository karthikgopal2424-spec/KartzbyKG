"""Deterministic synthetic-data generator for the SKILLKARTZ / ForecastAI capstone.

Run this to (re)build the three JSON files the pipeline reads:

    python data/generate_dataset.py

Everything is seeded, so the output is byte-stable across machines. The generated
files are committed to the repo as well, so the project runs without this step.

Design intent (maps to design doc section 3 "Operating Environment" and section 9
"Retrieval Design"): we need a corpus where per-sector skill percentages are
*known by construction*, so the Forecast Bot's arithmetic can be checked against
ground truth, and where duplicates and offline-only niche skills actually exist so
the dedupe step and the niche-extension path have something to act on.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260726
DATA_DIR = Path(__file__).resolve().parent
TODAY = date(2026, 9, 2)  # frozen "now" so posting ages are stable

SECTORS = [
    "Banking",
    "Healthcare",
    "Retail",
    "Technology",
    "Manufacturing",
    "Education",
    "Logistics",
]

LOCATIONS = [
    ("Chicago", 0.22),
    ("New York", 0.20),
    ("London", 0.16),
    ("Bangalore", 0.16),
    ("Singapore", 0.12),
    ("Remote", 0.14),
]

EXPERIENCE = [("Entry", 0.30), ("Mid", 0.45), ("Senior", 0.25)]

ONLINE_SOURCES = [
    "GlobalJobBoard",
    "CareerPortalX",
    "EmployerCareersPage",
    "PublicLaborFeed",
]
OFFLINE_SOURCES = [
    "RegionalTribune",
    "CommunityJobBoard",
    "TradeAssociationBulletin",
    "GovEmploymentExchange",
    "TrainingInstitutePlacementCell",
]

# --- Skill taxonomy -----------------------------------------------------------
# canonical -> synonyms, related terms, and whether it is an offline-leaning
# niche/trade skill (structurally under-represented on job boards).
TAXONOMY = {
    "Python": {
        "synonyms": ["py"],
        "related": ["pandas", "numpy", "scripting", "data pipelines"],
        "niche": False,
        "complementary": ['SQL', 'Data Analysis', 'Machine Learning'],
    },
    "SQL": {
        "synonyms": ["structured query language"],
        "related": ["databases", "query optimization", "etl"],
        "niche": False,
        "complementary": ['Python', 'Data Analysis', 'Power BI'],
    },
    "Machine Learning": {
        "synonyms": ["ml", "ai/ml"],
        "related": ["scikit-learn", "model training", "mlops"],
        "niche": False,
        "complementary": ['Python', 'SQL', 'Data Analysis'],
    },
    "Excel": {
        "synonyms": ["spreadsheets", "microsoft excel"],
        "related": ["pivot tables", "vlookup"],
        "niche": False,
        "complementary": ['Data Analysis', 'SQL', 'Power BI'],
    },
    "Data Analysis": {
        "synonyms": ["data analytics", "data analyst"],
        "related": ["statistics", "reporting"],
        "niche": False,
        "complementary": ['SQL', 'Excel', 'Python', 'Tableau'],
    },
    "Tableau": {
        "synonyms": [],
        "related": ["data visualization", "bi dashboards"],
        "niche": False,
        "complementary": ['Power BI', 'SQL', 'Data Analysis'],
    },
    "Power BI": {
        "synonyms": ["powerbi"],
        "related": ["dax", "data visualization"],
        "niche": False,
        "complementary": ['Tableau', 'SQL', 'Excel'],
    },
    "Java": {
        "synonyms": [],
        "related": ["spring", "jvm", "backend"],
        "niche": False,
        "complementary": ['SQL', 'AWS', 'Docker'],
    },
    "React": {
        "synonyms": ["react.js", "reactjs"],
        "related": ["javascript", "frontend", "typescript"],
        "niche": False,
        "complementary": ['Java', 'AWS', 'Communication'],
    },
    "AWS": {
        "synonyms": ["amazon web services"],
        "related": ["cloud", "ec2", "s3"],
        "niche": False,
        "complementary": ['Docker', 'Kubernetes', 'Python'],
    },
    "Docker": {
        "synonyms": ["containers"],
        "related": ["kubernetes", "ci/cd", "AWS"],
        "niche": False,
        "complementary": ['Kubernetes', 'AWS', 'Python'],
    },
    "Kubernetes": {
        "synonyms": ["k8s"],
        "related": ["containers", "helm"],
        "niche": False,
        "complementary": ['Docker', 'AWS', 'Python'],
    },
    "Project Management": {
        "synonyms": ["pm", "project manager"],
        "related": ["agile", "scrum", "jira"],
        "niche": False,
        "complementary": ['Communication', 'Data Analysis', 'Excel'],
    },
    "Communication": {
        "synonyms": ["verbal communication", "written communication"],
        "related": ["stakeholder management", "presentation"],
        "niche": False,
        "complementary": ['Project Management', 'Data Analysis'],
    },
    "Cybersecurity": {
        "synonyms": ["infosec", "information security"],
        "related": ["siem", "soc", "incident response"],
        "niche": False,
        "complementary": ['Python', 'AWS', 'SQL'],
    },
    "Welding": {
        "synonyms": ["mig welding", "tig welding"],
        "related": ["fabrication", "blueprint reading", "metalwork"],
        "niche": True,
        "complementary": ['CNC Machining', 'Project Management', 'Communication'],
    },
    "HVAC": {
        "synonyms": ["heating ventilation air conditioning", "hvac technician"],
        "related": ["refrigeration", "ductwork", "epa 608"],
        "niche": True,
        "complementary": ['Welding', 'Project Management', 'Communication'],
    },
    "CNC Machining": {
        "synonyms": ["cnc", "cnc operator"],
        "related": ["g-code", "lathe operation", "milling"],
        "niche": True,
        "complementary": ['Welding', 'Data Analysis', 'Project Management'],
    },
    "Phlebotomy": {
        "synonyms": ["phlebotomist"],
        "related": ["venipuncture", "specimen handling", "patient care"],
        "niche": True,
        "complementary": ['Communication', 'Data Analysis', 'Project Management'],
    },
}

# Probability a posting in <sector> mentions <canonical skill>.
# Rows omitted => 0.02 (rare background noise). These are the ground-truth
# percentages the Forecast Bot should recover (± sampling noise).
PROPENSITY = {
    "Banking": {
        "Python": 0.22, "SQL": 0.55, "Excel": 0.68, "Data Analysis": 0.50,
        "Communication": 0.60, "Project Management": 0.38, "Tableau": 0.24,
        "Power BI": 0.30, "Machine Learning": 0.12, "Java": 0.28,
        "Cybersecurity": 0.18,
    },
    "Healthcare": {
        "Communication": 0.72, "Excel": 0.40, "Data Analysis": 0.30,
        "Project Management": 0.28, "SQL": 0.20, "Python": 0.10,
        "Phlebotomy": 0.16, "Cybersecurity": 0.10,
    },
    "Retail": {
        "Communication": 0.70, "Excel": 0.55, "Data Analysis": 0.34,
        "SQL": 0.26, "Power BI": 0.22, "Project Management": 0.30,
        "Python": 0.09, "Tableau": 0.16,
    },
    "Technology": {
        "Python": 0.66, "SQL": 0.52, "Machine Learning": 0.40, "AWS": 0.58,
        "Docker": 0.50, "Kubernetes": 0.36, "React": 0.44, "Java": 0.46,
        "Communication": 0.48, "Cybersecurity": 0.30, "Data Analysis": 0.40,
    },
    "Manufacturing": {
        "Welding": 0.30, "CNC Machining": 0.26, "HVAC": 0.12,
        "Project Management": 0.34, "Excel": 0.44, "Communication": 0.50,
        "Data Analysis": 0.22, "SQL": 0.16, "Python": 0.08,
    },
    "Education": {
        "Communication": 0.78, "Excel": 0.42, "Project Management": 0.32,
        "Data Analysis": 0.24, "Python": 0.14, "SQL": 0.14, "Tableau": 0.10,
    },
    "Logistics": {
        "Excel": 0.58, "Communication": 0.56, "SQL": 0.30, "Data Analysis": 0.36,
        "Power BI": 0.26, "Project Management": 0.36, "Python": 0.12,
        "AWS": 0.14,
    },
}

SECTOR_SIZE = {
    "Banking": 130,
    "Healthcare": 120,
    "Retail": 110,
    "Technology": 160,
    "Manufacturing": 120,
    "Education": 90,
    "Logistics": 100,
}

TITLE_STEMS = {
    "Banking": ["Financial Analyst", "Risk Analyst", "Operations Associate",
                "Data Analyst", "Compliance Specialist", "Portfolio Analyst"],
    "Healthcare": ["Clinical Coordinator", "Health Data Analyst", "Lab Technician",
                   "Care Manager", "Medical Office Administrator"],
    "Retail": ["Store Manager", "Merchandising Analyst", "Category Manager",
               "Inventory Analyst", "Customer Insights Analyst"],
    "Technology": ["Software Engineer", "Data Scientist", "DevOps Engineer",
                   "Backend Developer", "Frontend Developer", "ML Engineer",
                   "Platform Engineer"],
    "Manufacturing": ["Production Technician", "Quality Engineer", "Plant Supervisor",
                      "Maintenance Technician", "Process Engineer"],
    "Education": ["Instructional Designer", "Program Coordinator", "Data Specialist",
                  "Academic Advisor", "Curriculum Analyst"],
    "Logistics": ["Supply Chain Analyst", "Operations Planner", "Logistics Coordinator",
                  "Demand Planner", "Warehouse Supervisor"],
}


def weighted_choice(rng, pairs):
    r = rng.random()
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if r <= acc:
            return value
    return pairs[-1][0]


def build_postings(rng):
    postings = []
    pid = 1
    for sector in SECTORS:
        n = SECTOR_SIZE[sector]
        props = PROPENSITY[sector]
        for _ in range(n):
            skills = sorted(s for s, p in props.items() if rng.random() < p)
            # guarantee at least one skill so a posting is never empty
            if not skills:
                skills = [max(props, key=props.get)]
            location = weighted_choice(rng, LOCATIONS)
            exp = weighted_choice(rng, EXPERIENCE)
            age = int(rng.triangular(2, 395, 90))
            posted = TODAY - timedelta(days=age)
            title = f"{rng.choice(TITLE_STEMS[sector])}"
            source = rng.choice(ONLINE_SOURCES)
            desc = (
                f"{title} role in the {sector.lower()} sector based in {location}. "
                f"We are seeking a {exp.lower()}-level professional. "
                f"Required skills: {', '.join(skills)}. "
                f"Familiarity with {rng.choice(skills)} tooling is expected."
            )
            postings.append({
                "id": f"JP-{pid:05d}",
                "title": title,
                "employer": f"{sector[:4].upper()}Corp-{rng.randint(10, 99)}",
                "sector": sector,
                "location": location,
                "experience_level": exp,
                "posting_date": posted.isoformat(),
                "source": source,
                "skills": skills,
                "description": desc,
            })
            pid += 1

    # Inject ~8% exact-content duplicates (a new id + possibly a different
    # source, identical role/employer/description) so dedupe is exercised.
    dup_count = int(len(postings) * 0.08)
    for original in rng.sample(postings, dup_count):
        clone = dict(original)
        clone["id"] = f"JP-{pid:05d}"
        clone["source"] = rng.choice(ONLINE_SOURCES)
        postings.append(clone)
        pid += 1

    rng.shuffle(postings)
    return postings


def build_local_signals(rng):
    """Approved offline sources for niche/trade skills (design doc section 8)."""
    records = []
    lid = 1
    niche_targets = [
        ("Welding", "Manufacturing", ["Chicago", "Singapore", "Bangalore"]),
        ("HVAC", "Manufacturing", ["Chicago", "New York", "London"]),
        ("CNC Machining", "Manufacturing", ["Bangalore", "Singapore", "Chicago"]),
        ("Phlebotomy", "Healthcare", ["New York", "London", "Chicago"]),
    ]
    for skill, sector, locs in niche_targets:
        for loc in locs:
            for _ in range(rng.randint(2, 4)):
                source = rng.choice(OFFLINE_SOURCES)
                reliability = round(rng.uniform(0.45, 0.95), 2)
                age = rng.randint(5, 160)
                posted = TODAY - timedelta(days=age)
                records.append({
                    "id": f"LS-{lid:04d}",
                    "skill": skill,
                    "sector": sector,
                    "location": loc,
                    "employer": f"Local-{skill.split()[0]}-{rng.randint(2, 40)}",
                    "posting_date": posted.isoformat(),
                    "source": source,
                    "source_reliability": reliability,
                    "licensing_ok": reliability >= 0.5,
                    "description": (
                        f"{skill} opening advertised via {source} in {loc}. "
                        f"Hands-on {skill.lower()} experience required."
                    ),
                })
                lid += 1
    rng.shuffle(records)
    return records


COURSE_PROVIDERS = ["OpenLearn", "SkillBridge", "CampusPro", "CertifyNow", "TradeGuild"]
FORMATS = ["online", "in-person", "hybrid"]


def build_course_catalog(rng):
    catalog = []
    cid = 1
    for skill, meta in TAXONOMY.items():
        n = rng.randint(2, 4)
        for i in range(n):
            provider = rng.choice(COURSE_PROVIDERS)
            level = rng.choice(["Beginner", "Intermediate", "Advanced"])
            fmt = "in-person" if meta["niche"] and i == 0 else rng.choice(FORMATS)
            cost = rng.choice([0, 39, 99, 149, 249, 399, 699, 1200])
            weeks = rng.choice([2, 4, 6, 8, 10, 12])
            # A small number of deliberately inactive/discontinued courses so the
            # Governance Bot's groundedness check has something to catch.
            active = not (cid % 17 == 0)
            catalog.append({
                "id": f"CR-{cid:04d}",
                "skill": skill,
                "title": f"{skill} {level} ({fmt})",
                "provider": provider,
                "cost_usd": cost,
                "duration_weeks": weeks,
                "format": fmt,
                "level": level,
                "rating": round(rng.uniform(3.2, 4.9), 1),
                "active": active,
            })
            cid += 1
    return catalog


def main():
    rng = random.Random(SEED)
    postings = build_postings(rng)
    local_signals = build_local_signals(rng)
    courses = build_course_catalog(rng)

    taxonomy_out = {
        "canonical_skills": TAXONOMY,
        "sectors": SECTORS,
        "locations": [name for name, _ in LOCATIONS],
        "generated_for_date": TODAY.isoformat(),
    }

    (DATA_DIR / "job_postings.json").write_text(
        json.dumps({"as_of": TODAY.isoformat(), "postings": postings},
                   indent=2, sort_keys=True) + "\n")
    (DATA_DIR / "skill_taxonomy.json").write_text(
        json.dumps(taxonomy_out, indent=2, sort_keys=True) + "\n")
    (DATA_DIR / "local_signals.json").write_text(
        json.dumps({"as_of": TODAY.isoformat(), "records": local_signals},
                   indent=2, sort_keys=True) + "\n")
    (DATA_DIR / "course_catalog.json").write_text(
        json.dumps({"courses": courses}, indent=2, sort_keys=True) + "\n")

    print(f"job_postings.json      {len(postings):>4} postings")
    print(f"local_signals.json     {len(local_signals):>4} records")
    print(f"course_catalog.json    {len(courses):>4} courses")
    print(f"skill_taxonomy.json    {len(TAXONOMY):>4} canonical skills")


if __name__ == "__main__":
    main()
