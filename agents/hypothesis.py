"""
Hypothesis Generator Node (Phase 2) — LLM-powered (Groq).
Split from the Planner node. Receives the analysis plan and generates
richer, multi-angle hypotheses covering Causal, Comparative, Distributional,
and Temporal dimensions.
"""
import json
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_llm = None

HYPOTHESIS_SYSTEM_PROMPT = """You are a data science researcher who specializes in generating
falsifiable, multi-angle hypotheses for data analysis.

Given an analysis plan and dataset profile, generate 3–5 HYPOTHESES that the analysis should test.
Each hypothesis should be specific, falsifiable, and reference exact column names.

Cover a VARIETY of hypothesis types — aim for different types across your list:
- CAUSAL: "Increasing [X] causes [Y] to increase/decrease"
- COMPARATIVE: "[Group A] has significantly higher/lower [Metric] than [Group B]"
- DISTRIBUTIONAL: "[Column] follows a Pareto/normal/skewed distribution"
- TEMPORAL: "[Metric] shows a [seasonal/weekly/monthly] pattern"
- CORRELATION: "[Column A] is positively/negatively correlated with [Column B]"

RULES:
- Reference EXACT column names from the dataset profile.
- Each hypothesis must be testable with the data provided.
- Hypotheses should be diverse — do not repeat the same pattern.
- Output ONLY valid JSON. No markdown, no commentary.

OUTPUT FORMAT:
{
  "hypotheses": [
    "H1 [CAUSAL]: ...",
    "H2 [COMPARATIVE]: ...",
    "H3 [DISTRIBUTIONAL]: ...",
    "H4 [TEMPORAL]: ...",
    "H5 [CORRELATION]: ..."
  ]
}"""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_HYPOTHESIS", "llama-3.1-8b-instant"),
            temperature=0.5,   # Higher than planner — we want creative hypotheses
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def hypothesis_node(state: dict) -> dict:
    """
    Phase 2 — LLM Call #2 (new): Takes the plan + profile and generates
    multi-angle hypotheses. Previously this was bundled into the Planner.
    """
    plan = state.get("plan", [])
    profile = state.get("profile", {})
    query = state.get("query", "")

    logger.info("💡 [Hypothesis] Generating multi-angle hypotheses from plan...")

    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))

    columns_info = []
    for col in profile.get("columns", []):
        dtype = profile.get("dtypes", {}).get(col, "unknown")
        columns_info.append(f"  - {col} ({dtype})")
    columns_text = "\n".join(columns_info)

    numeric_cols = profile.get("numeric_columns", [])
    cat_cols = profile.get("categorical_columns", [])

    user_prompt = f"""ORIGINAL QUERY: {query}

ANALYSIS PLAN:
{plan_text}

DATASET COLUMNS:
{columns_text}

NUMERIC COLUMNS: {numeric_cols}
CATEGORICAL COLUMNS: {cat_cols}
ROWS: {profile.get('row_count', '?')}

Generate 3–5 diverse, falsifiable hypotheses. Output ONLY valid JSON."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=HYPOTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        hypotheses = parsed.get("hypotheses", [])

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"💡 [Hypothesis] Failed to parse LLM response: {e}. Using fallback.")
        hypotheses = [
            f"H1 [COMPARATIVE]: Different groups in the data have significantly different distributions",
            f"H2 [TEMPORAL]: There are time-based patterns in the key metrics",
            f"H3 [CORRELATION]: Key numeric columns are positively correlated with each other",
        ]

    logger.info(f"💡 [Hypothesis] Generated {len(hypotheses)} hypotheses.")
    for h in hypotheses:
        logger.info(f"   {h}")

    return {"hypotheses": hypotheses}
