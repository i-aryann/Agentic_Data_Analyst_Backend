"""
Visualizer Node (Phase 2) — LLM-powered (Groq).
Dedicated chart type selection and configuration agent.
Runs after the Executor and before the Synthesizer.
Relieves the Synthesizer of chart logic so it can focus on narrative formatting.
"""
import json
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_llm = None

VISUALIZER_SYSTEM_PROMPT = """You are a data visualization expert. Your job is to decide the best
chart types and configurations for a set of analysis results, then produce Plotly-compatible chart specs.

CHART TYPE SELECTION GUIDE:
- "bar": Category comparison, rankings, distributions across groups
- "line": Trends over time, sequential data, growth curves
- "scatter": Relationships between two numeric variables, correlation analysis
- "pie": Proportions/composition of a whole (use sparingly, max 8 slices)

RULES:
1. Produce 1–3 charts. Quality over quantity.
2. Each chart must map directly to data available in the EXECUTION RESULTS.
3. x must be a list of labels/categories. y must be a list of matching numeric values.
4. y2 is optional — use it for a secondary metric on the same chart (e.g. % alongside absolute values).
5. Limit data points to a maximum of 15 per chart (aggregate or top-N if needed).
6. All lists (x, y, y2) must have matching lengths.
7. Do NOT invent data. Use only values present in the execution results.
8. Use descriptive titles that explain what the chart shows.
9. Output ONLY valid JSON. No markdown, no commentary.

OUTPUT FORMAT:
{
  "charts": [
    {
      "id": "C-1",
      "title": "Descriptive chart title",
      "subtitle": "Brief subtitle explaining the metric or time period",
      "chart_type": "<bar|line|scatter|pie>",
      "data": {
        "x": ["Label1", "Label2", "..."],
        "y": [100, 200, "..."],
        "y2": [10.5, 22.3, "..."]
      }
    }
  ]
}"""


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_VISUALIZER", "llama-3.1-8b-instant"),
            temperature=0.2,   # Low — chart configs need to be precise and consistent
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def visualizer_node(state: dict) -> dict:
    """
    Phase 2 — LLM Call #4 (new): Takes execution results and the analysis plan,
    then produces Plotly-optimized chart configurations. The Synthesizer
    consumes charts_spec directly instead of generating charts itself.
    """
    execution_result = state.get("execution_result")
    execution_error = state.get("execution_error")
    plan = state.get("plan", [])
    query = state.get("query", "")

    logger.info("📊 [Visualizer] Selecting chart types and building chart specs...")

    # If there are no results to visualize, return an empty spec
    if not execution_result:
        logger.warning("📊 [Visualizer] No execution results available — skipping chart generation.")
        return {"charts_spec": []}

    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))

    # Truncate execution result to prevent token overflow
    exec_preview = execution_result[:4000] if len(execution_result) > 4000 else execution_result

    user_prompt = f"""ORIGINAL USER QUERY: {query}

ANALYSIS PLAN (what was analyzed):
{plan_text}

EXECUTION RESULTS (raw data to chart):
{exec_preview}

Based on the execution results and analysis plan, produce the best 1–3 chart configurations.
Each chart must use only data values present in the execution results.
Output ONLY valid JSON."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=VISUALIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        charts = parsed.get("charts", [])

        logger.info(f"📊 [Visualizer] Generated {len(charts)} chart spec(s).")
        for c in charts:
            logger.info(f"   📈 {c.get('chart_type', '?')} — {c.get('title', 'untitled')}")

        return {"charts_spec": charts}

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"📊 [Visualizer] Failed to generate chart specs: {e}. Returning empty.")
        return {"charts_spec": []}
