"""
System and user prompt templates for the Planner node (Phase 2+).
Responsibility: task decomposition ONLY. Hypothesis generation is handled by hypothesis.py.
"""

PLANNER_SYSTEM_PROMPT = """You are a senior data analyst working inside an autonomous analytical AI system.
Your ONLY job is to decompose the user's question into a concrete, step-by-step analysis PLAN.
Hypothesis generation is handled separately by another agent — do NOT include hypotheses here.

RULES:
- Produce 3–6 specific, actionable analysis steps that a Python/pandas programmer can execute.
- Reference EXACT column names from the dataset profile provided.
- Each step must describe a concrete operation (e.g. "Group by 'Region' and sum 'Revenue'" not "Analyze data").
- If the user's query is ambiguous, make reasonable assumptions and state them in a step.
- If CRITIC FEEDBACK is provided, your new plan MUST address the issues raised.
- Output ONLY valid JSON. No markdown, no commentary.

OUTPUT FORMAT (strict JSON):
{
  "plan": [
    "Step 1: description referencing exact column names",
    "Step 2: ...",
    "..."
  ]
}"""


def build_planner_user_prompt(query: str, profile: dict, memory: list, critic_feedback: str = "") -> str:
    """Build the user prompt. On refinement loops, critic_feedback is non-empty."""

    # Format memory as a brief conversation history (last 6 turns max)
    memory_text = "No previous conversation history."
    if memory:
        recent = memory[-6:]
        lines = []
        for m in recent:
            role = m.get("role", "user").upper()
            content = m.get("content", "")[:200]  # Truncate long content
            lines.append(f"  {role}: {content}")
        memory_text = "\n".join(lines)

    # Format the dataset profile compactly
    columns_info = []
    for col in profile.get("columns", []):
        dtype = profile.get("dtypes", {}).get(col, "unknown")
        missing = profile.get("missing_values", {}).get(col, 0)
        columns_info.append(f"  - {col} ({dtype}, {missing} missing)")

    columns_text = "\n".join(columns_info)

    # Numeric stats summary
    stats_text = ""
    numeric_stats = profile.get("numeric_stats", {})
    if numeric_stats:
        stat_lines = []
        for col, stats in numeric_stats.items():
            mean = stats.get("mean", "N/A")
            std = stats.get("std", "N/A")
            min_val = stats.get("min", "N/A")
            max_val = stats.get("max", "N/A")
            stat_lines.append(f"  - {col}: mean={mean}, std={std}, min={min_val}, max={max_val}")
        stats_text = "\n".join(stat_lines)

    # Categorical summary
    cat_text = ""
    cat_uniques = profile.get("categorical_uniques", {})
    if cat_uniques:
        cat_lines = []
        for col, info in cat_uniques.items():
            top = info.get("top_values", [])[:5]
            cat_lines.append(f"  - {col}: {info.get('unique_count', '?')} unique values. Top: {top}")
        cat_text = "\n".join(cat_lines)

    # Critic feedback block (only shown on refinement loops)
    feedback_block = ""
    if critic_feedback:
        feedback_block = f"""
CRITIC FEEDBACK (from previous attempt — you MUST address these issues):
{critic_feedback}

Your new plan must specifically fix the issues above. Think carefully before decomposing.
"""

    return f"""USER QUERY: {query}

DATASET PROFILE:
- Rows: {profile.get('row_count', '?')}
- Columns: {profile.get('column_count', '?')}

COLUMNS:
{columns_text}

NUMERIC STATS:
{stats_text if stats_text else '  No numeric columns.'}

CATEGORICAL COLUMNS:
{cat_text if cat_text else '  No categorical columns.'}

CONVERSATION HISTORY:
{memory_text}
{feedback_block}
Generate a structured analysis plan. Output ONLY valid JSON with a "plan" key."""
