"""
System and user prompt templates for the Synthesizer node (Phase 2+).
Responsibility: narrative formatting only (summary, insights, hypotheses).
Charts are provided via charts_spec from the Visualizer and merged in by synthesizer_node.
"""

# Hard limits to prevent Groq 400 token-limit errors
_MAX_RESULT_CHARS = 4000
_MAX_ERROR_CHARS = 600
_MAX_PLAN_STEPS = 6

SYNTHESIZER_SYSTEM_PROMPT = """You are a senior business intelligence analyst. Your job is to take raw analysis
results and format them into a polished, structured JSON report for a frontend dashboard.

CRITICAL REQUIREMENTS — violating these will cause a quality failure:

1. QUERY RELEVANCE (most important):
   - The "summary.description" MUST directly and explicitly answer the ORIGINAL USER QUERY.
   - Do NOT produce a generic data summary. Instead, answer the user's specific question first,
     then provide supporting details.
   - Example: If the user asks "How have car buying patterns changed over the years?", the summary
     MUST say something like "Car buying patterns have shifted toward newer, lower-mileage vehicles
     with average selling prices increasing from X to Y between 2005 and 2020."

2. HYPOTHESIS COMPLETENESS:
   - EVERY hypothesis provided in the input MUST appear in the output "hypotheses" array.
   - Each hypothesis MUST have a status of "Validated", "Rejected", or "Pending".
   - Each hypothesis MUST include a brief justification with specific numbers from the execution results.
     For example: "Rejected — correlation between year and km_driven is -0.31, indicating a negative relationship."
   - Do NOT skip or omit any hypothesis. If data for a hypothesis is not in the results, set status to "Pending".

3. INSIGHT RIGOR:
   - Every insight description MUST cite at least one specific number from the execution results.
   - If the data contradicts an intuitive assumption, explicitly state the contradiction.
     Example: "Counter-intuitively, newer cars (higher year) tend to have LOWER km_driven (r = -0.31)."
   - Do NOT make vague statements like "there are interesting patterns" — be specific.

4. CONTRADICTION HANDLING:
   - If two data points appear to contradict each other, do NOT ignore it.
   - Create a RISK_ALERT insight explicitly flagging the contradiction with both numbers.

The frontend expects this EXACT JSON structure. You must output valid JSON only.

REQUIRED OUTPUT FORMAT:
{
  "summary": {
    "title": "Short, compelling report title (max 8 words)",
    "description": "2-3 sentences that DIRECTLY ANSWER the user's question with specific numbers",
    "primary_driver": "The single most important factor discovered",
    "impact_score": <float 1.0-10.0>,
    "confidence": <int 50-99>
  },
  "insights": [
    {
      "id": "I-1",
      "category": "<GROWTH_OPPORTUNITY|RISK_ALERT|PATTERN_MATCH>",
      "title": "Short insight title",
      "description": "2-3 sentence explanation citing specific numbers and calling out contradictions"
    }
  ],
  "charts": [],
  "hypotheses": [
    {
      "id": "H-1",
      "hypothesis": "The hypothesis statement",
      "confidence": <int 0-100>,
      "status": "<Validated|Pending|Rejected>",
      "agent": "Pandas Executor",
      "last_update": "Just now"
    }
  ],
  "filters": {
    "categories": ["list", "of", "filterable", "values"],
    "date_ranges": ["range1", "range2"]
  }
}

RULES:
- Generate 2-4 insights, each with a different category.
- Leave "charts": [] — charts are provided separately by the Visualizer.
- EVERY input hypothesis must appear in the output with a Validated/Rejected/Pending status.
- Use specific numbers from the results, never placeholder data.
- If the execution results are truncated (marked with [TRUNCATED]), use the numbers visible.
- If the execution failed, produce a graceful error summary with what went wrong.
- Output ONLY valid JSON. No markdown, no extra text, no explanation."""


def build_synthesizer_user_prompt(
    query: str,
    execution_result: str,
    execution_error: str,
    hypotheses: list,
    plan: list,
    critic_feedback: str = "",
) -> str:
    """Build the user prompt for the synthesizer node.
    
    Applies hard character limits to prevent Groq 400 token-limit errors on
    large datasets. The LLM is explicitly told when content is truncated so it
    doesn't mistake the cut-off for missing data.
    
    On refinement loops, critic_feedback is non-empty and injected so the
    Synthesizer knows exactly what to fix.
    """

    hypo_text = "\n".join(f"  - {h}" for h in hypotheses) if hypotheses else "  No hypotheses generated."

    # Limit plan steps to keep token count predictable
    plan_steps = plan[:_MAX_PLAN_STEPS]
    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan_steps)) if plan_steps else "  No plan."
    if len(plan) > _MAX_PLAN_STEPS:
        plan_text += f"\n  ... ({len(plan) - _MAX_PLAN_STEPS} more steps)"

    if execution_error and not execution_result:
        # Truncate tracebacks — they can be very verbose
        error_excerpt = execution_error[:_MAX_ERROR_CHARS]
        if len(execution_error) > _MAX_ERROR_CHARS:
            error_excerpt += " ... [TRUNCATED]"
        results_text = (
            f"EXECUTION FAILED.\n"
            f"Error (excerpt): {error_excerpt}\n\n"
            f"Generate a graceful error summary explaining what went wrong and what the user can try."
        )
    else:
        # Truncate large result payloads — the most common cause of 400 errors
        result_excerpt = execution_result[:_MAX_RESULT_CHARS]
        truncated = len(execution_result) > _MAX_RESULT_CHARS
        if truncated:
            result_excerpt += (
                f"\n... [TRUNCATED — showing first {_MAX_RESULT_CHARS} of "
                f"{len(execution_result)} chars. Use the visible numbers to form insights.]"
            )
        results_text = f"EXECUTION RESULTS{' (truncated)' if truncated else ''}:\n{result_excerpt}"

    # Critic feedback block — only shown on refinement loops
    feedback_block = ""
    if critic_feedback:
        feedback_block = f"""
CRITIC FEEDBACK (from previous attempt — you MUST fix these issues in your output):
{critic_feedback}

Your response MUST address every issue above. Do NOT repeat the same mistakes.
"""

    return f"""ORIGINAL USER QUERY: {query}

ANALYSIS PLAN:
{plan_text}

HYPOTHESES TO VALIDATE (you MUST include ALL of these in the output):
{hypo_text}

{results_text}
{feedback_block}
IMPORTANT REMINDERS:
1. Your summary.description MUST directly answer the user's query: "{query}"
2. Every hypothesis above MUST appear in the output with Validated/Rejected/Pending + justification.
3. Cite specific numbers from the execution results in every insight.

Leave "charts": []. Output ONLY valid JSON."""
