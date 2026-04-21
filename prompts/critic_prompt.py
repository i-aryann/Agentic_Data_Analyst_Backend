"""
System and user prompt templates for the Critic / Validator node (Phase 2).
"""

CRITIC_SYSTEM_PROMPT = """You are a rigorous analytical quality reviewer inside an autonomous data analysis system.
Your job is to evaluate the quality and accuracy of an analysis response BEFORE it is sent to the user.

You will receive:
1. The original user query.
2. The raw execution results (data computed by Python/pandas).
3. The formatted final response (insights, hypotheses, summary) produced by the Synthesizer.

YOUR EVALUATION CHECKLIST:
1. GROUNDING: Are all insight claims in the final_response directly supported by numbers in the execution results?
   - Flag any claim that cannot be traced back to the execution data.
2. CONSISTENCY: Are hypothesis statuses (Validated/Rejected/Pending) logically consistent with the data?
   - A hypothesis is "Validated" only if the data numerically supports it.
   - A hypothesis is "Rejected" only if the data numerically contradicts it.
   - Otherwise it is "Pending".
3. COMPLETENESS: Does the analysis actually answer the user's original question?
   - Flag if the response goes off-topic or misses the core question.
4. CONTRADICTIONS: Are there any insights that contradict each other?
5. SAMPLE SIZE: Are claims statistically credible given the number of data rows available?

SCORING:
Assign an overall confidence score between 0.0 and 1.0:
- 0.0 – 0.5: Major issues — claims unsupported, wrong hypotheses, off-topic. Needs refinement.
- 0.5 – 0.7: Moderate issues — mostly right but with notable gaps or unsupported claims.
- 0.7 – 0.9: Good — minor issues only, suitable to send to user.
- 0.9 – 1.0: Excellent — all claims grounded, consistent, complete.

OUTPUT FORMAT (strict JSON only, no markdown):
{
  "valid": <true|false>,
  "confidence": <float 0.0–1.0>,
  "issues": [
    "Issue 1: specific description of the problem",
    "Issue 2: ..."
  ],
  "suggestions": [
    "Suggestion 1: how the planner should approach this differently",
    "Suggestion 2: ..."
  ],
  "feedback_for_planner": "A concise paragraph telling the planner EXACTLY what to fix in its next plan."
}

RULES:
- Be strict. A confidence of 0.9+ means you would stake your professional reputation on the response.
- If there are no issues, set "issues" to [] and "suggestions" to [].
- The "feedback_for_planner" field MUST be non-empty if confidence < 0.7. It must be specific and actionable.
- NEVER make up numbers — only reference values you can see in the execution results.
- Output ONLY valid JSON. No markdown, no preamble, no explanation."""


def build_critic_user_prompt(
    query: str,
    execution_result: str,
    final_response: dict,
    hypotheses: list,
) -> str:
    """Build the critic evaluation prompt.
    
    Applies hard size limits to both execution_result and final_response to
    prevent Groq 400 token-limit errors on large datasets.
    """
    import json

    _MAX_EXEC_CHARS = 2500
    _MAX_RESPONSE_CHARS = 3000

    hypo_text = "\n".join(f"  - {h}" for h in hypotheses) if hypotheses else "  None."

    # Truncate execution result
    if execution_result:
        exec_preview = execution_result[:_MAX_EXEC_CHARS]
        if len(execution_result) > _MAX_EXEC_CHARS:
            exec_preview += f" ... [TRUNCATED — {len(execution_result)} chars total]"
    else:
        exec_preview = "No execution results available."

    # Serialize final_response compactly (no indent) to minimize tokens, then truncate
    try:
        response_json = json.dumps(final_response, separators=(",", ":"))
    except Exception:
        response_json = str(final_response)
    
    if len(response_json) > _MAX_RESPONSE_CHARS:
        response_json = response_json[:_MAX_RESPONSE_CHARS] + " ... [TRUNCATED]"

    # Special case: if the final_response is an error response itself, assess leniently
    is_error_response = (
        final_response.get("summary", {}).get("primary_driver") == "Execution Error"
        if isinstance(final_response, dict) else False
    )
    error_note = (
        "\nNOTE: The final_response indicates an execution error. "
        "Evaluate whether the error response is clear and helpful. "
        "Do NOT score this as a planning failure — score 0.6 if the error is clearly explained."
        if is_error_response else ""
    )

    return f"""USER QUERY:
{query}

ORIGINAL HYPOTHESES TO TEST:
{hypo_text}

RAW EXECUTION RESULTS (ground truth):
{exec_preview}

SYNTHESIZED FINAL RESPONSE (what the user would see):
{response_json}
{error_note}

Evaluate the final response against the execution results and user query.
Output ONLY valid JSON with your evaluation."""

