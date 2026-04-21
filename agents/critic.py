"""
Critic / Validator Node (Phase 2) — LLM-powered (Groq).
Quality gate that evaluates the Synthesizer's output before it reaches the frontend.
If confidence is below CRITIC_CONFIDENCE_THRESHOLD, triggers a refinement loop back to the Planner.
"""
import json
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.critic_prompt import CRITIC_SYSTEM_PROMPT, build_critic_user_prompt

logger = logging.getLogger(__name__)

_llm = None

# Load thresholds from environment (set in .env)
_CONFIDENCE_THRESHOLD = float(os.getenv("CRITIC_CONFIDENCE_THRESHOLD", "0.70"))
_MAX_REFINEMENT_LOOPS = int(os.getenv("MAX_REFINEMENT_LOOPS", "2"))


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_CRITIC", "llama-3.1-8b-instant"),
            temperature=0.1,   # Very low — critic must be precise and repeatable
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def _fallback_validation(reason: str) -> dict:
    """Return a high-confidence pass when the critic itself fails, to avoid blocking the pipeline."""
    return {
        "valid": True,
        "confidence": 0.75,
        "issues": [],
        "suggestions": [],
        "feedback_for_planner": "",
        "_critic_error": reason,
    }


def critic_node(state: dict) -> dict:
    """
    Phase 2 — Quality Gate: Evaluates the final_response from the Synthesizer.
    Outputs a validation dict containing confidence score, issues, and refinement feedback.
    The graph's conditional edge reads this to decide: END or refine.
    """
    query = state.get("query", "")
    final_response = state.get("final_response", {})
    execution_result = state.get("execution_result", "")
    hypotheses = state.get("hypotheses", [])
    refinement_count = state.get("refinement_count", 0)

    logger.info(
        f"🔍 [Critic] Evaluating response quality "
        f"(refinement loop {refinement_count}/{_MAX_REFINEMENT_LOOPS})..."
    )

    llm = _get_llm()

    user_prompt = build_critic_user_prompt(
        query=query,
        execution_result=execution_result or "",
        final_response=final_response,
        hypotheses=hypotheses,
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
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

        validation = json.loads(raw_text)

        # Ensure all required keys exist
        validation.setdefault("valid", True)
        validation.setdefault("confidence", 0.75)
        validation.setdefault("issues", [])
        validation.setdefault("suggestions", [])
        validation.setdefault("feedback_for_planner", "")

        confidence = validation["confidence"]
        logger.info(f"🔍 [Critic] Confidence: {confidence:.2f} (threshold: {_CONFIDENCE_THRESHOLD})")

        if validation["issues"]:
            for issue in validation["issues"]:
                logger.warning(f"   ⚠️  {issue}")

        # If confidence is below threshold and we have refinement budget, flag for retry
        if confidence < _CONFIDENCE_THRESHOLD and refinement_count < _MAX_REFINEMENT_LOOPS:
            logger.info(
                f"🔍 [Critic] Below threshold — triggering refinement loop "
                f"({refinement_count + 1}/{_MAX_REFINEMENT_LOOPS})."
            )
            validation["valid"] = False
        elif confidence < _CONFIDENCE_THRESHOLD and refinement_count >= _MAX_REFINEMENT_LOOPS:
            logger.warning(
                "🔍 [Critic] Below threshold but max refinements reached. "
                "Passing with low-confidence warning."
            )
            # Mark response with a warning so the frontend can surface it
            validation["valid"] = True  # Force pass — no more retries
            validation["max_refinements_reached"] = True

        return {
            "validation": validation,
            "critic_feedback": validation.get("feedback_for_planner", ""),
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"🔍 [Critic] Failed to parse evaluation: {e}. Applying fallback pass.")
        return {
            "validation": _fallback_validation(str(e)),
            "critic_feedback": "",
        }
