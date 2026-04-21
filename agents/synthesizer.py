"""
Synthesizer Node (Phase 2+) — LLM-powered (Groq).
Focuses ONLY on narrative formatting: summary, insights, and hypothesis validation.
Chart configurations are now handled by agents/visualizer.py (charts_spec in state).
"""
import json
import uuid
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_user_prompt

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_SYNTHESIZER", "llama-3.3-70b-versatile"),
            temperature=0.4,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def _build_error_response(query: str, error: str) -> dict:
    """Generate a graceful fallback response when everything fails."""
    aid = f"A-{str(uuid.uuid4())[:6]}"
    return {
        "summary": {
            "title": "Analysis Could Not Be Completed",
            "description": f"The system encountered an error while trying to analyze your query: '{query}'. "
                           f"This may be due to incompatible data or an unsupported question format. "
                           f"Please try rephrasing your question or check the dataset.",
            "primary_driver": "Execution Error",
            "impact_score": 0.0,
            "confidence": 0,
        },
        "insights": [
            {
                "id": "I-ERR",
                "category": "RISK_ALERT",
                "title": "Analysis Failed",
                "description": f"Error details: {error[:300] if error else 'Unknown error'}",
            }
        ],
        "charts": [],
        "hypotheses": [],
        "filters": {},
        "analysis_id": aid,
    }


def synthesizer_node(state: dict) -> dict:
    """
    LLM Call #5 (Phase 2): Takes raw execution results and the original context, then
    formats the narrative portion (summary, insights, hypotheses) into the frontend JSON.
    Charts are merged in from charts_spec (produced by the Visualizer node).
    """
    query = state["query"]
    execution_result = state.get("execution_result")
    execution_error = state.get("execution_error")
    hypotheses = state.get("hypotheses", [])
    plan = state.get("plan", [])
    charts_spec = state.get("charts_spec", [])    # Phase 2: from Visualizer
    critic_feedback = state.get("critic_feedback", "")  # Phase 2: from Critic on refinement loops

    if critic_feedback:
        logger.info("📝 [Synthesizer] Refinement loop — incorporating critic feedback into narrative...")
    else:
        logger.info("📝 [Synthesizer] Formatting narrative for frontend...")

    # If both result and error are missing, return error response
    if not execution_result and not execution_error:
        logger.warning("📝 [Synthesizer] No results and no error — returning fallback.")
        return {"final_response": _build_error_response(query, "No results produced.")}

    llm = _get_llm()

    user_prompt = build_synthesizer_user_prompt(
        query=query,
        execution_result=execution_result or "",
        execution_error=execution_error or "",
        hypotheses=hypotheses,
        plan=plan,
        critic_feedback=critic_feedback,
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
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

        # Inject analysis_id if not present
        if "analysis_id" not in parsed:
            parsed["analysis_id"] = f"A-{str(uuid.uuid4())[:6]}"

        # Ensure all required top-level keys exist
        parsed.setdefault("summary", None)
        parsed.setdefault("insights", [])
        parsed.setdefault("hypotheses", [])
        parsed.setdefault("filters", {})

        # Phase 2: Merge in chart specs from the dedicated Visualizer node.
        # If the Visualizer produced charts, use them. Otherwise fall back to any
        # charts the Synthesizer LLM may have produced (backward-compatible).
        if charts_spec:
            parsed["charts"] = charts_spec
            logger.info(f"📝 [Synthesizer] Using {len(charts_spec)} chart(s) from Visualizer.")
        else:
            parsed.setdefault("charts", [])
            logger.info("📝 [Synthesizer] No Visualizer charts — using LLM-generated charts (if any).")

        logger.info(
            f"📝 [Synthesizer] Done. "
            f"{len(parsed.get('insights', []))} insights, "
            f"{len(parsed.get('charts', []))} charts, "
            f"{len(parsed.get('hypotheses', []))} hypotheses."
        )

        return {"final_response": parsed}

    except json.JSONDecodeError:
        logger.warning(f"📝 [Synthesizer] Failed to parse LLM JSON:\n{raw_text[:500]}")
        return {"final_response": _build_error_response(query, "Synthesizer failed to produce valid JSON.")}

    except Exception as e:
        logger.error(f"📝 [Synthesizer] Unexpected error: {e}")
        return {"final_response": _build_error_response(query, str(e))}
