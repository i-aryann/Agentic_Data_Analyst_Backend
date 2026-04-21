"""
Planner Node (Phase 2+) — LLM-powered (Groq).
Responsibility: task decomposition ONLY.
Hypothesis generation has been split into agents/hypothesis.py.

On refinement loops, the Critic's feedback is injected via state['critic_feedback']
so the Planner can produce a better decomposition on the second pass.
"""
import json
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt

logger = logging.getLogger(__name__)

# Lazy-initialize the LLM (created once, reused)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_PLANNER", "llama-3.3-70b-versatile"),
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def planner_node(state: dict) -> dict:
    """
    LLM Call #1: Decomposes the user query + dataset profile + memory into a
    structured analysis plan (list of tasks). On refinement loops, critic_feedback
    from the previous cycle is injected into the prompt.
    """
    query = state["query"]
    profile = state["profile"]
    memory = state.get("memory", [])
    critic_feedback = state.get("critic_feedback", "")
    refinement_count = state.get("refinement_count", 0)

    if critic_feedback:
        logger.info(
            f"🧠 [Planner] Refinement loop #{refinement_count + 1} — "
            f"incorporating critic feedback into new plan."
        )
    else:
        logger.info("🧠 [Planner] Generating analysis plan...")

    llm = _get_llm()

    user_prompt = build_planner_user_prompt(query, profile, memory, critic_feedback)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw_text = response.content.strip()

    # Parse the JSON response
    try:
        # Handle cases where LLM wraps output in markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        plan = parsed.get("plan", [])

    except json.JSONDecodeError:
        logger.warning(f"🧠 [Planner] Failed to parse LLM JSON. Raw output:\n{raw_text[:500]}")
        plan = [f"Analyze the dataset to answer: {query}"]

    logger.info(f"🧠 [Planner] Plan has {len(plan)} steps.")
    for i, step in enumerate(plan):
        logger.info(f"   📋 Step {i+1}: {step}")

    # On refinement loops, increment the counter
    updates = {"plan": plan}
    if critic_feedback:
        updates["refinement_count"] = refinement_count + 1

    return updates
