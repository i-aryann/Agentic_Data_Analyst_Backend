"""
Planner + Hypothesis Generator Node — LLM-powered (Groq).
Receives the user query, dataset profile, and memory context.
Produces a structured analysis plan and falsifiable hypotheses.
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
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def planner_node(state: dict) -> dict:
    """
    LLM Call #1: Takes the user query + dataset profile + memory and returns
    a structured plan (list of tasks) and hypotheses (list of claims to test).
    """
    query = state["query"]
    profile = state["profile"]
    memory = state.get("memory", [])

    logger.info("🧠 [Planner] Generating analysis plan and hypotheses...")

    llm = _get_llm()

    user_prompt = build_planner_user_prompt(query, profile, memory)

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
        hypotheses = parsed.get("hypotheses", [])
    except json.JSONDecodeError:
        logger.warning(f"🧠 [Planner] Failed to parse LLM JSON. Raw output:\n{raw_text[:500]}")
        # Fallback: create a basic plan from the query
        plan = [f"Analyze the dataset to answer: {query}"]
        hypotheses = [f"The data contains patterns related to: {query}"]

    logger.info(f"🧠 [Planner] Plan has {len(plan)} steps, {len(hypotheses)} hypotheses.")
    for i, step in enumerate(plan):
        logger.info(f"   📋 Step {i+1}: {step}")
    for h in hypotheses:
        logger.info(f"   💡 {h}")

    return {"plan": plan, "hypotheses": hypotheses}
