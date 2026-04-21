"""
Code Generator Node — LLM-powered (Groq).
Generates executable Python/pandas code based on the analysis plan,
hypotheses, and dataset profile.
"""
import json
import logging
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.codegen_prompt import CODEGEN_SYSTEM_PROMPT, build_codegen_user_prompt

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("MODEL_CODEGEN", "llama-3.3-70b-versatile"),
            temperature=0.1,  # Low temperature for code generation
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _llm


def code_generator_node(state: dict) -> dict:
    """
    LLM Call #2: Generates pandas code to execute the analysis plan.
    On retries, includes the previous code and error traceback so the LLM can fix it.
    """
    plan = state.get("plan", [])
    hypotheses = state.get("hypotheses", [])
    profile = state.get("profile", {})
    file_path = state["file_path"]
    execution_error = state.get("execution_error")
    previous_code = state.get("generated_code") if execution_error else None
    retry_count = state.get("retry_count", 0)

    if execution_error:
        logger.info(f"🔧 [CodeGen] RETRY #{retry_count} — Fixing previous error...")
    else:
        logger.info("🔧 [CodeGen] Generating analysis code...")

    llm = _get_llm()

    user_prompt = build_codegen_user_prompt(
        plan=plan,
        hypotheses=hypotheses,
        profile=profile,
        file_path=file_path,
        execution_error=execution_error,
        previous_code=previous_code,
    )

    messages = [
        SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw_code = response.content.strip()

    # Strip markdown code fences if present
    if raw_code.startswith("```"):
        lines = raw_code.split("\n")
        # Remove first and last lines (the fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_code = "\n".join(lines)

    logger.info(f"🔧 [CodeGen] Generated {len(raw_code)} chars of Python code.")
    logger.info(f"🔧 [CodeGen] Code preview:\n{raw_code[:300]}...")

    return {"generated_code": raw_code}
