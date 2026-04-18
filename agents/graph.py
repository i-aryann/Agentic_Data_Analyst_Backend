"""
LangGraph Compilation — Wires all Phase 1 agent nodes into a StateGraph.

Graph Flow:
    profiler → planner → code_generator → executor → (retry or synthesizer) → END
"""
import logging

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.profiler import profiler_node
from agents.planner import planner_node
from agents.code_generator import code_generator_node
from agents.executor import executor_node
from agents.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def _should_retry_or_synthesize(state: dict) -> str:
    """
    Conditional edge after the Executor node.
    If execution failed and we haven't exceeded max retries, loop back to code_generator.
    Otherwise, proceed to synthesizer (with results or graceful error).
    """
    has_error = state.get("execution_error") is not None
    retry_count = state.get("retry_count", 0)

    if has_error and retry_count < MAX_RETRIES:
        logger.info(f"🔁 [Router] Execution failed. Retrying... ({retry_count}/{MAX_RETRIES})")
        return "code_generator"
    
    if has_error:
        logger.warning(f"🔁 [Router] Max retries ({MAX_RETRIES}) exceeded. Proceeding with error to synthesizer.")
    else:
        logger.info("🔁 [Router] Execution succeeded. Routing to synthesizer.")
    
    return "synthesizer"


def build_graph() -> StateGraph:
    """Construct and compile the Phase 1 analysis graph."""

    graph = StateGraph(AgentState)

    # --- Add Nodes ---
    graph.add_node("profiler", profiler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    # --- Set entry point ---
    graph.set_entry_point("profiler")

    # --- Add Edges ---
    graph.add_edge("profiler", "planner")
    graph.add_edge("planner", "code_generator")
    graph.add_edge("code_generator", "executor")

    # Conditional edge: retry on error or proceed to synthesizer
    graph.add_conditional_edges(
        "executor",
        _should_retry_or_synthesize,
        {
            "code_generator": "code_generator",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# Compile once at module level so it's reused across requests
analysis_graph = build_graph()

logger.info("✅ LangGraph Phase 1 analysis graph compiled successfully.")
