"""
LangGraph Compilation — Phase 2.
Wires all agent nodes into a StateGraph.

Phase 2 Graph Flow:
    profiler
        → planner
        → hypothesis_generator  [NEW: split from planner]
        → code_generator
        → executor
            → (retry loop: code_generator on error, max 3 retries)
        → visualizer            [NEW: dedicated chart selection]
        → synthesizer
        → critic                [NEW: quality gate]
            → END               (if confidence >= threshold)
            → planner           (if confidence < threshold AND refinement_count < max)
            → END with warning  (if confidence < threshold AND refinement_count >= max)
"""
import logging
import os

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.profiler import profiler_node
from agents.planner import planner_node
from agents.hypothesis import hypothesis_node
from agents.code_generator import code_generator_node
from agents.executor import executor_node
from agents.visualizer import visualizer_node
from agents.synthesizer import synthesizer_node
from agents.critic import critic_node

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_CONFIDENCE_THRESHOLD = float(os.getenv("CRITIC_CONFIDENCE_THRESHOLD", "0.70"))
_MAX_REFINEMENT_LOOPS = int(os.getenv("MAX_REFINEMENT_LOOPS", "2"))


# ---------------------------------------------------------------------------
# Conditional edge: Executor → retry or visualizer
# ---------------------------------------------------------------------------

def _should_retry_or_visualize(state: dict) -> str:
    """
    After Executor: retry code generation on error (max 3 times),
    or proceed to the Visualizer on success.
    """
    has_error = state.get("execution_error") is not None
    retry_count = state.get("retry_count", 0)

    if has_error and retry_count < MAX_RETRIES:
        logger.info(f"🔁 [Router] Execution failed. Retrying... ({retry_count}/{MAX_RETRIES})")
        return "code_generator"

    if has_error:
        logger.warning(f"🔁 [Router] Max retries ({MAX_RETRIES}) exceeded. Proceeding to visualizer with error.")
    else:
        logger.info("🔁 [Router] Execution succeeded. Routing to visualizer.")

    return "visualizer"


# ---------------------------------------------------------------------------
# Conditional edge: Critic → end or refine
# ---------------------------------------------------------------------------

def _should_refine_or_end(state: dict) -> str:
    """
    After Critic: check validation confidence.
    - Pass (confidence >= threshold) → END
    - Fail (confidence < threshold, budget remaining) → planner (refinement loop)
    - Fail (confidence < threshold, budget exhausted) → END with low-confidence warning
    """
    validation = state.get("validation", {})
    confidence = validation.get("confidence", 1.0)
    valid = validation.get("valid", True)
    refinement_count = state.get("refinement_count", 0)

    if valid or confidence >= _CONFIDENCE_THRESHOLD:
        logger.info(f"✅ [Critic Router] Confidence {confidence:.2f} — PASS. Sending to frontend.")
        return "end"

    if refinement_count < _MAX_REFINEMENT_LOOPS:
        logger.info(
            f"🔄 [Critic Router] Confidence {confidence:.2f} below {_CONFIDENCE_THRESHOLD} — "
            f"triggering refinement #{refinement_count + 1}."
        )
        return "planner"

    logger.warning(
        f"⚠️  [Critic Router] Confidence {confidence:.2f} below {_CONFIDENCE_THRESHOLD} but "
        f"max refinements ({_MAX_REFINEMENT_LOOPS}) reached. Forcing END with warning."
    )
    return "end"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the Phase 2 analysis graph."""

    graph = StateGraph(AgentState)

    # --- Add Nodes ---
    graph.add_node("profiler", profiler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("hypothesis_generator", hypothesis_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)

    # --- Entry point ---
    graph.set_entry_point("profiler")

    # --- Linear edges (happy path) ---
    graph.add_edge("profiler", "planner")
    graph.add_edge("planner", "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "code_generator")
    graph.add_edge("code_generator", "executor")

    # --- Conditional: retry on exec error or proceed to visualizer ---
    graph.add_conditional_edges(
        "executor",
        _should_retry_or_visualize,
        {
            "code_generator": "code_generator",
            "visualizer": "visualizer",
        },
    )

    # --- Visualizer → Synthesizer (linear) ---
    graph.add_edge("visualizer", "synthesizer")

    # --- Synthesizer → Critic (linear) ---
    graph.add_edge("synthesizer", "critic")

    # --- Conditional: Critic passes, or routes back to Planner for refinement ---
    graph.add_conditional_edges(
        "critic",
        _should_refine_or_end,
        {
            "end": END,
            "planner": "planner",   # Refinement loop: back to planner with critic_feedback
        },
    )

    return graph.compile()


# Compile once at module level so it's reused across requests
analysis_graph = build_graph()

logger.info("✅ LangGraph Phase 2 analysis graph compiled successfully.")
logger.info(
    f"   Config — Confidence threshold: {_CONFIDENCE_THRESHOLD}, "
    f"Max refinement loops: {_MAX_REFINEMENT_LOOPS}"
)
