"""
Graph State Schema — shared TypedDict that flows through every node.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    # --- Inputs (set by main.py before invoking the graph) ---
    query: str                          # User's natural language question
    file_id: str                        # UUID pointing to processed/{file_id}.csv
    file_path: str                      # Resolved absolute path to the CSV
    session_id: str                     # Session ID for memory
    memory: list                        # Past interactions from local store

    # --- Profiler Outputs ---
    profile: dict                       # Column stats, types, correlations, nulls, sample rows

    # --- Planner Outputs ---
    plan: list[str]                     # Decomposed sub-tasks (decomposition only in Phase 2+)

    # --- Hypothesis Generator Outputs (Phase 2: split from Planner) ---
    hypotheses: list[str]               # Generated falsifiable hypotheses to test

    # --- Code Generator Outputs ---
    generated_code: str                 # Python/pandas code string

    # --- Executor Outputs ---
    execution_result: Optional[str]     # Serialized RESULTS dict (JSON string)
    execution_error: Optional[str]      # Traceback string if failed
    retry_count: int                    # Current retry attempt (max 3)

    # --- Visualizer Outputs (Phase 2: split from Synthesizer) ---
    charts_spec: list                   # Plotly-optimized chart configs produced by Visualizer

    # --- Synthesizer Outputs ---
    final_response: dict                # Complete JSON payload for the frontend

    # --- Critic / Validator Outputs (Phase 2) ---
    validation: dict                    # {valid, confidence, issues, suggestions}
    refinement_count: int               # Number of Planner→Critic loops taken (max = MAX_REFINEMENT_LOOPS)
    critic_feedback: str                # Textual feedback passed from Critic back to Planner on retry
