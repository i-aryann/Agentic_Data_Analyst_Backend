"""
Code Executor Node — Deterministic (no LLM).
Runs the LLM-generated Python code via exec() in a controlled namespace.
Returns the RESULTS dict or a traceback on failure.
"""
import json
import traceback
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _make_serializable(obj):
    """Recursively convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def executor_node(state: dict) -> dict:
    """
    Executes the generated Python code in a sandboxed namespace.
    Returns the RESULTS dict or captures the error traceback.
    """
    code = state["generated_code"]
    retry_count = state.get("retry_count", 0)

    logger.info(f"⚡ [Executor] Running generated code (attempt {retry_count + 1})...")

    # Controlled namespace — only pandas and numpy are available
    namespace = {
        "pd": pd,
        "np": np,
        "json": json,
    }

    try:
        exec(code, namespace)

        results = namespace.get("RESULTS", None)

        if results is None:
            return {
                "execution_result": None,
                "execution_error": "Code executed successfully but no RESULTS variable was defined. "
                                   "The generated code must store outputs in a variable called RESULTS.",
                "retry_count": retry_count + 1,
            }

        # Make everything JSON-serializable
        results = _make_serializable(results)

        serialized = json.dumps(results, default=str)

        logger.info(f"⚡ [Executor] Success! RESULTS has {len(results)} keys: {list(results.keys())}")

        return {
            "execution_result": serialized,
            "execution_error": None,
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.warning(f"⚡ [Executor] Code execution failed (attempt {retry_count + 1}):\n{error_trace[:500]}")

        return {
            "execution_result": None,
            "execution_error": error_trace,
            "retry_count": retry_count + 1,
        }
