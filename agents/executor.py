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

# Maximum size of the serialized RESULTS string passed through state.
# Downstream prompts truncate to ~4000 chars anyway, so anything larger
# is wasted tokens. 8 KB is generous for well-structured aggregated results.
_MAX_RESULT_BYTES = 8_000   # 8 KB


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
        # Convert DataFrame to compact summary instead of full dump
        if len(obj) > 10:
            return f"[DataFrame with {len(obj)} rows x {len(obj.columns)} cols — too large, use aggregation]"
        return obj.to_dict(orient="records")
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def _trim_results(results: dict) -> dict:
    """
    Aggressively trim oversized RESULTS to keep the serialized payload under
    _MAX_RESULT_BYTES. This prevents 400 token-limit errors in downstream LLM calls.

    Strategy:
    - List values with >10 items are sliced (charts/series data).
    - Any value that is a large list of dicts (raw DataFrame dump) is replaced
      with a compact summary.
    - Dict values with >15 keys are capped to their first 15 entries.
    """
    trimmed = {}
    for key, value in results.items():
        if isinstance(value, list):
            if len(value) > 10:
                # Check if it looks like a DataFrame dump (list of dicts)
                if value and isinstance(value[0], dict):
                    # Keep first 5 rows as a sample + note
                    trimmed[key] = {
                        "_note": f"Truncated: {len(value)} rows. Showing first 5.",
                        "sample": value[:5],
                    }
                else:
                    trimmed[key] = value[:10]
            else:
                trimmed[key] = value
        elif isinstance(value, dict):
            if len(value) > 15:
                # Large dict — keep first 15 keys
                keys_to_keep = list(value.keys())[:15]
                trimmed[key] = {k: value[k] for k in keys_to_keep}
                trimmed[key]["_note"] = f"Truncated: showing 15 of {len(value)} keys."
            elif value:
                trimmed[key] = _trim_results(value)
            else:
                trimmed[key] = value
        elif isinstance(value, str) and len(value) > 500:
            trimmed[key] = value[:500] + "... [TRUNCATED]"
        else:
            trimmed[key] = value
    return trimmed


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

        # Always run trim first to catch any raw dumps the LLM snuck in
        results = _trim_results(results)
        serialized = json.dumps(results, default=str)

        # Multi-pass: if still too large, trim again more aggressively
        pass_num = 0
        while len(serialized) > _MAX_RESULT_BYTES and pass_num < 3:
            pass_num += 1
            logger.warning(
                f"⚡ [Executor] RESULTS is {len(serialized):,} bytes "
                f"(limit: {_MAX_RESULT_BYTES:,}). Trim pass #{pass_num}..."
            )
            results = _trim_results(results)
            serialized = json.dumps(results, default=str)

        # Hard fallback: if trim loops couldn't bring it under the limit,
        # truncate the serialized string itself (preserving valid JSON structure)
        if len(serialized) > _MAX_RESULT_BYTES:
            logger.warning(
                f"⚡ [Executor] Still {len(serialized):,} bytes after trimming. "
                f"Hard-truncating serialized output to {_MAX_RESULT_BYTES:,} bytes."
            )
            serialized = serialized[:_MAX_RESULT_BYTES - 50] + '... [HARD TRUNCATED]"}'

        logger.info(
            f"⚡ [Executor] Success! RESULTS = {len(serialized):,} bytes, "
            f"{len(results)} keys: {list(results.keys())}"
        )

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
