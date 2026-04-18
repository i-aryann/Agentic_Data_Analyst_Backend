"""
System and user prompt templates for the Code Generator node.
"""

CODEGEN_SYSTEM_PROMPT = """You are an expert Python data analyst. You write clean, efficient pandas code.

Your job: Given an analysis plan, hypotheses, and dataset profile, write a COMPLETE standalone Python script
that reads a CSV file and performs the analysis.

CRITICAL RULES:
1. The CSV file path will be provided. Use pandas to read it: `df = pd.read_csv("<FILE_PATH>")`
2. Store ALL results in a dictionary called `RESULTS = {}`.
3. RESULTS must contain these keys:
   - "findings": list of strings, each describing a key finding from the data
   - "charts": list of chart config dicts, each with keys: "chart_type", "title", "subtitle", "x", "y", "y2" (optional)
     - chart_type must be one of: "bar", "line", "scatter"
     - x, y, y2 must be Python lists (not pandas Series)
4. Only import: pandas (as pd), numpy (as np). No other libraries.
5. Do NOT use print(), plt.show(), display(), or any I/O besides reading the CSV.
6. Do NOT use f-strings with nested quotes that could cause syntax errors.
7. Handle potential errors gracefully (e.g., missing columns, type mismatches).
8. Convert all numpy types to native Python types in RESULTS (use .tolist(), int(), float(), str()).
9. Limit chart data to a maximum of 20 data points (aggregate or take top-N if needed).
10. Output ONLY the Python code. No markdown fences, no commentary, no explanation.
"""


def build_codegen_user_prompt(
    plan: list,
    hypotheses: list,
    profile: dict,
    file_path: str,
    execution_error: str = None,
    previous_code: str = None,
) -> str:
    """Build the user prompt for code generation (or code fixing on retry)."""

    plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
    hypo_text = "\n".join(f"  - {h}" for h in hypotheses)

    columns_text = "\n".join(
        f"  - {col} ({profile.get('dtypes', {}).get(col, 'unknown')})"
        for col in profile.get("columns", [])
    )

    base_prompt = f"""ANALYSIS PLAN:
{plan_text}

HYPOTHESES TO TEST:
{hypo_text}

DATASET INFO:
- File path: {file_path}
- Rows: {profile.get('row_count', '?')}
- Columns:
{columns_text}

SAMPLE DATA (first 2 rows):
{profile.get('sample_rows', [])[:2]}

Write the complete Python script now. Output ONLY executable Python code."""

    # If this is a retry, include the error context
    if execution_error and previous_code:
        base_prompt = f"""YOUR PREVIOUS CODE FAILED. Fix the error and try again.

PREVIOUS CODE:
```python
{previous_code}
```

ERROR TRACEBACK:
{execution_error}

{base_prompt}

IMPORTANT: Fix the specific error above. Make sure all data types are correct and columns exist."""

    return base_prompt
