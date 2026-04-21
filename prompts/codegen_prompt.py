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
   - "findings": list of strings, each describing a key finding with SPECIFIC NUMBERS.
     Example: "Average selling price increased from 150000 in 2005 to 450000 in 2020."
   - "charts": list of chart config dicts, each with keys: "chart_type", "title", "subtitle", "x", "y", "y2" (optional)
     - chart_type must be one of: "bar", "line", "scatter"
     - x, y, y2 must be Python lists (not pandas Series)
   - "hypothesis_results": a dict mapping each hypothesis ID to its test result.
     Example: {"H1": {"verdict": "Validated", "evidence": "correlation = 0.82, p < 0.05"},
               "H2": {"verdict": "Rejected", "evidence": "mean A = 120, mean B = 118, difference not significant"}}
4. Only import: pandas (as pd), numpy (as np). No other libraries.
5. Do NOT use print(), plt.show(), display(), or any I/O besides reading the CSV.
6. Do NOT use f-strings with nested quotes that could cause syntax errors.
7. Handle potential errors gracefully (e.g., missing columns, type mismatches).
8. Convert all numpy types to native Python types in RESULTS (use .tolist(), int(), float(), str()).
9. Limit chart data to a maximum of 15 data points (aggregate or take top-N if needed).
10. Output ONLY the Python code. No markdown fences, no commentary, no explanation.

SIZE CONSTRAINTS — THIS IS CRITICAL:
11. RESULTS must be COMPACT. Target under 5 KB when serialized to JSON.
12. NEVER store raw DataFrames, raw row-level data, or df.to_dict() in RESULTS.
    - BAD:  RESULTS["data"] = df.to_dict("records")        # This can be megabytes!
    - BAD:  RESULTS["all_rows"] = df.values.tolist()        # Raw dump!
    - GOOD: RESULTS["avg_by_year"] = df.groupby("year")["price"].mean().to_dict()  # Aggregated!
    - GOOD: RESULTS["top_5"] = df.nlargest(5, "price")[["name","price"]].to_dict("records")  # Top-N!
13. For each analysis step, store ONLY the aggregated summary (mean, count, sum, correlation, top-N).
14. If a correlation or test is computed, store the numeric result directly:
    - GOOD: RESULTS["year_price_corr"] = float(df["year"].corr(df["selling_price"]))
    - BAD:  RESULTS["correlation_data"] = df[["year", "selling_price"]].to_dict()
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
