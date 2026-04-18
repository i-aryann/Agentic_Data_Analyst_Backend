"""
Data Profiler Node — Deterministic (no LLM call).
Reads the processed CSV and extracts a comprehensive statistical profile
that feeds all downstream agents.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def profiler_node(state: dict) -> dict:
    """
    Reads the dataset and produces a compact profile dict containing
    column types, statistics, missing values, correlations, and sample rows.
    """
    file_path = state["file_path"]
    logger.info(f"📊 [Profiler] Reading dataset from: {file_path}")

    df = pd.read_csv(file_path)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Compute correlation matrix (numeric only), handle edge case of 0 numeric cols
    correlations = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        # Replace NaN with None for JSON serialization
        correlations = corr_matrix.where(~np.isnan(corr_matrix), None).to_dict()

    # Top unique values for categorical columns (max 10 per column)
    categorical_uniques = {}
    for col in categorical_cols:
        top_values = df[col].value_counts().head(10)
        categorical_uniques[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": top_values.index.tolist(),
            "top_counts": top_values.values.tolist(),
        }

    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "numeric_stats": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "correlations": correlations,
        "categorical_uniques": categorical_uniques,
        "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
    }

    logger.info(
        f"📊 [Profiler] Done. {profile['row_count']} rows, "
        f"{profile['column_count']} cols, "
        f"{len(numeric_cols)} numeric, {len(categorical_cols)} categorical."
    )

    return {"profile": profile}
