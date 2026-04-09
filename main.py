from fastapi import FastAPI, APIRouter, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import io
import os
import uuid
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

api_router = APIRouter(prefix="/api")

def infer_column_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    try:
        pd.to_datetime(series.dropna().head(5))
        return "datetime"
    except Exception:
        pass
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "string"

@api_router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or "dataset"
    file_id = str(uuid.uuid4())
    
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    with open(file_path, "wb") as f:
        f.write(contents)
        
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), nrows=50)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents), nrows=50)
        else:
            return {"error": "Unsupported file type. Please upload CSV or Excel files."}
    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}

    columns = df.columns.tolist()
    schema = [{"name": c, "type": infer_column_type(df[c])} for c in columns]
    preview_rows = df.fillna("").to_dict(orient="records")
    size = len(contents)
    file_size = f"{size / (1024 * 1024):.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.1f}KB"

    metadata = {
        "filename": filename,
        "file_path": file_path,
        "original_schema": schema
    }
    with open(os.path.join(UPLOAD_DIR, f"{file_id}_meta.json"), "w") as f:
        json.dump(metadata, f)

    return {
        "file_id": file_id,
        "columns": columns,
        "schema": schema,
        "preview_rows": preview_rows,
        "filename": filename,
        "total_rows": "Unknown (pending)",
        "file_size": file_size
    }

class ProcessRequest(BaseModel):
    file_id: str
    schema: Dict[str, str]

@api_router.post("/process")
async def process_dataset(request: ProcessRequest):
    file_id = request.file_id
    user_schema = request.schema
    
    meta_path = os.path.join(UPLOAD_DIR, f"{file_id}_meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="File metadata not found")
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    file_path = metadata["file_path"]
    filename = metadata["filename"]
    
    dtype_map = {}
    parse_dates = []
    
    for col, col_type in user_schema.items():
        if col_type == "integer":
            dtype_map[col] = "Int64"
        elif col_type == "float":
            dtype_map[col] = float
        elif col_type == "boolean":
            dtype_map[col] = bool
        elif col_type == "string":
            dtype_map[col] = str
        elif col_type == "datetime":
            parse_dates.append(col)
            
    try:
        usecols = list(user_schema.keys())
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, usecols=usecols, dtype=dtype_map, parse_dates=parse_dates)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, usecols=usecols, dtype=dtype_map, parse_dates=parse_dates)
            
        processed_path = os.path.join(PROCESSED_DIR, f"{file_id}.csv")
        df.to_csv(processed_path, index=False)
        
        return {"status": "success", "message": "Dataset processed and typed successfully", "rows_processed": len(df)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


# -----------------------------------------------------
# Stubs for existing frontend operations
# -----------------------------------------------------

class AnalyzeRequest(BaseModel):
    query: str

@api_router.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    aid = f"A-{str(uuid.uuid4())[:6]}"

    result = {
        "summary": {
            "title": "APAC Market expansion showing 24% higher velocity than forecasted.",
            "description": "...",
            "primary_driver": "Cloud Adoption",
            "impact_score": 8.4,
            "confidence": 94
        },
        "insights": [],
        "charts": [],
        "hypotheses": [],
        "filters": {
            "regions": ["APAC", "North America", "Europe", "Latin America"],
            "categories": ["Enterprise", "Mid-Market", "SMB"],
            "date_ranges": ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"]
        },
        "analysis_id": aid
    }

    return result

@api_router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    return {"error": "Stub method"}

@api_router.get("/analyses")
async def list_analyses():
    return []

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Agentic Data Backend API running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
