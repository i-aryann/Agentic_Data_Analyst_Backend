from fastapi import FastAPI, APIRouter, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import io
import os
import uuid
import json
import logging

from dotenv import load_dotenv
load_dotenv()

# Set up the formatter and file handler for all backend logs
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# Configure our main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Adding a StreamHandler so we get formatted logs in the console too
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Capture Uvicorn's internal logs (access and errors) into our file as well
logging.getLogger("uvicorn.access").addHandler(file_handler)
logging.getLogger("uvicorn.error").addHandler(file_handler)

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    user_schema: Dict[str, str]

@api_router.post("/process")
async def process_dataset(request: ProcessRequest):
    file_id = request.file_id
    user_schema = request.user_schema
    
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
# In-Memory Storage for Chat / LLM Context
# (We will replace this with Mem0 later for persistent storage)
# -----------------------------------------------------
chat_memory: Dict[str, List[Dict[str, Any]]] = {}

def add_to_memory(session_id: str, role: str, content: str, metadata: dict = None):
    """
    Saves a message to the local memory array.
    When migrating to Mem0, this function will call the Mem0 client add() method.
    """
    if session_id not in chat_memory:
        chat_memory[session_id] = []
        
    chat_memory[session_id].append({
        "role": role, # 'user' or 'assistant'
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}
    })

def get_memory(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the memory history for a session.
    When migrating to Mem0, this will call the Mem0 client search() or history() method.
    """
    return chat_memory.get(session_id, [])

# -----------------------------------------------------
# -----------------------------------------------------
# Agentic Analysis Pipeline (LangGraph Phase 1)
# -----------------------------------------------------
from agents.graph import analysis_graph

class AnalyzeRequest(BaseModel):
    query: str
    session_id: str = "default"
    file_id: Optional[str] = None

@api_router.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    logger.info("=" * 50)
    logger.info("🎯 NEW ANALYSIS REQUEST RECEIVED!")
    logger.info(f"💬 Query: {request.query}")
    logger.info(f"🧠 Session ID: {request.session_id}")
    logger.info(f"📁 File ID: {request.file_id}")
    logger.info("=" * 50)

    # Validate that a file_id was provided
    if not request.file_id:
        raise HTTPException(
            status_code=400,
            detail="No dataset selected. Please upload and process a dataset first.",
        )

    # Resolve the processed CSV path
    file_path = os.path.join(PROCESSED_DIR, f"{request.file_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Processed dataset not found for file_id '{request.file_id}'. "
                   "Please confirm & process your dataset from the Data Source page.",
        )

    # Retrieve memory context
    memory = get_memory(request.session_id)
    if memory:
        logger.info(f"Found {len(memory)} previous interactions for memory context.")

    # Save the user query to memory
    add_to_memory(request.session_id, "user", request.query)

    # Build the initial state for the graph
    initial_state = {
        "query": request.query,
        "file_id": request.file_id,
        "file_path": os.path.abspath(file_path),
        "session_id": request.session_id,
        "memory": memory,
        "retry_count": 0,
    }

    logger.info("🚀 Invoking LangGraph analysis pipeline...")

    try:
        # Run the compiled LangGraph
        final_state = analysis_graph.invoke(initial_state)

        result = final_state.get("final_response", {})

        # Save the response to memory
        add_to_memory(
            request.session_id,
            "assistant",
            json.dumps(result, default=str)[:500],  # Truncate for memory
            metadata={"analysis_id": result.get("analysis_id", "unknown")},
        )

        logger.info(f"✅ Analysis complete. ID: {result.get('analysis_id')}")
        return result

    except Exception as e:
        logger.error(f"❌ Graph execution failed: {e}", exc_info=True)
        # Return a graceful error response instead of crashing
        aid = f"A-{str(uuid.uuid4())[:6]}"
        return {
            "summary": {
                "title": "Analysis Error",
                "description": f"An unexpected error occurred: {str(e)[:200]}",
                "primary_driver": "System Error",
                "impact_score": 0,
                "confidence": 0,
            },
            "insights": [],
            "charts": [],
            "hypotheses": [],
            "filters": {},
            "analysis_id": aid,
        }

@api_router.get("/memory/{session_id}")
async def fetch_memory(session_id: str):
    """
    API endpoint to inspect current memory for debugging.
    """
    return {"session_id": session_id, "history": get_memory(session_id)}

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
