"""
main.py — FastAPI Application

5 ENDPOINTS:
  GET  /health          → is the server alive?
  POST /ingest          → load docs → chunk → embed → build FAISS index
  POST /query           → run the LangGraph agent, get an answer
  GET  /prompt-version  → which prompt version is currently active?
  POST /prompt-version  → switch to a different prompt version (v1/v2)

WHY FASTAPI?
  - Auto-generates interactive docs at /docs (Swagger UI)
  - Pydantic validation: wrong input = clear error, not a crash
  - Async support: can handle many requests without blocking
  - Type hints = self-documenting code
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from app.rag import ingest
from app.agent import run_agent
from app.prompts import get_prompt, get_active_version

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title="Mini-MARG",
    description="Mini Multi-Agent RAG — Banking Knowledge Assistant",
    version="1.0.0"
)

PROMPTS_PATH = Path(__file__).parent.parent / "prompts.yaml"


# ═══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS (Pydantic)
# ═══════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    Input model for POST /query
    
    Pydantic validates this automatically:
    - If 'query' is missing → 422 Unprocessable Entity (not a 500 crash)
    - If 'query' is not a string → 422 with clear message
    """
    query: str

    class Config:
        json_schema_extra = {
            "example": {"query": "What is the interest rate for personal loans?"}
        }


class QueryResponse(BaseModel):
    """Output model for POST /query"""
    answer:         str
    intent:         Optional[str]      = None    # "rag" or "calculator"
    prompt_version: Optional[str]      = None    # "v1" or "v2"
    messages:       Optional[list]     = None    # internal trace log
    error:          Optional[str]      = None    # error if any


class PromptSwitchRequest(BaseModel):
    """Input model for POST /prompt-version"""
    version: str

    class Config:
        json_schema_extra = {"example": {"version": "v1"}}


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 1 — HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """
    Simple liveness check.
    
    Used by:
    - AWS ALB (load balancer): pings /health every 30s to know if instance is alive
    - Docker healthcheck: container restarts if this fails
    - Monitoring: alerts if /health stops returning 200
    
    Rule: this endpoint must NEVER require a database or LLM call.
    It should respond in <10ms even if everything else is broken.
    """
    return {
        "status": "healthy",
        "service": "mini-marg",
        "version": "1.0.0"
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 2 — INGEST
# ═══════════════════════════════════════════════════════════════

@app.post("/ingest")
def ingest_documents():
    """
    Trigger the ingestion pipeline:
    load docs → inject config → chunk → embed → build FAISS + BM25 → save
    
    WHEN TO CALL THIS:
    - First time setup (before any /query calls)
    - After editing documents in data/docs/
    - After changing rates in config/rates.yaml
    
    This takes ~30-60 seconds (embedding all chunks with Ollama).
    In production you'd run this as a background job, not a synchronous endpoint.
    For our project, synchronous is fine.
    """
    try:
        result = ingest()          # calls rag.py ingest()
        return {
            "status": "success",
            "message": "Documents ingested and index built successfully.",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 3 — QUERY (MAIN ENDPOINT)
# ═══════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """
    Main endpoint — runs the LangGraph agent.
    
    FLOW:
    1. Receive user query
    2. Pass to run_agent() in agent.py
    3. Agent: classify intent → call tool → generate answer
    4. Return structured response
    
    EXAMPLE REQUEST:
      POST /query
      {"query": "What documents do I need for KYC?"}
    
    EXAMPLE RESPONSE:
      {
        "answer": "For KYC you need...",
        "intent": "rag",
        "prompt_version": "v2",
        "messages": ["INTENT CLASSIFIED: rag", "TOOL CALLED: rag_tool", ...]
      }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = run_agent(request.query)
        return QueryResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Call POST /ingest first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 4 — GET PROMPT VERSION
# ═══════════════════════════════════════════════════════════════

@app.get("/prompt-version")
def get_current_prompt_version():
    """
    Returns the currently active prompt version and its metadata.
    
    Useful for:
    - Knowing which version is live in production
    - A/B testing: check which version a specific instance is using
    - Debugging: "why did the answer format change?" → check prompt version
    """
    try:
        prompt = get_prompt()
        return {
            "active_version": prompt["version"],
            "name":           prompt["name"],
            "system_preview": prompt["system"][:100] + "...",    # first 100 chars
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 5 — SWITCH PROMPT VERSION
# ═══════════════════════════════════════════════════════════════

@app.post("/prompt-version")
def switch_prompt_version(request: PromptSwitchRequest):
    """
    Switch the active prompt version by updating prompts.yaml.
    
    THIS IS PROMPT VERSIONING IN ACTION:
    - POST /prompt-version {"version": "v1"}  → rolls back to v1
    - POST /prompt-version {"version": "v2"}  → switches to v2
    - No code change, no restart needed
    - Takes effect on the NEXT /query call immediately
    
    In production this would:
    - Require auth (only admins can switch prompts)
    - Log the switch with timestamp and user ID
    - Potentially trigger an evaluation run to compare versions
    """
    try:
        # Load current prompts.yaml
        with open(PROMPTS_PATH) as f:
            data = yaml.safe_load(f)

        # Validate the requested version exists
        available_versions = list(data["prompts"].keys())
        if request.version not in available_versions:
            raise HTTPException(
                status_code=400,
                detail=f"Version '{request.version}' not found. Available: {available_versions}"
            )

        old_version = data["active_version"]

        # Update active_version
        data["active_version"] = request.version

        # Write back to prompts.yaml
        with open(PROMPTS_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return {
            "status":      "success",
            "switched_from": old_version,
            "switched_to":   request.version,
            "message":     f"Active prompt switched from {old_version} to {request.version}. "
                           f"Next /query call will use {request.version}."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch version: {str(e)}")