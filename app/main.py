"""
main.py — FastAPI Application

5 ENDPOINTS:
  GET  /health          → is the server alive?
  POST /ingest          → load docs → chunk → embed → build FAISS index
  POST /query           → run the LangGraph agent, get an answer
  GET  /prompt-version  → which prompt version is currently active?
  POST /prompt-version  → switch to a different prompt version (v1/v2)

QUERY PIPELINE (5 layers):
  1. Input Guardrail   → block injection / harmful / out-of-scope
  2. PII Mask          → replace PAN/Aadhaar/names with deterministic tokens
  3. RAG Agent         → intent → retrieve → generate
  4. Output Guardrail  → block hallucinations / financial advice / leaks
  5. PII Unmask        → restore original values in final answer
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yaml
from pathlib import Path

from app.rag import ingest
from app.agent import run_agent
from app.prompts import get_prompt, get_active_version
from app.pii import mask_pii, unmask_pii, pii_report
from app.guardrails import check_input, check_output, apply_output, GuardrailAction

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
    query: str

    class Config:
        json_schema_extra = {
            "example": {"query": "What is the interest rate for personal loans?"}
        }


class QueryResponse(BaseModel):
    answer:           str
    intent:           Optional[str]   = None
    prompt_version:   Optional[str]   = None
    messages:         Optional[list]  = None
    error:            Optional[str]   = None
    # New security fields
    pii_detected:     Optional[bool]  = None   # was PII found in the query?
    guardrail_warned: Optional[bool]  = None   # did output guardrail trigger a warning?


class PromptSwitchRequest(BaseModel):
    version: str

    class Config:
        json_schema_extra = {"example": {"version": "v1"}}


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 1 — HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
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
    try:
        result = ingest()
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
    5-layer secure query pipeline:

    Layer 1 — INPUT GUARDRAIL
      Check for prompt injection, blocked topics, SQL injection, length limits.
      If blocked → return 400 immediately, never reach the LLM.

    Layer 2 — PII MASKING
      Replace PAN, Aadhaar, names, phone numbers with deterministic tokens.
      The LLM never sees raw PII — only tokens like [IN_PAN_A3F2B891].
      Deterministic = same value → same token → FAISS vectors stay consistent.

    Layer 3 — RAG AGENT
      Intent classification → tool call → retrieval → LLM generation.
      Runs on the masked query, produces a masked answer.

    Layer 4 — OUTPUT GUARDRAIL
      Check LLM response for hallucination signals, definitive financial advice,
      internal system info leakage. Block or append disclaimer as needed.

    Layer 5 — PII UNMASKING
      Restore original PII values in the final answer using the token mapping
      from Layer 2. User sees their data back, LLM never stored it.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── Layer 1: Input Guardrail ──────────────────────────────
    input_check = check_input(request.query)
    if input_check.action == GuardrailAction.BLOCK:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked by safety guardrail: {input_check.reason}"
        )

    # ── Layer 2: PII Masking ──────────────────────────────────
    masked_query, pii_mapping = mask_pii(request.query)
    pii_detected = len(pii_mapping) > 0

    # Audit log — log what PII types were found, never log actual values
    if pii_detected:
        audit = pii_report(request.query)
        detected_types = [r["entity_type"] for r in audit]
        print(f"[PII AUDIT] Detected entities: {detected_types} — masked before LLM call")

    # ── Layer 3: RAG Agent ────────────────────────────────────
    try:
        result = run_agent(masked_query)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Call POST /ingest first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    raw_answer = result.get("answer", "")

    # ── Layer 4: Output Guardrail ─────────────────────────────
    output_check = check_output(raw_answer)

    if output_check.action == GuardrailAction.BLOCK:
        final_answer = f"Response blocked by safety policy: {output_check.reason}"
        guardrail_warned = False
    else:
        final_answer = apply_output(raw_answer, output_check)
        guardrail_warned = output_check.action.value in ("warn", "redact")

    # ── Layer 5: PII Unmasking ────────────────────────────────
    if pii_mapping:
        final_answer = unmask_pii(final_answer, pii_mapping)

    return QueryResponse(
        answer=final_answer,
        intent=result.get("intent"),
        prompt_version=result.get("prompt_version"),
        messages=result.get("messages"),
        error=result.get("error"),
        pii_detected=pii_detected,
        guardrail_warned=guardrail_warned,
    )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 4 — GET PROMPT VERSION
# ═══════════════════════════════════════════════════════════════

@app.get("/prompt-version")
def get_current_prompt_version():
    try:
        prompt = get_prompt()
        return {
            "active_version": prompt["version"],
            "name":           prompt["name"],
            "system_preview": prompt["system"][:100] + "...",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 5 — SWITCH PROMPT VERSION
# ═══════════════════════════════════════════════════════════════

@app.post("/prompt-version")
def switch_prompt_version(request: PromptSwitchRequest):
    try:
        with open(PROMPTS_PATH) as f:
            data = yaml.safe_load(f)

        available_versions = list(data["prompts"].keys())
        if request.version not in available_versions:
            raise HTTPException(
                status_code=400,
                detail=f"Version '{request.version}' not found. Available: {available_versions}"
            )

        old_version = data["active_version"]
        data["active_version"] = request.version

        with open(PROMPTS_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return {
            "status":        "success",
            "switched_from": old_version,
            "switched_to":   request.version,
            "message":       f"Active prompt switched from {old_version} to {request.version}."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch version: {str(e)}")