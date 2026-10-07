# Mini-MARG

A prototype implementation of a production-grade AI/ML system covering core GenAI concepts end-to-end.

## Architecture

- **RAG Pipeline** — FAISS vector store + BM25 hybrid retrieval with LangChain
- **LLM** — Ollama (Llama3) running locally, accessed via LangGraph agent
- **PII Protection** — 3-layer masking: Presidio + spaCy NER + config-driven regex recognizers
- **Guardrails** — Custom 5-check engine: injection, length, blocked patterns, topics, hallucination
- **Query Pipeline** — 5-layer: input guardrail → PII mask → RAG agent → output guardrail → PII unmask
- **API** — FastAPI with Swagger UI

## Concepts Covered (in progress)

- [x] RAG (Retrieval-Augmented Generation)
- [x] Hybrid Retrieval (FAISS + BM25)
- [x] PII Detection & Masking (Presidio + spaCy)
- [x] Guardrails (input/output safety)
- [x] LangGraph Agent
- [x] Docker containerization
- [ ] LangGraph Checkpoints (conversation memory)
- [ ] Redis persistent memory
- [ ] RAGAS evaluation
- [ ] Neo4j / GraphRAG (ontology)
- [ ] Multi-agent architecture
- [ ] Multi-model routing (LiteLLM)
- [ ] CI/CD + AWS deployment (GitHub Actions + Terraform)

## Tech Stack

Python 3.11 · FastAPI · LangChain · LangGraph · Ollama · FAISS · Presidio · spaCy · Docker

## Quick Start

\`\`\`bash
docker build -t mini-marg:latest .
docker run -d --name mini-marg -p 8000:8000 --add-host=host.docker.internal:host-gateway mini-marg:latest
# Open http://localhost:8000/docs