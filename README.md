# Mini-MARG

> **Mini** · **M**ulti-layer **A**I **R**eference **G**raph  
> A complete reference implementation of production-grade AI/ML/GenAI concepts — built to learn, reference, and adapt for real systems.

---

## What This Is

Mini-MARG is a prototype that deliberately implements **every major concept** in modern AI/ML systems — RAG, GraphRAG, memory stacks, agent patterns, evaluation, observability, and AWS deployment — in one codebase. The goal is not a product; it is a living reference you can pull patterns from when building real systems.

---

## Architecture Overview

```
                         User Query
                              │
              ┌───────────────▼───────────────┐
              │      Input Guardrail           │  injection · length · topics
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │         PII Masking            │  Presidio · spaCy · regex
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │     Adaptive Query Router      │  classifies: vector / bm25 /
              │   (rule-based + LLM fallback)  │  graph / hybrid / all
              └───────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
   ┌─────▼──────┐    ┌────────▼───────┐    ┌───────▼───────┐
   │ Vector     │    │  BM25 Keyword  │    │  GraphRAG     │
   │ Store      │    │  Search        │    │  Neo4j/Neptune│
   │ FAISS /    │    │                │    │  entity graph │
   │ Pinecone   │    │                │    │               │
   └─────┬──────┘    └────────┬───────┘    └───────┬───────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    Ensemble Merger (RRF)      │  Reciprocal Rank Fusion
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    Reranker (cross-encoder)   │  FlashRank · Cohere · BGE
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    LangGraph Agent            │  retrieve → generate
              │    + Memory Stack             │  checkpoints · Redis · entity
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    Output Guardrail           │  hallucination · citations
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    PII Unmask                 │  restore original values
              └───────────────┬───────────────┘
                              │
                           Response
                     (streamed · cited · safe)
```

---

## Concepts Implemented

### ✅ Completed

- [x] **RAG** — Retrieval-Augmented Generation (FAISS + BM25 hybrid)
- [x] **Hybrid Retrieval** — EnsembleRetriever with Reciprocal Rank Fusion (RRF)
- [x] **PII Detection & Masking** — 3-layer: Presidio + spaCy NER + config-driven regex (IN/UK/SG/US)
- [x] **Guardrails** — 5-check engine: SQL injection, script injection, length, blocked topics, internal info patterns
- [x] **LangGraph Agent** — StatefulGraph with retrieve_node → generate_node
- [x] **Docker** — containerised, Ollama via host bridge
- [x] **FastAPI** — Swagger UI, structured request/response models

### 🔄 In Progress / Planned

#### Step 11 — Vector Store + Advanced Retrieval
- [ ] FAISS ↔ Pinecone config switch (`vector_store: faiss | pinecone`)
- [ ] Namespace isolation per doc type (tanfeeth / bec / general)
- [ ] Rich metadata on every chunk (doc_type, source, page, section, date, language)
- [ ] Chunking strategies: fixed / recursive / semantic / late chunking
- [ ] Ingestion pipeline: PDF, DOCX, hierarchy preservation
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Multi-query retrieval + query rewriting
- [ ] Parent-child chunking (small-to-big retrieval)
- [ ] Adaptive Query Classifier (rule-based + LLM fallback)

#### Step 12 — GraphRAG
- [ ] Neo4j (local) / Amazon Neptune (AWS) — same openCypher, different URI
- [ ] Extraction strategy 1: LLM Triple Extractor (free-form Subject | Predicate | Object)
- [ ] Extraction strategy 2: Schema-Induced (auto-discover ontology from docs)
- [ ] Extraction strategy 3: spaCy Hybrid (spaCy NER + LLM for relationships)
- [ ] Extraction strategy 4: Microsoft GraphRAG (community detection, global/local search)
- [ ] Entity Resolution (embedding clustering, cosine sim > 0.92, canonical merge)
- [ ] Graph Admin API (correct/merge/reingest)
- [ ] Storage option A: separate (Neo4j + FAISS/Pinecone)
- [ ] Storage option B: unified (Neo4j vector index on Document nodes)
- [ ] Visualization: Neo4j Browser / Neptune Workbench / Neptune Graph Explorer (Docker)

#### Step 13 — Reranker + Retrieval Quality
- [ ] FlashRank cross-encoder (local, free)
- [ ] Cohere Rerank (cloud, production)
- [ ] BGE Reranker (open source)
- [ ] MMR — Max Marginal Relevance (deduplicate redundant chunks)
- [ ] Contextual Compression (strip irrelevant parts of chunks)
- [ ] Hallucination detection (verify answer grounded in retrieved chunks)
- [ ] Citation attribution (which chunk answered which claim)

#### Step 14 — Streaming
- [ ] SSE (Server-Sent Events) streaming endpoint
- [ ] LangGraph token-by-token streaming
- [ ] WebSocket alternative

#### Step 15 — Structured Output + Prompt Engineering
- [ ] Pydantic output enforcement + auto-retry on bad format
- [ ] Chain of Thought (CoT)
- [ ] Few-shot prompting
- [ ] Step-back prompting
- [ ] RAG-Fusion (multi-query + RRF)
- [ ] Prompt versioning + A/B testing

#### Step 16 — LangGraph Checkpoints (Short-term Memory)
- [ ] SQLite checkpointer (local dev)
- [ ] Redis checkpointer (AWS prod)
- [ ] Conversation replay and branching

#### Step 17 — Redis Memory Stack (Long-term Memory)
- [ ] Long-term episodic memory (conversation history across sessions)
- [ ] Entity memory (per user_id facts — "John prefers Arabic")
- [ ] Semantic cache (skip LLM if similar query answered recently)

#### Step 18 — Procedural Memory
- [ ] Tool / skill registry (what the agent knows how to do)
- [ ] Dynamic tool registration at runtime

#### Step 19 — Agent Patterns
- [ ] ReAct (Reason + Act) — foundational agentic loop
- [ ] Plan-and-Execute — plan full steps, then execute
- [ ] Self-Reflection / Self-Correction — agent critiques its own output
- [ ] Human-in-the-Loop — pause for approval before irreversible actions

#### Step 20 — Multi-Agent Architecture
- [ ] Orchestrator agent (decides which specialist to call)
- [ ] Tanfeeth specialist agent
- [ ] BEC specialist agent
- [ ] Regulation specialist agent
- [ ] Synthesizer agent (merges multi-agent answers)

#### Step 21 — Multi-model + LiteLLM
- [ ] LiteLLM as unified gateway
- [ ] Ollama (local) · OpenAI · Anthropic · Bedrock behind same interface
- [ ] Model routing by task type (cheap for classification, strong for generation)

#### Step 22 — RAGAS Evaluation Pipeline
- [ ] Faithfulness score
- [ ] Context precision + recall
- [ ] Answer relevance
- [ ] Regression suite on every change

#### Step 23 — Harness + Observability
- [ ] LangSmith / LangFuse tracing (every request, every step)
- [ ] Prompt versioning
- [ ] Cost tracking per query
- [ ] Latency alerts
- [ ] RAGAS scores in dashboard

#### Step 24 — Security Layer
- [ ] JWT authentication
- [ ] Rate limiting per user/IP
- [ ] Audit logging (who asked what, when)

#### Step 25 — Multimodal (flag-gated)
- [ ] OCR for scanned PDFs (pytesseract / AWS Textract)
- [ ] Vision LLM for diagram/image description
- [ ] Table extraction from PDFs

#### Step 26 — CI/CD + AWS + VPC
- [ ] GitHub Actions (test → build → push → deploy on every PR)
- [ ] Terraform (all infrastructure as code)
- [ ] VPC: public subnets (ALB) + private subnets (ECS, Neptune, Redis)
- [ ] NAT Gateway + Bastion Host (local dev access to private resources)
- [ ] Security Groups (ALB → ECS → Neptune/Redis, nothing else)
- [ ] VPC Endpoints (S3, Secrets Manager, ECR — no internet needed)
- [ ] Neptune + ElastiCache Redis in private subnets only
- [ ] Neptune Graph Explorer on ECS (visual graph browser)
- [ ] AWS Secrets Manager (never hardcode credentials)

---

## Memory Architecture

| Memory Type | Implementation | Step |
|---|---|---|
| Sensory (context window) | LLM context — built-in | ✅ |
| Working / In-context | Active conversation state | ✅ |
| Short-term Episodic | LangGraph checkpoints (SQLite/Redis) | Step 16 |
| Long-term Episodic | Redis persistent store | Step 17 |
| Semantic Memory | Vector store (FAISS/Pinecone) | Step 11 |
| Associative Memory | Neo4j knowledge graph | Step 12 |
| Entity Memory | Redis per-user facts | Step 17 |
| Semantic Cache | Redis similarity cache | Step 17 |
| Procedural Memory | Tool/skill registry | Step 18 |

---

## GraphRAG Extraction Strategies

| Strategy | When to use | Schema needed |
|---|---|---|
| LLM Triple Extractor | Unknown domain, exploration | No |
| Schema-Induced | Known domain, need consistency | Auto-discovered |
| spaCy Hybrid | Large volumes, cost matters | Optional |
| Microsoft GraphRAG | Production out-of-box, community detection | No |

All strategies share the same output format → Entity Resolution always runs after.

---

## Configuration

Everything is config-driven. One file controls the entire system:

```yaml
# config/settings.yaml

vector_store: faiss              # faiss | pinecone
graph_store: neo4j               # neo4j | neptune
graph_mode: separate             # separate | unified (neo4j vector index)

graph_extraction:
  strategy: triple               # triple | schema_induced | spacy_hybrid | microsoft_graphrag
  entity_resolution: true
  resolution_threshold: 0.92

reranker: flashrank              # flashrank | cohere | bge | none

retrieval:
  hyde_enabled: false            # hypothetical document embeddings
  multi_query: false             # generate multiple query variants
  parent_child: false            # small-to-big retrieval
  streaming: false               # SSE token streaming

memory:
  checkpointer: sqlite           # sqlite | redis
  episodic: false                # long-term episodic memory
  entity_memory: false           # per user_id facts
  semantic_cache: false          # skip LLM on similar queries

llm_gateway: ollama              # ollama | litellm

multimodal:
  ocr_enabled: false
  vision_llm: false
```

---

## Project Structure

```
mini-marg/
├── app/
│   ├── main.py                    # FastAPI entry point, 5-layer pipeline
│   ├── rag.py                     # RAG agent, LangGraph, retrieval
│   ├── pii.py                     # PII masking/unmasking
│   ├── guardrails.py              # Input/output safety checks
│   ├── vector/
│   │   ├── store_factory.py       # FAISS / Pinecone switch
│   │   └── retrievers.py          # HyDE, multi-query, parent-child
│   ├── graph/
│   │   ├── extractor_factory.py   # picks extraction strategy
│   │   ├── triple_extractor.py    # LLM triple extraction
│   │   ├── schema_induced.py      # auto-ontology discovery
│   │   ├── spacy_hybrid.py        # spaCy NER + LLM relationships
│   │   ├── msft_graphrag.py       # Microsoft GraphRAG wrapper
│   │   ├── entity_resolver.py     # embedding dedup + canonical merge
│   │   └── graph_admin.py         # correct/merge/reingest API
│   ├── memory/
│   │   ├── checkpointer.py        # SQLite / Redis checkpoints
│   │   ├── episodic.py            # long-term Redis memory
│   │   ├── entity_memory.py       # per-user facts
│   │   └── semantic_cache.py      # similarity cache
│   ├── agents/
│   │   ├── orchestrator.py        # multi-agent coordinator
│   │   ├── react_agent.py         # ReAct pattern
│   │   ├── plan_execute.py        # Plan-and-Execute pattern
│   │   └── specialists/           # domain-specific agents
│   └── evaluation/
│       └── ragas_runner.py        # evaluation pipeline
├── config/
│   ├── settings.yaml              # master config switch
│   ├── pii_recognizers.yaml       # country-specific PII patterns
│   ├── guardrails.yaml            # input/output rules
│   └── graph_ontology.yaml        # entity + relationship types
├── terraform/                     # AWS infrastructure as code
│   ├── vpc.tf                     # VPC, subnets, NAT, bastion
│   ├── neptune.tf                 # Amazon Neptune cluster
│   ├── elasticache.tf             # Redis cluster
│   └── ecs.tf                     # ECS service + ALB
├── .github/workflows/
│   └── deploy.yml                 # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── docker-compose.yml             # local dev with Neo4j + Redis
└── README.md
```

---

## Tech Stack

| Layer | Local Dev | AWS Production |
|---|---|---|
| **API** | FastAPI | FastAPI on ECS Fargate |
| **LLM** | Ollama (Llama3) | Amazon Bedrock / LiteLLM |
| **Embeddings** | nomic-embed-text (Ollama) | Bedrock Titan / OpenAI |
| **Vector Store** | FAISS | Pinecone |
| **Graph DB** | Neo4j Desktop | Amazon Neptune |
| **Memory / Cache** | SQLite | ElastiCache Redis |
| **Observability** | LangSmith | LangFuse |
| **Container** | Docker | ECS Fargate |
| **IaC** | — | Terraform |
| **CI/CD** | — | GitHub Actions |
| **VPC** | — | Private subnets, NAT, Bastion |

**Core libraries:** Python 3.11 · LangChain · LangGraph · Presidio · spaCy · FAISS · FlashRank · neo4j · redis · pydantic · RAGAS · LiteLLM

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/vishnuraj/mini-marg.git
cd mini-marg

# Local dev with Neo4j + Redis (docker-compose)
docker-compose up -d

# Or standalone
docker build -t mini-marg:latest .
docker run -d --name mini-marg -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  mini-marg:latest

# Open API explorer
open http://localhost:8000/docs

# Neo4j Browser (graph explorer)
open http://localhost:7474
```

### Environment Variables

```bash
# .env (never commit this)
VECTOR_STORE=faiss
GRAPH_STORE=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password
REDIS_URL=redis://localhost:6379
PINECONE_API_KEY=          # only if vector_store=pinecone
PINECONE_ENV=              # only if vector_store=pinecone
LANGSMITH_API_KEY=         # only if observability enabled
COHERE_API_KEY=            # only if reranker=cohere
```

---

## Security Notes

- `.env` files are never committed — always in `.gitignore`
- AWS credentials never stored in code or committed to git
- CSV files (`*.csv`, `*accessKeys*`) blocked by `.gitignore`
- All AWS secrets managed via AWS Secrets Manager in production
- Neptune and Redis run in private VPC subnets — no public internet access

---

## Roadmap Status

```
✅ Steps 1–10   Core pipeline (RAG · PII · Guardrails · LangGraph · Docker)
🔄 Step 11      Vector Store Factory + Advanced Retrieval          ← next
⬜ Step 12      GraphRAG (Neo4j/Neptune + 4 extraction strategies)
⬜ Step 13      Reranker + Hallucination Detection + Citations
⬜ Step 14      Streaming (SSE + WebSocket)
⬜ Step 15      Structured Output + Prompt Engineering
⬜ Step 16      LangGraph Checkpoints (short-term memory)
⬜ Step 17      Redis Memory Stack (long-term · entity · semantic cache)
⬜ Step 18      Procedural Memory (tool registry)
⬜ Step 19      Agent Patterns (ReAct · Plan-Execute · HITL · Self-reflect)
⬜ Step 20      Multi-Agent Architecture
⬜ Step 21      Multi-model + LiteLLM
⬜ Step 22      RAGAS Evaluation Pipeline
⬜ Step 23      Harness + Observability (LangSmith/LangFuse)
⬜ Step 24      Security Layer (JWT · rate limiting · audit log)
⬜ Step 25      Multimodal (OCR · vision · tables) — flag-gated
⬜ Step 26      CI/CD + AWS + VPC (GitHub Actions · Terraform · private subnets)
```

---

*Built as a reference implementation — every concept intentionally exposed, every switch demonstrable.*
