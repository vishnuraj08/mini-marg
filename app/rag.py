"""
rag.py — FAISS Ingestion + Hybrid Search (BM25 + Vector + RRF)

FLOW:
  ingest()  → load docs → inject config → chunk → embed → FAISS index → save
  search()  → embed query → FAISS search + BM25 search → RRF merge → top-k chunks
"""

import os
import json
import yaml
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # MiniMarg/
DOCS_DIR    = BASE_DIR / "data" / "docs"
CONFIG_PATH = BASE_DIR / "config" / "rates.yaml"
INDEX_DIR   = BASE_DIR / "faiss_index"

INDEX_FILE  = INDEX_DIR / "index.faiss"             # the FAISS binary index
META_FILE   = INDEX_DIR / "metadata.pkl"            # chunk texts + source names
BM25_FILE   = INDEX_DIR / "bm25.pkl"                # BM25 model (for hybrid search)

# ─────────────────────────────────────────────
# EMBEDDINGS MODEL (Ollama nomic-embed-text)
# ─────────────────────────────────────────────
# nomic-embed-text produces 768-dim vectors
# OllamaEmbeddings calls the local Ollama server at http://localhost:11434
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG LOADER (rates.yaml → flat dict for templates)
# ═══════════════════════════════════════════════════════════════

def load_config() -> Dict:
    """Load rates.yaml and return raw nested dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def flatten_config(config: Dict, prefix: str = "", result: Dict = None) -> Dict:
    """
    Convert nested YAML → flat keys for string.format()
    
    Input:  {"personal_loan": {"min_amount": "INR 50,000"}}
    Output: {"personal_loan.min_amount": "INR 50,000"}
    
    This lets us do: template.format(**flat_config)
    which replaces {personal_loan.min_amount} in the .txt files.
    """
    if result is None:
        result = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten_config(value, full_key, result)
        else:
            result[full_key] = str(value)
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DOCUMENT LOADER (reads .txt files, injects config)
# ═══════════════════════════════════════════════════════════════

def load_documents() -> List[Dict]:
    """
    Load all .txt files from data/docs/
    Inject config values into placeholders like {personal_loan.min_amount}
    
    Returns list of: {"content": "...", "source": "loans.txt"}
    """
    flat_config = flatten_config(load_config())
    documents = []

    for txt_file in DOCS_DIR.glob("*.txt"):
        with open(txt_file, encoding="utf-8") as f:
            template = f.read()

        # Inject all {placeholder} values from rates.yaml
        try:
            content = template.format(**flat_config)
        except KeyError as e:
            # If a placeholder has no matching config key, warn and use raw text
            print(f"WARNING: Missing config key {e} in {txt_file.name}, using raw text")
            content = template

        documents.append({
            "content": content,
            "source": txt_file.name
        })
        print(f"  Loaded: {txt_file.name} ({len(content)} chars)")

    return documents


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CHUNKING (RecursiveCharacterTextSplitter)
# ═══════════════════════════════════════════════════════════════

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split documents into chunks.
    
    WHY RecursiveCharacterTextSplitter?
    It tries to split on: paragraph → sentence → word → character
    So it never cuts mid-sentence (unlike FixedSizeChunker).
    
    chunk_size=500   → each chunk is ~500 chars
    chunk_overlap=50 → 50 chars shared between consecutive chunks
                       (so context doesn't get lost at boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        raw_chunks = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "chunk_id": i
            })

    print(f"  Total chunks created: {len(chunks)}")
    return chunks


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — EMBEDDING + FAISS INDEX BUILDING
# ═══════════════════════════════════════════════════════════════

def build_faiss_index(chunks: List[Dict]) -> faiss.Index:
    """
    Convert each chunk to a vector using nomic-embed-text.
    Build a FAISS Flat L2 index (exact search, no approximation).
    
    FAISS IndexFlatL2:
      - Flat = stores all vectors, no compression
      - L2   = Euclidean distance (smaller = more similar)
      - Good for our small dataset (<10k chunks)
      - For millions of docs, use IndexIVFFlat (approximate, faster)
    """
    texts = [c["text"] for c in chunks]

    print("  Embedding chunks with nomic-embed-text... (takes ~30 seconds)")
    vectors = embeddings_model.embed_documents(texts)  # List[List[float]]
    vectors_np = np.array(vectors, dtype=np.float32)   # FAISS needs numpy float32

    dimension = vectors_np.shape[1]  # nomic-embed-text = 768 dims
    print(f"  Embedding dimension: {dimension}")

    # Build index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)                              # add all vectors to index
    print(f"  FAISS index built: {index.ntotal} vectors")

    return index


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — BM25 INDEX BUILDING (for hybrid search)
# ═══════════════════════════════════════════════════════════════

def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """
    BM25 = keyword-based retrieval (TF-IDF variant).
    
    BM25Okapi is the standard formula:
      score(q, d) = Σ IDF(qi) × (TF(qi,d) × (k1+1)) / (TF(qi,d) + k1 × (1 - b + b × |d|/avgdl))
    
    Tokenization: we just split on spaces (good enough for our case).
    In production: use NLTK or spaCy tokenizer for better results.
    """
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25 index built: {len(tokenized)} documents")
    return bm25


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — SAVE / LOAD INDEX
# ═══════════════════════════════════════════════════════════════

def save_index(index: faiss.Index, chunks: List[Dict], bm25: BM25Okapi):
    """Save FAISS index, chunk metadata, and BM25 model to disk."""
    INDEX_DIR.mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"  FAISS index saved: {INDEX_FILE}")

    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Metadata saved: {META_FILE}")

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 model saved: {BM25_FILE}")


def load_index() -> Tuple[faiss.Index, List[Dict], BM25Okapi]:
    """Load FAISS index, chunk metadata, and BM25 model from disk."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError("Index not found. Call POST /ingest first.")

    index  = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    return index, chunks, bm25


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — RRF HYBRID FUSION
# ═══════════════════════════════════════════════════════════════

def rrf_merge(vector_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion — combines two ranked lists into one.
    
    Formula: score(doc) = Σ 1 / (k + rank)
    
    WHY k=60? It's the standard constant that prevents top-1 from dominating.
    A doc ranked #1 gets 1/(60+1) = 0.016 from each retriever.
    A doc ranked #10 gets 1/(60+10) = 0.014 — close but lower.
    
    This way even if BM25 ranks a doc at #3 and vector ranks it #5,
    it beats a doc that only one retriever found at #1.
    """
    scores: Dict[int, float] = {}

    for rank, doc_idx in enumerate(vector_ranks):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_idx in enumerate(bm25_ranks):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    # Sort by combined score, highest first
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MAIN INGEST FUNCTION (called by POST /ingest)
# ═══════════════════════════════════════════════════════════════

def ingest() -> Dict:
    """
    Full ingestion pipeline:
    load docs → inject config → chunk → embed → FAISS + BM25 → save
    
    Called once to build the index. Re-call whenever docs or config change.
    """
    print("\n=== INGESTION PIPELINE START ===")

    print("\n[1/4] Loading and parametrizing documents...")
    documents = load_documents()

    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)

    print("\n[3/4] Building FAISS index...")
    index = build_faiss_index(chunks)

    print("\n[4/4] Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    save_index(index, chunks, bm25)

    print("\n=== INGESTION COMPLETE ===\n")
    return {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "index_size": index.ntotal
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MAIN SEARCH FUNCTION (called by tools.py)
# ═══════════════════════════════════════════════════════════════

def search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Hybrid search: BM25 + FAISS → RRF merge → top_k chunks
    
    Steps:
    1. Embed the query using nomic-embed-text
    2. Search FAISS (vector similarity) → get top 10 chunk indices
    3. Search BM25 (keyword match) → get top 10 chunk indices
    4. RRF merge both lists → reranked indices
    5. Return top_k chunks with text + source
    """
    index, chunks, bm25 = load_index()

    # --- Vector search ---
    query_vector = embeddings_model.embed_query(query)
    query_np = np.array([query_vector], dtype=np.float32)

    # FAISS returns distances and indices of nearest neighbors
    # We search top 10 then re-rank, not just top_k directly
    distances, faiss_indices = index.search(query_np, min(10, len(chunks)))
    vector_ranks = faiss_indices[0].tolist()  # list of chunk indices

    # --- BM25 search ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Sort chunk indices by BM25 score, highest first
    bm25_ranks = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:10]

    # --- RRF Fusion ---
    merged_ranks = rrf_merge(vector_ranks, bm25_ranks)

    # --- Build results ---
    results = []
    for idx in merged_ranks[:top_k]:
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"]
        })

    return results