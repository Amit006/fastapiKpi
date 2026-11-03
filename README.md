# RAG Pipeline — FastAPI + LangChain + LangGraph + Anthropic

A working, stage-by-stage RAG pipeline built for learning. Every node in the graph is annotated with the failure mode it addresses from the RAG Failure Modes reference guide.

---

## Architecture

```
POST /rag/ingest
  │
  └─► ingest_graph  (LangGraph)
        START
          └─► chunk_node          Stage 2 — RecursiveCharacterTextSplitter
        END

POST /rag/query
  │
  └─► query_graph   (LangGraph)
        START
          └─► retrieve_node       Stage 3 — BM25Retriever (sparse / lexical)
                └─► generate_node Stage 4 — ChatAnthropic  (Claude Haiku, temp=0)
        END
```

### Stage-by-Stage

| # | Stage | Component | Failure Mode Addressed |
|---|-------|-----------|----------------------|
| 2 | Chunking | `RecursiveCharacterTextSplitter` (500 tok, 50 overlap) | Answers Split Across Chunks · No Metadata |
| 3 | Retrieval | `BM25Retriever` (rank_bm25, k=4) | No Hybrid Search *(shown)* · Top-K Miscalibration |
| 4 | Generation | `ChatAnthropic` claude-haiku-4-5, temperature=0 | Ignoring Context · Conflicting Chunks |

> **Stage 1 (Ingestion)** and the production upgrade path (dense embeddings, hybrid RRF, cross-encoder re-ranker) are called out in comments throughout the code but not implemented — they require additional services (embedding API, vector DB, re-ranker API).

---

## Quick Start

### 1. Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)

### 2. Install dependencies

```bash
cd fastapiKpi
uv sync
```

### 3. Set your API key

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Start the server

```bash
uv run fastapi dev main.py
```

The server starts at `http://localhost:8000`. Open the interactive docs at `http://localhost:8000/docs`.

---

## Loading Sample Data

The pipeline is **pre-seeded** with the RAG Failure Modes reference text on startup (9 chunks) — `/rag/query` works immediately.

To load richer data from arXiv and Wikipedia, run the sample data loader while the server is running:

```bash
# In a second terminal (server must be running)
uv run python scripts/load_sample_data.py
```

This fetches:
- **10 arXiv paper abstracts** on RAG / LLM topics (no API key needed)
- **5 Wikipedia articles**: RAG, LLMs, Vector Databases, BM25, Transformers

After loading (~15 seconds), the document store grows from 9 to ~60+ chunks.

### Kaggle datasets (optional)

```bash
# Install the Kaggle CLI and configure credentials
pip install kaggle
# Place kaggle.json in ~/.kaggle/  (download from kaggle.com → Account → API)

# Then download any text dataset, e.g. the "News Category Dataset"
kaggle datasets download -d rmisra/news-category-dataset --unzip -p data/

# Ingest via the API
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "<article text here>", "source": "kaggle:news-category"}'
```

---

## API Reference

### `GET /rag/status`

Returns the current state of the in-memory document store.

```json
{
  "total_chunks": 9,
  "sources": ["rag_failure_modes_reference"]
}
```

---

### `POST /rag/ingest`

Chunk and store a document (runs `ingest_graph`).

**Request body:**
```json
{
  "text": "Your document text here...",
  "source": "my-document-v1"
}
```

**Response:**
```json
{
  "chunks_added": 4,
  "total_chunks_in_store": 13,
  "chunk_previews": [
    "Your document text here. This is the first 80 chars of…",
    "...continuing chunk..."
  ]
}
```

---

### `POST /rag/query`

Retrieve relevant chunks via BM25 and generate an answer with Claude (runs `query_graph`).

**Request body:**
```json
{
  "question": "What is the mitigation for No Hybrid Search?"
}
```

**Response:**
```json
{
  "answer": "According to [Chunk 1], the mitigation for 'No Hybrid Search' is to implement hybrid search that fuses BM25 sparse lexical search with dense vector search via Reciprocal Rank Fusion (RRF)...",
  "retrieved_chunks_count": 4,
  "retrieved_sources": [
    {
      "chunk_preview": "No Hybrid Search: Pure dense retrieval misses exact keyword and ID lookups...",
      "source": "rag_failure_modes_reference"
    }
  ]
}
```

---

## Example Queries

After loading sample data, try these questions at `POST /rag/query`:

```json
{ "question": "What causes answers to be split across chunks?" }
{ "question": "How does BM25 scoring work?" }
{ "question": "What is the lost-in-the-middle problem?" }
{ "question": "What failure modes exist in the ingestion stage?" }
{ "question": "How do cross-encoder re-rankers improve retrieval precision?" }
{ "question": "What is retrieval augmented generation?" }
{ "question": "How does a vector database store embeddings?" }
```

---

## Extending the Pipeline

The pipeline is intentionally minimal — each upgrade maps to a failure mode from the reference guide.

### Add dense vector retrieval (Stage 3 upgrade)

```python
# In rag_pipeline.py — replace BM25Retriever with hybrid search

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings)

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
bm25_retriever = BM25Retriever.from_documents(chunks, k=20)

# Reciprocal Rank Fusion — the hybrid search fix
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5],
)
```

### Add a re-ranker (Stage 3 second pass)

```python
# Requires: pip install cohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,
)
```

### Upgrade to Claude Sonnet for better generation

```python
# In generate_node — swap model
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
```

---

## Project Structure

```
fastapiKpi/
├── main.py              # FastAPI app — existing endpoints + /rag/* endpoints
├── rag_pipeline.py      # LangGraph graphs: ingest_graph + query_graph
├── scripts/
│   └── load_sample_data.py   # Fetch arXiv + Wikipedia data, POST to /rag/ingest
├── pyproject.toml       # uv project — dependencies
└── README.md
```

---

## Dependencies

| Package | Role |
|---------|------|
| `fastapi[standard]` | HTTP API framework |
| `langchain` | Document primitives, text splitters |
| `langchain-anthropic` | `ChatAnthropic` LLM wrapper |
| `langchain-community` | `BM25Retriever` |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langgraph` | State machine / pipeline orchestration |
| `rank-bm25` | BM25 scoring algorithm |
