# cspell:ignore FAISS

"""
RAG Pipeline — LangChain + LangGraph + Anthropic

Maps to the RAG Failure Modes stages:

  Stage 2 – Chunking   → chunk_node    (RecursiveCharacterTextSplitter)
  Stage 3 – Retrieval  → retrieve_node (FAISS dense  +  BM25 sparse  →  RRF hybrid)
  Stage 4 – Generation → generate_node (ChatAnthropic / Claude Haiku)

Two compiled LangGraph graphs:
  ingest_graph  :  START → chunk    → END
  query_graph   :  START → retrieve → generate → END

Vector store:
  Dense index  — FAISS (faiss-cpu) with HuggingFace all-MiniLM-L6-v2 embeddings
                 (384-dim, ~90 MB download on first use, cached in ~/.cache/huggingface)
  Sparse index — BM25 (rank_bm25), rebuilt per query from document_store list
  Fusion       — EnsembleRetriever (Reciprocal Rank Fusion, weights 0.5 / 0.5)

The module-level stores are shared with main.py's endpoints.
Pre-seeded with the RAG failure modes reference text on import.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ── Embeddings ─────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dimensional sentence embeddings, ~90 MB download.
# Model is cached in ~/.cache/huggingface/ after first use.
# Swap for a domain-adapted model (e.g. voyage-code-2, medicalai/ClinicalBERT)
# to fix the 'Embedding Mismatch' failure mode for specialist domains.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Document stores ────────────────────────────────────────────────────────────
# document_store  — raw list kept for BM25 (BM25Retriever.from_documents needs it)
# _faiss_store    — FAISS dense index; None until the first document is ingested
document_store: list[Document] = []
_faiss_store: FAISS | None = None


def _rrf_merge(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion — merge multiple ranked retrieval lists.

    For each document, its RRF score is the sum across all lists of:
        1 / (k + rank)   where rank is 1-indexed, k=60 is the smoothing constant.

    Documents appearing in multiple lists accumulate score from each.
    Result is sorted descending by total RRF score.

    This is the same algorithm used internally by LangChain's EnsembleRetriever.
    """
    scores: dict[str, tuple[float, Document]] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            prev_score, _ = scores.get(key, (0.0, doc))
            scores[key] = (prev_score + 1.0 / (k + rank), doc)
    return [doc for _, doc in sorted(scores.values(), key=lambda x: x[0], reverse=True)]


def add_documents(docs: list[Document]) -> None:
    """
    Add chunks to both the BM25 list and the FAISS dense index.

    Call this instead of document_store.extend() so both stores stay in sync.
    On the first call, FAISS.from_documents() builds a fresh index.
    Subsequent calls use FAISS.add_documents() to append incrementally.
    """
    global _faiss_store
    if not docs:
        return
    document_store.extend(docs)
    if _faiss_store is None:
        _faiss_store = FAISS.from_documents(docs, embeddings)
    else:
        _faiss_store.add_documents(docs)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — INGESTION   Stage 2: Chunking
# Flow: START → chunk → END
# ══════════════════════════════════════════════════════════════════════════════

class IngestionState(TypedDict):
    raw_text: str          # full document text (input)
    metadata: dict         # provenance: source, date, author (input)
    chunks: list[Document] # populated by chunk_node (output)


def chunk_node(state: IngestionState) -> dict:
    """
    Stage 2 — Chunking.

    Failure mode AVOIDED: 'Answers Split Across Chunks'
      Fixed-size splitters cut sentences mid-thought.
      RecursiveCharacterTextSplitter tries these boundaries in priority order:
        paragraph (\n\n) → line (\n) → sentence (". ") → word (" ") → char ("")
      So semantic units stay together as long as they fit the chunk_size.

    Failure mode AVOIDED: 'No Metadata'
      metadata from the source document is propagated to every child chunk,
      preserving provenance (source, date) for citation in Stage 4.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,                            # overlap bridges boundary gaps
        separators=["\n\n", "\n", ". ", " ", ""],    # tried left-to-right
    )
    doc = Document(page_content=state["raw_text"], metadata=state["metadata"])
    chunks = splitter.split_documents([doc])
    return {"chunks": chunks}


def _build_ingest_graph():
    g = StateGraph(IngestionState)
    g.add_node("chunk", chunk_node)
    g.add_edge(START, "chunk")
    g.add_edge("chunk", END)
    return g.compile()


ingest_graph = _build_ingest_graph()


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — QUERY   Stage 3: Retrieval  +  Stage 4: Generation
# Flow: START → retrieve → generate → END
# ══════════════════════════════════════════════════════════════════════════════

class QueryState(TypedDict):
    query: str                        # user question (input)
    retrieved_chunks: list[Document]  # top-K after retrieval (intermediate)
    answer: str                       # Claude's response (output)


def retrieve_node(state: QueryState) -> dict:
    """
    Stage 3 — Hybrid retrieval: FAISS dense + BM25 sparse, fused via RRF.

    WHY HYBRID?
      Dense (FAISS):  finds semantically similar text even when exact words differ.
                      Good for: conceptual questions, synonyms, paraphrases.
      Sparse (BM25):  scores by term frequency × inverse document frequency.
                      Good for: exact keyword/ID matches, error codes, proper nouns.

      Reciprocal Rank Fusion (RRF) re-ranks by combining positions from both lists:
        score(doc) = Σ  1 / (k + rank_i(doc))   where k=60 (LangChain default)
      This is the fix for the 'No Hybrid Search' failure mode.

    Failure mode FIXED: 'No Hybrid Search'
      Previous implementation used BM25 only. This now fuses dense + sparse,
      the production upgrade described in the RAG failure modes reference.

    Failure mode MITIGATED: 'Top-K Miscalibration'
      Each retriever fetches k=6; after RRF deduplication we cap at 5.
      The cap prevents context window pollution / lost-in-the-middle.

    Falls back to BM25-only when _faiss_store is None (before first ingest).
    """
    if not document_store:
        return {"retrieved_chunks": []}

    bm25_hits = BM25Retriever.from_documents(document_store, k=6).invoke(state["query"])

    if _faiss_store is not None:
        faiss_hits = _faiss_store.similarity_search(state["query"], k=6)
        # Fuse both ranked lists with Reciprocal Rank Fusion
        retrieved = _rrf_merge([bm25_hits, faiss_hits])
    else:
        retrieved = bm25_hits

    return {"retrieved_chunks": retrieved[:5]}   # cap to avoid context window pollution


def generate_node(state: QueryState) -> dict:
    """
    Stage 4 — Generation using Claude Haiku (via langchain-anthropic).

    Failure mode MITIGATED: 'Ignoring Context'
      System prompt explicitly constrains Claude to the provided chunks.
      temperature=0 minimises hallucination (deterministic sampling).

    Failure mode MITIGATED: 'Conflicting Chunks'
      Each chunk is labelled with its source so Claude can surface conflicts
      rather than silently choosing one version.
    """
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    # Build the context string — each chunk labelled for citation
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1} | source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for i, doc in enumerate(state["retrieved_chunks"])
    )

    messages = [
        SystemMessage(content=(
            "You are a RAG architecture expert. "
            "Answer the user's question using ONLY the context chunks below. "
            "Cite the chunk number(s) you draw from, e.g. '[Chunk 2]'. "
            "If the answer is not present in the context, respond with exactly: "
            "'I don't know — the provided context does not cover this.'"
        )),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {state['query']}"
        ),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


def _build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


query_graph = _build_query_graph()


# ── Seed document ──────────────────────────────────────────────────────────────
# The RAG failure modes reference from the learning material.
# Pre-ingested so /rag/query works immediately without calling /rag/ingest.

_RAG_SEED = """
Stage 1 - Ingestion Failures:

Bad PDF Parsing and Tables Lost: Complex multi-column layouts, header/footer artifacts, and raw PyPDF text extraction miss tables and charts. Use multimodal parsers like Amazon Textract, LlamaParse, or unstructured that preserve table HTML/Markdown structure and suppress boilerplate.

Duplicates and Redundancy: Ingesting multiple revisions or overlapping sections of the same corpus produces noisy retrieval results. Implement semantic deduplication using MinHash/LSH or embedding cosine thresholding during ingestion pipelines.

Stage 2 - Chunking Failures:

Answers Split Across Chunks: Fixed-size chunking of 500 tokens with 50 overlap can cut off sentences mid-thought or split context. Use semantic chunking that splits on embedding distance/cosine dissimilarity drops, or use RecursiveCharacterTextSplitter with larger context boundaries.

Chunks Too Big or Too Small: Too big dilutes exact answers with noise. Too small lacks sufficient surrounding context. Aim for 256-512 token sweet spots combined with Parent-Document Retrieval which searches small child chunks and returns the parent context.

No Metadata: Raw chunks that lack provenance like title, section, date, and author lead to blind retrieval without citation ability. Enrich every chunk with contextual headers, document titles, and temporal tags.

Stage 3 - Retrieval Failures:

Embedding Mismatch: General-purpose embedding models fail on niche legal, medical, or code domains because they lack domain vocabulary. Fine-tune embedding models via Matryoshka Representation Learning or sentence-transformers, or switch to domain-adapted retrievers like Voyage AI.

No Hybrid Search: Pure dense retrieval misses exact keyword and ID lookups like error codes and part numbers. Implement hybrid search fusing BM25 sparse lexical search and dense vector search via Reciprocal Rank Fusion (RRF).

No Re-ranker: Relying purely on embedding vector cosine distance prioritises semantic angle over exact precision. Add a cross-encoder re-ranker like Cohere Rerank or BGE-Reranker as a second-stage filter on the top 50 chunks to narrow down to the top 5.

Top-K Miscalibration: Top-K too low misses relevant data. Top-K too high causes context window pollution and the lost-in-the-middle effect where the LLM ignores middle chunks. Use dynamic top-K or threshold-based filtering that discards chunks below a similarity score cutoff.

Stage 4 - Generation Failures:

Ignoring Context: The LLM hallucinates or relies on pre-training weights instead of the provided context. Tune the system prompt to instruct the model to answer only using the provided context, state uncertainty rather than guessing, and lower temperature to 0.

Conflicting Chunks: Retrieved chunks contain contradictory statements from different document versions. Pass chunk metadata timestamps to the LLM so it can prefer the most recent source, or run a conflict-resolution summarisation step before final generation.
"""


def _seed_document_store() -> None:
    result = ingest_graph.invoke({
        "raw_text": _RAG_SEED,
        "metadata": {"source": "rag_failure_modes_reference"},
        "chunks": [],
    })
    add_documents(result["chunks"])   # populates both document_store and _faiss_store


_seed_document_store()
