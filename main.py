import threading
from typing import Annotated, Any
from unittest import skip

from fastapi import FastAPI, Query, HTTPException  # type: ignore[reportUnknownVariableType]
from pydantic import BaseModel  # type: ignore[reportUnknownVariableType]

# RAG pipeline (LangChain + LangGraph + Anthropic) — see rag_pipeline.py
from rag_pipeline import document_store, ingest_graph, query_graph, add_documents

# Advanced RAG techniques — see rag_advanced.py
from rag_advanced import (
    parent_store, child_store, contextual_store,
    parent_ingest_graph,
    contextual_ingest_graph,
    advanced_query_graph,
    diagnostic_graph,
)



app: FastAPI = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello from fastapiKpi!"}


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    item: dict[str, str | bool] = {"item_id": item_id}
    if q:
        item["q"] = q
    if not short:
        item["description"] = "This is an amazing item that has a long description"
    item["short"] = short
    return item


@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):
    item: dict[str, str | int | bool] = {"item_id": item_id, "owner_id": user_id}
    if q:
        item["q"] = q
    if not short:
        item["description"] = "This is an amazing item that has a long description"
    item["short"] = short
    return item



@app.get("/newItems/{item_id}")
async def read_user_item(item_id: str, needy: str):
    item: dict[str, str] = {"item_id": item_id, "needy": needy}
    return item



@app.get("/multiItems/{item_id}")
async def read_user_item(
    item_id: str, needy: str, skip: int = 0, limit: int | None = None
):
    item: dict[str, str | int | None] = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}
    return item

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/items")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.put("/viewItems/{item_id}")
async def update_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result.update({"q": q})
    return result


@app.get("/itemsParam/")
async def read_items(q: str | None = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results




@app.get("/itemsABC/")
async def read_items(q: Annotated[str | None, Query(max_length=10)] = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/itemsNonAnnotated/")
async def read_items(q: str | None = Query(default=None, max_length=10)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/itemsValidation/")
async def read_items(
    q: Annotated[str | None, Query(min_length=3, max_length=10)] = None,
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results



@app.get("/itemsDefault/")
async def read_items(q: Annotated[str, Query(min_length=3)]):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results




@app.get("/itemsStillRequired/")
async def read_items(q: Annotated[str | None, Query(min_length=3)]):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results



@app.get("/itemsMultipleValue/")
async def read_items(q: Annotated[list[str] | None, Query()] = ["foo", "bar"]):
    query_items = {"q": q}
    return query_items


@app.get("/itemsMultipleValuev2/")
async def read_items(
    q: Annotated[str | None, Query(title="Query string", min_length=3)] = None,
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


class WorkspaceManager:

    def __init__(self):
        # Use an RLock because methods will recursively call each other
        self._lock = threading.RLock()
        self._configs: dict[str, dict] = {}

    def get_config(self, tenant_id: str) -> dict:
        with self._lock:  # Acquisition 1 (or standalone read)
            return self._configs.get(tenant_id, {"status": "default"})

    def update_config(self, tenant_id: str, new_data: dict):
        with self._lock:  # 1st acquisition for this thread
            # Mutate the primary config state
            current = self.get_config(
                tenant_id
            )  # <-- NESTED CALL! Tries to acquire the SAME lock.
            current.update(new_data)
            self._configs[tenant_id] = current

            # Trigger an internal sub-routine that also locks
            self._write_audit_log(tenant_id, "CONFIG_UPDATED")

    def _write_audit_log(self, tenant_id: str, action: str):
        with self._lock:  # 2nd/3rd nested acquisition by the SAME thread!
            # Safe because RLock increments its internal counter instead of blocking.
            print(
                f"[AUDIT] Tenant {tenant_id}: {action} logged successfully."
            )


# Instantiate our thread-safe manager
manager = WorkspaceManager()


class ConfigUpdatePayload(BaseModel):
    theme: str
    feature_flags: dict[str, bool]


@app.put("/workspace/{tenant_id}/config")
def update_workspace_endpoint(tenant_id: str, payload: ConfigUpdatePayload):
    try:
        # This will safely flow through update_config -> get_config -> _write_audit_log
        # all under the protection of the same RLock without deadlocking.
        manager.update_config(tenant_id, payload.model_dump())
        return {
            "status": "success",
            "config": manager.get_config(tenant_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    











def main():
    print("Hello from fastapiKpi!")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# RAG ENDPOINTS — LangChain + LangGraph + Anthropic
#
# Three endpoints demonstrate the RAG pipeline stages:
#   POST /rag/ingest  →  Stage 2 (Chunking)    via ingest_graph
#   POST /rag/query   →  Stage 3+4 (Retrieval + Generation) via query_graph
#   GET  /rag/status  →  inspect the document store
#
# The pipeline is already seeded with the RAG failure modes reference text
# (see rag_pipeline._RAG_SEED) so /rag/query works without calling /rag/ingest.
# ══════════════════════════════════════════════════════════════════════════════


class IngestRequest(BaseModel):
    text: str
    source: str = "unknown"


class QueryRequest(BaseModel):
    question: str


@app.post("/rag/ingest", tags=["RAG"])
async def rag_ingest(req: IngestRequest):
    """
    Ingest a document into the RAG system.

    Internally runs **ingest_graph**:
      START → chunk_node (RecursiveCharacterTextSplitter, 500 tok / 50 overlap) → END

    The resulting Document chunks are appended to the shared document_store.
    """
    result = ingest_graph.invoke({
        "raw_text": req.text,
        "metadata": {"source": req.source},
        "chunks": [],
    })
    add_documents(result["chunks"])   # updates both BM25 list and FAISS index
    return {
        "chunks_added": len(result["chunks"]),
        "total_chunks_in_store": len(document_store),
        "chunk_previews": [c.page_content[:80] + "…" for c in result["chunks"]],
    }


@app.post("/rag/query", tags=["RAG"])
async def rag_query(req: QueryRequest):
    """
    Query the RAG system with a natural-language question.

    Internally runs **query_graph**:
      START
        → retrieve_node  (BM25Retriever — sparse keyword search, k=4)
        → generate_node  (Claude Haiku via ChatAnthropic, temperature=0)
      END

    Returns Claude's answer plus the retrieved chunks used as context.
    """
    if not document_store:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. POST to /rag/ingest first.",
        )
    result = query_graph.invoke({
        "query": req.question,
        "retrieved_chunks": [],
        "answer": "",
    })
    return {
        "answer": result["answer"],
        "retrieved_chunks_count": len(result["retrieved_chunks"]),
        "retrieved_sources": [
            {
                "chunk_preview": chunk.page_content[:150] + "…",
                "source": chunk.metadata.get("source"),
            }
            for chunk in result["retrieved_chunks"]
        ],
    }


@app.get("/rag/status", tags=["RAG"])
async def rag_status():
    """Inspect the current state of the in-memory document store."""
    return {
        "total_chunks": len(document_store),
        "sources": sorted({c.metadata.get("source", "unknown") for c in document_store}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG ENDPOINTS — Diagnostic Protocol + 3 Retrieval Techniques
#
#   POST /rag/ingest/parent       Technique A — Parent-Document Retrieval
#   POST /rag/ingest/contextual   Technique B — Contextual Chunking (Claude)
#   POST /rag/query/advanced      Technique C — Query Rewriting + multi-BM25
#   POST /rag/diagnose            Diagnostic Protocol — trace failure stage
#   GET  /rag/stores              Overview of all document stores
#
# See rag_advanced.py for the LangGraph implementation of each.
# ══════════════════════════════════════════════════════════════════════════════


# ── Technique A: Parent-Document Retrieval ────────────────────────────────────

@app.post("/rag/ingest/parent", tags=["RAG Advanced"])
async def rag_ingest_parent(req: IngestRequest):
    """
    Ingest using **Parent-Document Retrieval** (Technique A).

    Runs **parent_ingest_graph**:
      START → split_parent_child_node → END

    Creates two chunk sizes from the same document:
      • Child chunks (128 tok) — stored in child_store, used for BM25 retrieval
      • Parent chunks (1024 tok) — stored in parent_store, fed to the LLM

    Use /rag/query/advanced to query — it automatically detects parent_id
    metadata and swaps child hits for their parent sections.
    """
    result = parent_ingest_graph.invoke({
        "raw_text": req.text,
        "metadata": {"source": req.source},
        "parent_count": 0,
        "child_count": 0,
    })
    return {
        "technique": "parent_document_retrieval",
        "parent_chunks_created": result["parent_count"],
        "child_chunks_created": result["child_count"],
        "total_child_chunks": len(child_store),
        "total_parent_sections": len(parent_store),
    }


# ── Technique B: Contextual Chunking ─────────────────────────────────────────

@app.post("/rag/ingest/contextual", tags=["RAG Advanced"])
async def rag_ingest_contextual(req: IngestRequest):
    """
    Ingest using **Contextual Chunking** (Technique B).

    Runs **contextual_ingest_graph**:
      START → plain_chunk_node → contextualize_node → END

    For each chunk, Claude Haiku prepends a 1-sentence context summary
    so the chunk retains global document meaning in isolation.

    Note: makes 1 Claude API call per chunk — expect a few seconds latency
    for documents with many chunks.

    Use /rag/query/advanced (or /rag/query) to query contextual_store.
    """
    result = contextual_ingest_graph.invoke({
        "raw_text": req.text,
        "metadata": {"source": req.source},
        "raw_chunks": [],
        "contextualized_chunks": [],
        "chunk_count": 0,
    })
    return {
        "technique": "contextual_chunking",
        "chunks_enriched": result["chunk_count"],
        "total_contextual_chunks": len(contextual_store),
        "sample_enriched_chunk": (
            result["contextualized_chunks"][0].page_content[:250] + "…"
            if result["contextualized_chunks"] else None
        ),
    }


# ── Technique C: Query Rewriting + Advanced Query ─────────────────────────────

class AdvancedQueryRequest(BaseModel):
    question: str
    store: str = "default"   # "default" | "child" | "contextual"


@app.post("/rag/query/advanced", tags=["RAG Advanced"])
async def rag_query_advanced(req: AdvancedQueryRequest):
    """
    Query using **Query Rewriting + Expansion** (Technique C).

    Runs **advanced_query_graph**:
      START → rewrite_query_node (Claude — 3 query variants)
            → multi_retrieve_node (BM25 × 3 queries, merged + deduplicated)
            → [parent_lookup_node if child chunks detected]
            → generate_node_advanced (Claude)
      END

    The `store` parameter selects which document store to search:
      • "default"    — baseline document_store (from /rag/ingest)
      • "child"      — child_store (from /rag/ingest/parent)
      • "contextual" — contextual_store (from /rag/ingest/contextual)

    When "child" is selected, parent_lookup_node automatically activates
    (via conditional edge in the graph) and feeds parent sections to the LLM.
    """
    stores = {
        "default":    document_store,
        "child":      child_store,
        "contextual": contextual_store,
    }
    chunks = stores.get(req.store, document_store)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"Store '{req.store}' is empty. Ingest documents first.",
        )

    result = advanced_query_graph.invoke({
        "query": req.question,
        "all_chunks": chunks,
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "final_chunks": [],
        "answer": "",
    })
    return {
        "answer": result["answer"],
        "rewritten_queries": result["rewritten_queries"],
        "retrieved_chunks_count": len(result["retrieved_chunks"]),
        "final_chunks_count": len(result.get("final_chunks") or []),
        "used_parent_lookup": any(
            "parent_id" in c.metadata for c in result["retrieved_chunks"]
        ),
        "retrieved_sources": [
            {
                "preview": c.page_content[:120] + "…",
                "source": c.metadata.get("source"),
                "has_context_prefix": c.metadata.get("has_context_prefix", False),
            }
            for c in result["retrieved_chunks"]
        ],
    }


# ── Diagnostic Protocol ───────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    question: str


@app.post("/rag/diagnose", tags=["RAG Advanced"])
async def rag_diagnose(req: DiagnoseRequest):
    """
    **Diagnostic Protocol** — trace a bad answer backward through the pipeline.

    Runs **diagnostic_graph**:
      START → retrieve → generate
            → check_retrieval_node  (Claude: was answer in top-K?)
            → check_grounding_node  (Claude: did LLM hallucinate?)
            → diagnose_node         (deterministic failure classification)
      END

    Diagnostic matrix:
      Grounded + any retrieval    → No failure detected
      Hallucinated + retrieval=YES → Lost-in-middle / context clutter
      Hallucinated + retrieval=NO  → Retrieval or ingestion failure
      Partially grounded          → Partial retrieval hit

    Returns the failure stage, diagnosis, and concrete recommendations.
    """
    if not document_store:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. POST to /rag/ingest first.",
        )

    result = diagnostic_graph.invoke({
        "query": req.question,
        "all_chunks": document_store,
        "retrieved_chunks": [],
        "answer": "",
        "retrieval_verdict": "",
        "grounding_verdict": "",
        "failure_stage": "",
        "diagnosis": "",
        "recommendations": [],
    })
    return {
        "question": req.question,
        "generated_answer": result["answer"],
        "diagnostic": {
            "retrieval_verdict": result["retrieval_verdict"],
            "grounding_verdict": result["grounding_verdict"],
            "failure_stage": result["failure_stage"],
            "diagnosis": result["diagnosis"],
            "recommendations": result["recommendations"],
        },
        "retrieved_chunks": [
            {
                "preview": c.page_content[:120] + "…",
                "source": c.metadata.get("source"),
            }
            for c in result["retrieved_chunks"]
        ],
    }


# ── Store overview ─────────────────────────────────────────────────────────────

@app.get("/rag/stores", tags=["RAG Advanced"])
async def rag_stores():
    """Overview of all document stores and their chunk counts."""
    return {
        "baseline": {
            "store": "document_store",
            "chunks": len(document_store),
            "endpoint": "POST /rag/ingest",
        },
        "parent_document": {
            "child_store_chunks": len(child_store),
            "parent_store_sections": len(parent_store),
            "endpoint": "POST /rag/ingest/parent",
        },
        "contextual": {
            "store": "contextual_store",
            "chunks": len(contextual_store),
            "endpoint": "POST /rag/ingest/contextual",
        },
    }
