"""
Advanced RAG Techniques — LangGraph + LangChain + Anthropic

Builds on the baseline pipeline (rag_pipeline.py) with:

  Technique A — Parent-Document Retrieval
    Index 128-token child chunks for precise BM25 matching.
    Retrieve the matched child, then return its 1024-token parent
    section to the LLM for richer generation context.

  Technique B — Contextual Chunking  (Anthropic's contextual retrieval)
    Use Claude Haiku to prepend a 1-sentence document context to each
    chunk before storing, so isolated chunks retain global meaning.

  Technique C — Query Rewriting / Expansion
    Use Claude Haiku to rewrite the user query into 3 distinct search
    queries (acronyms, synonyms, sub-questions), run BM25 on each,
    then merge and deduplicate the results.

  Diagnostic Protocol
    Trace a bad answer backward through the pipeline:
      1. Was the right chunk in the top-K results?   → check retrieval
      2. Did the LLM use the context correctly?       → check grounding
      3. Classify failure stage and emit recommendations.

LangGraph graphs in this module:
  parent_ingest_graph      START → split_parent_child   → END
  contextual_ingest_graph  START → plain_chunk → contextualize → END
  advanced_query_graph     START → rewrite_query → multi_retrieve
                                 → [parent_lookup | generate] → generate → END
  diagnostic_graph         START → retrieve → generate
                                 → check_retrieval → check_grounding
                                 → diagnose → END
"""

from __future__ import annotations

import hashlib
from typing import TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# ── Stores for advanced techniques ───────────────────────────────────────────
#
# parent_store:     large (1024-tok) sections keyed by parent_id
#                   → looked up after child retrieval, fed to LLM
# child_store:      small (128-tok) search chunks, each has metadata["parent_id"]
#                   → used for BM25 retrieval
# contextual_store: chunks with Claude-generated context prefix
#                   → used for BM25 retrieval (richer keyword surface)
#
parent_store: dict[str, Document] = {}
child_store: list[Document] = []
contextual_store: list[Document] = []


# ── Shared LLM instance ───────────────────────────────────────────────────────
# Claude Haiku: fast and cheap — ideal for the helper calls in Techniques B & C
# and the grounding / retrieval checks in the Diagnostic Protocol.
_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)


# ── Utility ───────────────────────────────────────────────────────────────────

def _content_hash(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def _dedup(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by content hash, preserving order."""
    seen: set[str] = set()
    unique = []
    for d in docs:
        h = _content_hash(d)
        if h not in seen:
            seen.add(h)
            unique.append(d)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE A — PARENT-DOCUMENT RETRIEVAL
#
# Ingest graph:  START → split_parent_child → END
# Query path:    advanced_query_graph uses parent_lookup_node (conditional)
# ══════════════════════════════════════════════════════════════════════════════

class ParentIngestState(TypedDict):
    raw_text: str
    metadata: dict
    parent_count: int   # populated by node
    child_count: int    # populated by node


def split_parent_child_node(state: ParentIngestState) -> dict:
    """
    Technique A — Parent-Document ingestion.

    WHY TWO SIZES?
      Retrieval precision vs context richness trade-off:
        • 128-token child chunks → short, focused, high BM25 precision
          (fewer unrelated terms means the BM25 score is less diluted)
        • 1024-token parent sections → wide context window for the LLM
          (the LLM needs surrounding sentences to reason about the answer)

    Each child chunk carries metadata["parent_id"] pointing to its parent
    in parent_store, enabling a post-retrieval parent lookup step.
    """
    source = state["metadata"].get("source", "unknown")
    doc = Document(page_content=state["raw_text"], metadata=state["metadata"])

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=128,  chunk_overlap=20)

    parents = parent_splitter.split_documents([doc])
    child_count = 0

    for i, parent in enumerate(parents):
        parent_id = f"{source}__parent_{i}"
        parent_store[parent_id] = parent       # store for later lookup

        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child_store.append(child)
            child_count += 1

    return {"parent_count": len(parents), "child_count": child_count}


def _build_parent_ingest_graph():
    g = StateGraph(ParentIngestState)
    g.add_node("split_parent_child", split_parent_child_node)
    g.add_edge(START, "split_parent_child")
    g.add_edge("split_parent_child", END)
    return g.compile()


parent_ingest_graph = _build_parent_ingest_graph()


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE B — CONTEXTUAL CHUNKING
#
# Graph: START → plain_chunk → contextualize → END
# ══════════════════════════════════════════════════════════════════════════════

class ContextualIngestState(TypedDict):
    raw_text: str
    metadata: dict
    raw_chunks: list[Document]            # after plain chunking
    contextualized_chunks: list[Document] # after Claude enrichment
    chunk_count: int


def plain_chunk_node_b(state: ContextualIngestState) -> dict:
    """Stage 2 chunking — same parameters as baseline pipeline."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=state["raw_text"], metadata=state["metadata"])
    return {"raw_chunks": splitter.split_documents([doc])}


def contextualize_node(state: ContextualIngestState) -> dict:
    """
    Technique B — Contextual Chunking (Anthropic's contextual retrieval).

    For each chunk, Claude Haiku writes one sentence that situates the
    excerpt within its broader document. This sentence is prepended to
    the chunk content before it is stored and indexed.

    EXAMPLE — without context:
      "The error code is 0x80070005."
      → BM25 matches: "error code 0x80070005"

    EXAMPLE — with context:
      "[Context: This is from an Azure RBAC guide on access-denied errors.]
       The error code is 0x80070005."
      → BM25 also matches: "Azure", "RBAC", "access-denied", "guide"

    The context prefix dramatically improves recall for queries that use
    domain terminology or acronyms not present in the raw chunk text.

    Cost: 1 Claude Haiku call per chunk. For a 10-chunk document ≈ 10 calls.
    Production note: run these in parallel with asyncio.gather() or a batch.
    """
    # Give Claude the opening of the full document as broad context
    doc_preview = state["raw_text"][:400].strip()

    enriched: list[Document] = []
    for chunk in state["raw_chunks"]:
        messages = [
            SystemMessage(content=(
                "You are a document analysis assistant. "
                "Write exactly ONE sentence (max 30 words) that describes "
                "what broader topic this excerpt belongs to, so it can be "
                "understood without the surrounding document."
            )),
            HumanMessage(content=(
                f"Document opening:\n\"{doc_preview}...\"\n\n"
                f"Excerpt to describe:\n{chunk.page_content[:400]}"
            )),
        ]
        context_sentence = _llm.invoke(messages).content.strip()

        enriched_content = f"[Context: {context_sentence}]\n\n{chunk.page_content}"
        enriched_chunk = Document(
            page_content=enriched_content,
            metadata={**chunk.metadata, "has_context_prefix": True},
        )
        enriched.append(enriched_chunk)
        contextual_store.append(enriched_chunk)

    return {"contextualized_chunks": enriched, "chunk_count": len(enriched)}


def _build_contextual_ingest_graph():
    g = StateGraph(ContextualIngestState)
    g.add_node("plain_chunk", plain_chunk_node_b)
    g.add_node("contextualize", contextualize_node)
    g.add_edge(START, "plain_chunk")
    g.add_edge("plain_chunk", "contextualize")
    g.add_edge("contextualize", END)
    return g.compile()


contextual_ingest_graph = _build_contextual_ingest_graph()


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE C — QUERY REWRITING + PARENT LOOKUP
#
# Graph: START → rewrite_query → multi_retrieve
#              → (conditional) parent_lookup  ─┐
#                              generate  ←──────┘
#              → END
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedQueryState(TypedDict):
    query: str                           # original user query (input)
    all_chunks: list[Document]           # store to search against (input)
    rewritten_queries: list[str]         # from rewrite_query_node
    retrieved_chunks: list[Document]     # from multi_retrieve_node (child or regular)
    final_chunks: list[Document]         # from parent_lookup or same as retrieved_chunks
    answer: str


def rewrite_query_node(state: AdvancedQueryState) -> dict:
    """
    Technique C — Query Rewriting / Expansion.

    Ambiguous queries fail retrieval for three common reasons:
      1. Acronyms:       "RAG"  →  "Retrieval Augmented Generation"
      2. Synonyms:       "fast" →  "low latency", "efficient", "quick"
      3. Multi-part Qs:  "What is X and why does Y fail?" needs 2 searches

    Claude Haiku rewrites the original query into 3 distinct formulations
    that together cover more of the semantic surface. BM25 runs once per
    formulation; results are merged and deduplicated.

    Failure mode addressed: 'Embedding Mismatch' / 'No Hybrid Search'
    Even with pure BM25, rewriting surfaces synonyms the original query missed.
    """
    messages = [
        SystemMessage(content=(
            "You are a search query expert. Given a user question, "
            "rewrite it into 3 distinct search queries that together cover "
            "all aspects of the question. Handle acronyms, synonyms, and "
            "sub-questions. Output ONLY the 3 queries, one per line, "
            "numbered 1. 2. 3. — no other text."
        )),
        HumanMessage(content=f"Original question: {state['query']}"),
    ]
    response = _llm.invoke(messages).content.strip()

    # Parse the numbered list
    queries = []
    for line in response.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            queries.append(line.split(". ", 1)[1].strip())
    # Always include the original as a fallback
    if state["query"] not in queries:
        queries.insert(0, state["query"])

    return {"rewritten_queries": queries[:4]}   # cap at 4


def multi_retrieve_node(state: AdvancedQueryState) -> dict:
    """
    Run BM25 on each rewritten query, merge, and deduplicate results.

    Each BM25 run retrieves k=4 chunks. With 3-4 queries, up to 16 raw
    results feed into deduplication, yielding 4-10 unique chunks covering
    different aspects of the question.
    """
    if not state["all_chunks"]:
        return {"retrieved_chunks": []}

    retriever = BM25Retriever.from_documents(state["all_chunks"], k=4)
    all_retrieved: list[Document] = []

    for q in state["rewritten_queries"]:
        hits = retriever.invoke(q)
        all_retrieved.extend(hits)

    unique = _dedup(all_retrieved)
    return {"retrieved_chunks": unique[:8]}   # cap context at 8 chunks


def _needs_parent_lookup(state: AdvancedQueryState) -> str:
    """Conditional edge: branch to parent_lookup if chunks carry a parent_id."""
    if any("parent_id" in c.metadata for c in state["retrieved_chunks"]):
        return "parent_lookup"
    return "generate_advanced"


def parent_lookup_node(state: AdvancedQueryState) -> dict:
    """
    Technique A — Parent lookup after child retrieval.

    Retrieved child chunks (128 tok) are precise but narrow.
    This node swaps them for their corresponding parent sections
    (1024 tok) so the LLM has full surrounding context.

    Each unique parent_id is looked up once, deduplicating cases where
    multiple children from the same parent section were retrieved.
    """
    seen_parents: set[str] = set()
    parents: list[Document] = []

    for child in state["retrieved_chunks"]:
        pid = child.metadata.get("parent_id")
        if pid and pid not in seen_parents:
            parent_doc = parent_store.get(pid)
            if parent_doc:
                seen_parents.add(pid)
                parents.append(parent_doc)

    # Fall back to child chunks if parent store is empty
    return {"final_chunks": parents if parents else state["retrieved_chunks"]}


def generate_node_advanced(state: AdvancedQueryState) -> dict:
    """
    Stage 4 — Generation using the final context (parents or plain chunks).

    Uses state["final_chunks"] which was set by either:
      • parent_lookup_node  (if parent-document retrieval was active)
      • a direct edge that copies retrieved_chunks → final_chunks
    """
    # If final_chunks was never set, fall back to retrieved_chunks
    chunks = state.get("final_chunks") or state["retrieved_chunks"]
    if not chunks:
        return {"answer": "No relevant context found in the document store."}

    context = "\n\n---\n\n".join(
        f"[Source {i+1} | {c.metadata.get('source', 'unknown')}]\n{c.page_content}"
        for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=(
            "You are a RAG architecture expert. "
            "Answer using ONLY the context below. "
            "Cite the source number(s) used, e.g. '[Source 2]'. "
            "If the context doesn't cover the question, say: "
            "'I don't know — context does not cover this.'"
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
    ]
    response = _llm.invoke(messages)
    return {"answer": response.content}


def _copy_retrieved_to_final(state: AdvancedQueryState) -> dict:
    """Pass-through node: copies retrieved_chunks → final_chunks when no parent lookup."""
    return {"final_chunks": state["retrieved_chunks"]}


def _build_advanced_query_graph():
    g = StateGraph(AdvancedQueryState)

    g.add_node("rewrite_query",      rewrite_query_node)
    g.add_node("multi_retrieve",     multi_retrieve_node)
    g.add_node("parent_lookup",      parent_lookup_node)
    g.add_node("copy_to_final",      _copy_retrieved_to_final)
    g.add_node("generate_advanced",  generate_node_advanced)

    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "multi_retrieve")

    # Conditional: if chunks carry parent_id metadata → lookup parents
    g.add_conditional_edges(
        "multi_retrieve",
        _needs_parent_lookup,
        {"parent_lookup": "parent_lookup", "generate_advanced": "copy_to_final"},
    )
    g.add_edge("parent_lookup", "generate_advanced")
    g.add_edge("copy_to_final", "generate_advanced")
    g.add_edge("generate_advanced", END)

    return g.compile()


advanced_query_graph = _build_advanced_query_graph()


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PROTOCOL
#
# When debugging a bad answer, always trace BACKWARD from generation to ingestion:
#
#   Step 1 — Check generation:   did the LLM hallucinate despite correct context?
#   Step 2 — Check retrieval:    was the correct chunk even in the top-K results?
#   Step 3 — Diagnose:           classify failure stage and emit recommendations.
#
# Failure classification matrix:
#
#   grounding=GROUNDED                         → No failure, answer is correct
#   grounding=HALLUCINATED, retrieval=YES      → "Lost in middle" / context clutter
#   grounding=HALLUCINATED, retrieval=NO       → Retrieval or ingestion failure
#   grounding=PARTIALLY_GROUNDED              → Partial retrieval hit
#
# Graph: START → retrieve → generate
#              → check_retrieval → check_grounding → diagnose → END
# ══════════════════════════════════════════════════════════════════════════════

class DiagnosticState(TypedDict):
    query: str
    all_chunks: list[Document]

    # Stage outputs (intermediate)
    retrieved_chunks: list[Document]
    answer: str

    # Diagnostic verdicts
    retrieval_verdict: str    # "YES" | "PARTIAL" | "NO"
    grounding_verdict: str    # "GROUNDED" | "PARTIALLY_GROUNDED" | "HALLUCINATED"

    # Final diagnosis
    failure_stage: str        # "none" | "lost_in_middle" | "retrieval" | "ingestion"
    diagnosis: str
    recommendations: list[str]


def retrieve_node_diag(state: DiagnosticState) -> dict:
    """
    Hybrid retrieval for diagnostic — mirrors retrieve_node in rag_pipeline.py.

    Uses the same FAISS + BM25 EnsembleRetriever so the diagnostic accurately
    reflects what the production query path would retrieve.
    """
    from rag_pipeline import _faiss_store, _rrf_merge

    chunks = state["all_chunks"]
    if not chunks:
        return {"retrieved_chunks": []}

    bm25_hits = BM25Retriever.from_documents(chunks, k=6).invoke(state["query"])

    if _faiss_store is not None:
        faiss_hits = _faiss_store.similarity_search(state["query"], k=6)
        retrieved = _rrf_merge([bm25_hits, faiss_hits])
    else:
        retrieved = bm25_hits

    return {"retrieved_chunks": retrieved[:5]}


def generate_node_diag(state: DiagnosticState) -> dict:
    """Generation step — identical to baseline generate_node."""
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"answer": "No relevant chunks retrieved."}

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}"
        for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(content=(
            "You are a RAG expert. Answer using ONLY the provided context. "
            "Cite chunk numbers. If unsure, say so."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
    ]
    return {"answer": _llm.invoke(messages).content}


def check_retrieval_node(state: DiagnosticState) -> dict:
    """
    Diagnostic Step 1 — Top-K Inspection.

    Asks Claude: "Does any retrieved chunk contain enough information
    to answer the question?"

    Verdict:
      YES     → The answer WAS retrievable; if the LLM got it wrong,
                 the failure is in generation (lost-in-middle, context clutter).
      PARTIAL → Some related information present but answer not explicit.
      NO      → The failure is upstream: retrieval or ingestion.
    """
    if not state["retrieved_chunks"]:
        return {"retrieval_verdict": "NO"}

    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}"
        for i, c in enumerate(state["retrieved_chunks"])
    )
    messages = [
        SystemMessage(content=(
            "You are evaluating whether retrieved text chunks contain enough "
            "information to answer a question. "
            "Respond with exactly one of: YES, PARTIAL, NO. "
            "Then add one sentence of explanation after a newline."
        )),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Retrieved chunks:\n{chunks_text}"
        )),
    ]
    response = _llm.invoke(messages).content.strip()
    # Extract the verdict from the first word/line
    first_line = response.splitlines()[0].strip().upper()
    if first_line.startswith("YES"):
        verdict = "YES"
    elif first_line.startswith("PARTIAL"):
        verdict = "PARTIAL"
    else:
        verdict = "NO"

    return {"retrieval_verdict": verdict}


def check_grounding_node(state: DiagnosticState) -> dict:
    """
    Diagnostic Step 2 — Grounding Check.

    Asks Claude: "Is the generated answer grounded in the provided context,
    or did the LLM rely on its training weights?"

    Verdict:
      GROUNDED           → Answer matches context; no hallucination detected.
      PARTIALLY_GROUNDED → Some facts from context, some from training.
      HALLUCINATED       → Answer contradicts context or invents facts.
    """
    if not state["retrieved_chunks"] or not state["answer"]:
        return {"grounding_verdict": "HALLUCINATED"}

    context = "\n\n---\n\n".join(c.page_content for c in state["retrieved_chunks"])
    messages = [
        SystemMessage(content=(
            "You are evaluating whether a generated answer is grounded in "
            "a provided context. "
            "Respond with exactly one of: GROUNDED, PARTIALLY_GROUNDED, HALLUCINATED. "
            "Then add one sentence of explanation after a newline."
        )),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Generated answer: {state['answer']}\n\n"
            f"Context that was provided to generate the answer:\n{context}"
        )),
    ]
    response = _llm.invoke(messages).content.strip()
    first_line = response.splitlines()[0].strip().upper()
    if first_line.startswith("GROUNDED"):
        verdict = "GROUNDED"
    elif first_line.startswith("PARTIALLY"):
        verdict = "PARTIALLY_GROUNDED"
    else:
        verdict = "HALLUCINATED"

    return {"grounding_verdict": verdict}


def diagnose_node(state: DiagnosticState) -> dict:
    """
    Diagnostic Step 3 — Classify failure stage and emit recommendations.

    Applies the diagnostic matrix (no LLM call — pure deterministic logic):

    ┌─────────────────────┬──────────────┬──────────────────────────────────┐
    │ Grounding           │ Retrieval    │ Failure Stage                    │
    ├─────────────────────┼──────────────┼──────────────────────────────────┤
    │ GROUNDED            │ any          │ none — answer is correct         │
    │ HALLUCINATED        │ YES          │ lost_in_middle / context clutter │
    │ HALLUCINATED        │ PARTIAL      │ partial_retrieval                │
    │ HALLUCINATED        │ NO           │ retrieval or ingestion failure   │
    │ PARTIALLY_GROUNDED  │ YES / PARTIAL│ weak grounding                   │
    └─────────────────────┴──────────────┴──────────────────────────────────┘
    """
    g = state["grounding_verdict"]
    r = state["retrieval_verdict"]

    if g == "GROUNDED":
        return {
            "failure_stage": "none",
            "diagnosis": "Answer appears correct and grounded in the retrieved context.",
            "recommendations": ["No action needed."],
        }

    if g in ("HALLUCINATED", "PARTIALLY_GROUNDED") and r == "YES":
        return {
            "failure_stage": "lost_in_middle",
            "diagnosis": (
                "The correct chunk WAS retrieved (top-K inspection: YES) but the "
                "LLM did not use it correctly. This is the 'lost in the middle' effect: "
                "LLMs systematically under-attend to information in the center of long contexts."
            ),
            "recommendations": [
                "Re-rank: move the most relevant chunk to the top or bottom of the prompt.",
                "Reduce K from 4 to 2-3 to cut context clutter.",
                "Try Parent-Document Retrieval (POST /rag/ingest/parent) to shrink "
                "the number of chunks while keeping rich context.",
                "Lower temperature further or tighten the system prompt.",
            ],
        }

    if r == "PARTIAL":
        return {
            "failure_stage": "partial_retrieval",
            "diagnosis": (
                "The retrieved chunks have related information but the answer was not "
                "explicitly present. The LLM partially hallucinated to fill the gap."
            ),
            "recommendations": [
                "Add hybrid search: fuse BM25 with dense vector retrieval via RRF.",
                "Try Query Rewriting (POST /rag/query/advanced) to expand the query "
                "and surface semantically related chunks.",
                "Try Contextual Chunking (POST /rag/ingest/contextual) to enrich chunks "
                "with document-level context before indexing.",
            ],
        }

    # r == "NO": the failure is in retrieval or ingestion
    return {
        "failure_stage": "retrieval_or_ingestion",
        "diagnosis": (
            "The correct chunk was NOT found in the top-K results. "
            "The failure is upstream in retrieval or ingestion."
        ),
        "recommendations": [
            "Check chunking: fixed-size splits may have cut the answer across two chunks "
            "(see 'Answers Split Across Chunks' failure mode).",
            "Add hybrid search: BM25 alone misses semantic synonyms — add dense retrieval.",
            "Try Query Rewriting (/rag/query/advanced) to expand the search surface.",
            "Try Contextual Chunking (/rag/ingest/contextual) to enrich sparse chunks.",
            "Increase K temporarily to verify the chunk exists anywhere in the store.",
        ],
    }


def _build_diagnostic_graph():
    g = StateGraph(DiagnosticState)

    g.add_node("retrieve",         retrieve_node_diag)
    g.add_node("generate",         generate_node_diag)
    g.add_node("check_retrieval",  check_retrieval_node)
    g.add_node("check_grounding",  check_grounding_node)
    g.add_node("diagnose",         diagnose_node)

    g.add_edge(START,             "retrieve")
    g.add_edge("retrieve",        "generate")
    g.add_edge("generate",        "check_retrieval")
    g.add_edge("check_retrieval", "check_grounding")
    g.add_edge("check_grounding", "diagnose")
    g.add_edge("diagnose",        END)

    return g.compile()


diagnostic_graph = _build_diagnostic_graph()
