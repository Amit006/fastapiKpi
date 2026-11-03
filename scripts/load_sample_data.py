"""
Sample data loader for the RAG pipeline.

Fetches data from two free, no-auth-required sources:
  1. arXiv API    — paper abstracts on RAG / LLM topics
  2. Wikipedia    — article summaries via MediaWiki REST API

Data is POSTed to the running FastAPI server at /rag/ingest so it
integrates with the normal pipeline (Stage 2 chunking + store).

Usage (server must be running on localhost:8000):
    cd fastapiKpi
    uv run python scripts/load_sample_data.py

Optional flags:
    --url   http://localhost:8000   # server base URL
    --arxiv 15                      # number of arXiv papers to fetch
    --no-wiki                       # skip Wikipedia articles

Why arXiv + Wikipedia instead of Kaggle?
    Both require zero credentials and return structured text directly.
    For Kaggle datasets see README.md § "Kaggle datasets (optional)".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# ── httpx is already installed as a FastAPI dependency ────────────────────────
try:
    import httpx
except ImportError:
    print("httpx not found — run: uv add httpx")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — arXiv
# API docs: https://info.arxiv.org/help/api/index.html
# No authentication required.
# ══════════════════════════════════════════════════════════════════════════════

_ARXIV_NS = "http://www.w3.org/2005/Atom"

# Topics to search — each becomes a separate ingest call with its own source tag
ARXIV_QUERIES = [
    "retrieval augmented generation",
    "large language model hallucination",
    "vector database embedding retrieval",
]


def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Call the arXiv Atom feed API and return a list of paper dicts.

    Each dict has: title, abstract, authors, arxiv_id, url
    """
    params = urllib.parse.urlencode({
        "search_query": f"abs:{query}",
        "sortBy": "relevance",
        "max_results": max_results,
    })
    url = f"http://export.arxiv.org/api/query?{params}"

    with urllib.request.urlopen(url, timeout=20) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    ns = {"a": _ARXIV_NS}
    papers = []

    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", namespaces=ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("a:summary", namespaces=ns) or "").strip().replace("\n", " ")
        link = (entry.findtext("a:id", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("a:name", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        arxiv_id = link.split("/")[-1] if link else "unknown"

        # Skip the feed-level entry (has no summary)
        if not abstract:
            continue

        papers.append({
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors[:4]),
            "arxiv_id": arxiv_id,
            "url": link,
        })

    return papers


def paper_to_text(paper: dict) -> str:
    """Format a paper dict as a single readable document for ingestion."""
    return (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"arXiv: {paper['url']}\n\n"
        f"Abstract:\n{paper['abstract']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Wikipedia (MediaWiki REST API)
# https://en.wikipedia.org/api/rest_v1/
# No authentication required.
# ══════════════════════════════════════════════════════════════════════════════

WIKIPEDIA_TOPICS = [
    "Retrieval-augmented generation",
    "Large language model",
    "Vector database",
    "BM25",
    "Transformer (deep learning architecture)",
    "Sentence embedding",
    "Named-entity recognition",    # shows NLP breadth
]


def fetch_wikipedia_summary(title: str) -> dict | None:
    """
    Return {title, extract, url} for a Wikipedia article via the summary API.
    Returns None on error.

    Falls back to SSL-unverified request when the system cert store rejects
    Wikipedia's certificate (common on some Windows machines).
    """
    import ssl

    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    # Wikipedia requires a descriptive User-Agent (returns 403 otherwise)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RAGPipelineDemo/1.0 (learning project; contact via GitHub)"},
    )

    def _fetch(ctx=None):
        kwargs = {"timeout": 10}
        if ctx is not None:
            kwargs["context"] = ctx
        with urllib.request.urlopen(req, **kwargs) as resp:
            return json.loads(resp.read())

    def _is_ssl_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "certificate" in msg or "ssl" in msg or "tls" in msg

    try:
        data = _fetch()
    except Exception as exc:
        if _is_ssl_error(exc):
            # urllib wraps ssl errors in URLError — retry using certifi's bundle
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
            try:
                data = _fetch(ctx)
            except Exception as exc2:
                print(f"    ⚠  Could not fetch '{title}': {exc2}")
                return None
        else:
            print(f"    ⚠  Could not fetch '{title}': {exc}")
            return None

    return {
        "title": data.get("title", title),
        "extract": data.get("extract", ""),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


def wikipedia_to_text(article: dict) -> str:
    return f"# {article['title']}\n\nSource: {article['url']}\n\n{article['extract']}"


# ══════════════════════════════════════════════════════════════════════════════
# Ingest helpers — POST to the running FastAPI server
# ══════════════════════════════════════════════════════════════════════════════

def ingest(client: httpx.Client, text: str, source: str, base_url: str) -> int:
    """POST text to /rag/ingest; return number of chunks added."""
    resp = client.post(
        f"{base_url}/rag/ingest",
        json={"text": text, "source": source},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["chunks_added"]


def get_status(client: httpx.Client, base_url: str) -> dict:
    return client.get(f"{base_url}/rag/status").json()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(base_url: str, arxiv_count: int, skip_wiki: bool) -> None:
    print(f"\n{'=' * 60}")
    print("  RAG Sample Data Loader")
    print(f"  Server: {base_url}")
    print(f"{'=' * 60}\n")

    # Check server is up
    try:
        with httpx.Client() as probe:
            probe.get(f"{base_url}/rag/status", timeout=5).raise_for_status()
    except Exception:
        print(f"ERROR: Server not reachable at {base_url}")
        print("       Start it first: uv run fastapi dev main.py")
        sys.exit(1)

    total_chunks = 0

    with httpx.Client() as client:
        # ── 1. arXiv ──────────────────────────────────────────────────────────
        print(f"[1/2] Fetching arXiv papers ({arxiv_count} per query × {len(ARXIV_QUERIES)} queries)...\n")
        for query in ARXIV_QUERIES:
            print(f"  Query: \"{query}\"")
            try:
                papers = fetch_arxiv_papers(query, max_results=arxiv_count)
                for paper in papers:
                    n = ingest(client, paper_to_text(paper), f"arxiv:{paper['arxiv_id']}", base_url)
                    total_chunks += n
                    print(f"    + {n:>2} chunks | {paper['title'][:65]}")
                time.sleep(0.5)   # be polite to arXiv API
            except Exception as exc:
                print(f"  ✗ arXiv fetch failed for '{query}': {exc}")

        # ── 2. Wikipedia ──────────────────────────────────────────────────────
        if not skip_wiki:
            print(f"\n[2/2] Fetching Wikipedia article summaries ({len(WIKIPEDIA_TOPICS)} articles)...\n")
            for topic in WIKIPEDIA_TOPICS:
                article = fetch_wikipedia_summary(topic)
                if article and article["extract"]:
                    source = f"wikipedia:{article['title'].replace(' ', '_')}"
                    n = ingest(client, wikipedia_to_text(article), source, base_url)
                    total_chunks += n
                    print(f"    + {n:>2} chunks | {article['title']}")
                time.sleep(0.2)

        # ── Summary ───────────────────────────────────────────────────────────
        status = get_status(client, base_url)
        print(f"\n{'─' * 60}")
        print(f"  Added {total_chunks} new chunks this run.")
        print(f"  Document store total: {status['total_chunks']} chunks")
        print(f"  Sources: {', '.join(status['sources'])}")
        print(f"{'─' * 60}")

    print("\nReady. Try these queries at POST /rag/query:\n")
    sample_questions = [
        "What is retrieval augmented generation and why does it help?",
        "How does BM25 scoring work and when should I use it?",
        "What is the lost-in-the-middle problem in LLMs?",
        "How does a vector database store and search embeddings?",
        "What failure modes exist in the chunking stage of RAG?",
        "What are the trade-offs between dense and sparse retrieval?",
    ]
    for q in sample_questions:
        print(f"  → {q}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load sample data into the RAG pipeline.")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--arxiv", type=int, default=5, help="arXiv papers per query (default 5)")
    parser.add_argument("--no-wiki", action="store_true", help="Skip Wikipedia articles")
    args = parser.parse_args()

    main(base_url=args.url, arxiv_count=args.arxiv, skip_wiki=args.no_wiki)
