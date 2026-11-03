"""Minimal RAG pipeline: load -> chunk -> embed -> FAISS -> retrieve -> answer.

pip install langchain langchain-community langchain-text-splitters \
            langchain-huggingface sentence-transformers faiss-cpu rank_bm25
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, no API key


def load_docs(folder: str) -> list[Document]:
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in Path(folder).glob("**/*.md")
    ]


def build_index(folder: str):
    docs = load_docs(folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)  # metadata (source) carried to each chunk
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_documents(chunks, emb)
    store.save_local(INDEX_DIR)
    return store, chunks


def load_index():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


def hybrid_search(store: FAISS, chunks: list[Document], query: str, k: int = 4):
    """Dense (FAISS) + sparse (BM25) merged with Reciprocal Rank Fusion."""
    dense = store.similarity_search(query, k=k * 2)
    bm25 = BM25Retriever.from_documents(chunks, k=k * 2).invoke(query)

    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranked in (dense, bm25):
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)  # RRF, c=60
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [by_key[t] for t in top]


def build_prompt(query: str, docs: list[Document]) -> str:
    context = "\n\n".join(f"[{d.metadata.get('source')}] {d.page_content}" for d in docs)
    return (
        "Answer ONLY using the context below. If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer (cite sources in [brackets]):"
    )


if __name__ == "__main__":
    store, chunks = build_index("docs")  # put .md files in ./docs
    hits = hybrid_search(store, chunks, "What is the chunking strategy?")
    prompt = build_prompt("What is the chunking strategy?", hits)
    print(prompt)  # send this to your LLM with temperature=0
