import os
from typing import List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain Retrieval & Embedding Imports
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # Kept for embeddings, or swap with another provider
from langchain.retrievers import EnsembleRetriever

# Anthropic Integration
from langchain_anthropic import ChatAnthropic

# 1. Initialize FastAPI app
app = FastAPI(
    title="Anthropic Hybrid Search RAG API",
    description="A FastAPI app running a LangChain hybrid search pipeline powered by Anthropic.",
    version="1.0.0"
)

# Secure your API keys in environment variables
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-key-for-embeddings"

# 2. Mock Dataset
sample_docs = [
    Document(page_content="Model X-5000 requires a 12V DC power input and a minimum of 4GB RAM to execute scripts."),
    Document(page_content="To reset your account password, click on the profile icon and select 'Security Settings'."),
    Document(page_content="Our next-generation systems utilize quantum-inspired architectures to optimize vector workloads."),
    Document(page_content="Troubleshooting error code ERR_404: Check your internet gateway or verify host configuration.")
]

# 3. Setup Hybrid Retriever Components
# Note: Anthropic doesn't host an embedding API, so we keep OpenAI Embeddings here. 
# (You could also use HuggingFace/HuggingFaceEmbeddings for local embedding generation).
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(sample_docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(sample_docs)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 4. Define the Anthropic LLM and RAG Prompt Chain
# Using claude-3-5-sonnet for state-of-the-art response quality
llm = ChatAnthropic(
    model="claude-3-5-sonnet-latest", 
    temperature=0,
    max_tokens=1024
)

prompt = ChatPromptTemplate.from_template("""
You are an expert technical assistant. Answer the user question based only on the provided context.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
Answer:
""")

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Define Pydantic Models
class QueryRequest(BaseModel):
    text: str

# 6. API Endpoints

@app.post("/query")
async def query_rag(request: QueryRequest):
    response = await rag_chain.ainvoke(request.text)
    return {"query": request.text, "response": response}


@app.post("/stream")
async def stream_rag(request: QueryRequest):
    async def generate_chunks():
        async for chunk in rag_chain.astream(request.text):
            yield chunk

    return StreamingResponse(generate_chunks(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
