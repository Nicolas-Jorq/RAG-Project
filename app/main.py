from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import get_chain

app = FastAPI(title="RAG API")


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def preload_chain() -> None:
    get_chain()      # warm-up so first user is fast


@app.post("/query")
async def query(body: Query):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    result = get_chain()({"query": body.question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "file": d.metadata.get("source", ""),
                "excerpt": d.page_content[:160] + "…",
            }
            for d in result["source_documents"]
        ],
    }
