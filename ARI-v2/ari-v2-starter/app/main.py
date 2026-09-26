"""ARI v2 Sprint 1: paper discovery and semantic ranking."""
import asyncio
import os
from functools import lru_cache
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ARI v2", version="0.1.0")
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    limit: int = Field(default=10, ge=1, le=30)


class Paper(BaseModel):
    title: str
    paper_id: str
    year: Optional[int] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    citation_count: int = 0
    relevance: float = 0.0


class ResearchResponse(BaseModel):
    question: str
    papers: list[Paper]
    caveat: str = "Discovery and semantic ranking only; findings and citations have not been independently verified."


@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def rank_papers(question: str, papers: list[Paper]) -> list[Paper]:
    if not papers:
        return []
    model = embedding_model()
    texts = [question] + [p.title + ". " + (p.abstract or "") for p in papers]
    vectors = model.encode(texts, normalize_embeddings=True)
    scores = vectors[1:] @ vectors[0]
    for paper, score in zip(papers, scores):
        paper.relevance = round(float(score), 4)
    return sorted(papers, key=lambda p: p.relevance, reverse=True)


async def discover(question: str, limit: int) -> list[Paper]:
    headers = {}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    params = {
        "query": question,
        "limit": limit,
        "fields": "title,year,abstract,url,citationCount",
    }
    async with httpx.AsyncClient(timeout=25) as client:
        for attempt in range(3):
            response = await client.get(S2_URL, params=params, headers=headers)
            if response.status_code == 429 and attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return [Paper(
                title=item.get("title") or "Untitled",
                paper_id=item["paperId"],
                year=item.get("year"),
                abstract=item.get("abstract"),
                url=item.get("url"),
                citation_count=item.get("citationCount") or 0,
            ) for item in response.json().get("data", []) if item.get("paperId")]
    return []


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    try:
        papers = await discover(request.question, request.limit)
        ranked = await asyncio.to_thread(rank_papers, request.question, papers)
        return ResearchResponse(question=request.question, papers=ranked)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise HTTPException(status_code=503, detail="Paper provider rate limited the request. Retry later or set an API key.") from exc
        raise HTTPException(status_code=502, detail="Paper provider returned an error.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Paper provider could not be reached.") from exc
