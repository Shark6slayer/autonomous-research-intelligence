import asyncio
import json
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = Path(os.getenv("ARI_DB_PATH", "ari.sqlite3"))

SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                report TEXT,
                error TEXT
            )
        """)

    yield


app = FastAPI(
    title="ARI v2 Research Intelligence",
    version="0.2.0",
    lifespan=lifespan
)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=12, max_length=1000)
    max_papers: int = Field(default=12, ge=1, le=50)


def search_queries(question: str):
    cleaned = re.sub(r"[^\w\s-]", " ", question).strip()
    words = cleaned.split()

    return [
        " ".join(words[:20]),
        f"{' '.join(words[:12])} research",
    ]


async def semantic_scholar_search(question: str, limit: int):
    headers = {}

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(
        timeout=25,
        headers=headers
    ) as client:

        for query in search_queries(question):

            response = await client.get(
                SEMANTIC_URL,
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,year,url,authors,citationCount,externalIds"
                }
            )

            if response.status_code == 429:
                return []

            response.raise_for_status()

            return response.json().get("data", [])

    return []


async def arxiv_search(question: str, limit: int):

    params = {
        "search_query": f"all:{question}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.get(
            ARXIV_URL,
            params=params
        )

        response.raise_for_status()

        text = response.text

    entries = re.findall(
        r"<entry>(.*?)</entry>",
        text,
        flags=re.DOTALL
    )

    papers = []

    for entry in entries:

        title_match = re.search(
            r"<title>(.*?)</title>",
            entry,
            flags=re.DOTALL
        )

        summary_match = re.search(
            r"<summary>(.*?)</summary>",
            entry,
            flags=re.DOTALL
        )

        published_match = re.search(
            r"<published>(.*?)</published>",
            entry,
            flags=re.DOTALL
        )

        id_match = re.search(
            r"<id>(.*?)</id>",
            entry,
            flags=re.DOTALL
        )

        if not title_match:
            continue

        title = re.sub(
            r"\s+",
            " ",
            title_match.group(1)
        ).strip()

        abstract = ""

        if summary_match:
            abstract = re.sub(
                r"\s+",
                " ",
                summary_match.group(1)
            ).strip()

        papers.append({
            "title": title,
            "abstract": abstract,
            "year": (
                published_match.group(1)[:4]
                if published_match
                else None
            ),
            "url": (
                id_match.group(1).strip()
                if id_match
                else None
            ),
            "authors": [],
            "citationCount": None,
            "externalIds": {}
        })

    return papers


async def discover(question: str, limit: int):

    # Try Semantic Scholar first.
    papers = await semantic_scholar_search(
        question,
        limit
    )

    provider = "Semantic Scholar"

    # Automatic fallback when rate-limited.
    if not papers:

        papers = await arxiv_search(
            question,
            limit
        )

        provider = "arXiv"

    # Deduplicate.
    unique = {}

    for paper in papers:

        title = paper.get("title", "").strip().lower()

        if title:
            unique[title] = paper

    return list(unique.values())[:limit], provider


def make_report(
    question: str,
    papers: list,
    provider: str
):

    lines = [
        f"# Research Brief",
        "",
        f"**Research question:** {question}",
        "",
        f"**Source provider:** {provider}",
        "",
        f"**Papers discovered:** {len(papers)}",
        "",
        "This is an automated literature discovery pass. "
        "Abstract-level evidence should be verified against "
        "full text before drawing conclusions.",
        "",
        "## Literature",
        ""
    ]

    for idx, paper in enumerate(papers, 1):

        title = paper.get(
            "title",
            "Untitled"
        )

        year = paper.get(
            "year",
            "n.d."
        )

        abstract = (
            paper.get("abstract")
            or "Abstract unavailable."
        )

        abstract = abstract.replace(
            "\n",
            " "
        )

        url = paper.get(
            "url",
            "No URL"
        )

        lines.extend([
            f"### {idx}. {title}",
            "",
            f"**Year:** {year}",
            "",
            f"**Source:** {url}",
            "",
            f"**Abstract:** {abstract[:1500]}",
            ""
        ])

    lines.extend([
        "## Initial Research Questions",
        "",
        "- Which methods are evaluated at low SNR?",
        "- Which datasets and modulation classes are used?",
        "- Which baselines are used?",
        "- What evaluation metrics are reported?",
        "- Are results independently reproduced?",
        "- What limitations are reported?",
        "",
        "## Current Limitations",
        "",
        "ARI v0.2 performs literature discovery and "
        "abstract-level synthesis. It does not yet perform "
        "claim verification, full-text reasoning, autonomous "
        "research planning, experiment execution, or "
        "research-gap validation."
    ])

    return "\n".join(lines)


async def run_research(
    project_id: str,
    question: str,
    limit: int
):

    try:

        papers, provider = await discover(
            question,
            limit
        )

        if not papers:
            raise RuntimeError(
                "No papers found from Semantic Scholar or arXiv."
            )

        report = {
            "question": question,
            "provider": provider,
            "papers": papers,
            "markdown": make_report(
                question,
                papers,
                provider
            )
        }

        with db() as conn:
            conn.execute(
                """
                UPDATE projects
                SET status=?, report=?, error=NULL
                WHERE id=?
                """,
                (
                    "completed",
                    json.dumps(report),
                    project_id
                )
            )

    except Exception as exc:

        with db() as conn:
            conn.execute(
                """
                UPDATE projects
                SET status=?, error=?
                WHERE id=?
                """,
                (
                    "failed",
                    str(exc),
                    project_id
                )
            )


@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "0.2.0"
    }


@app.post("/research", status_code=201)
async def create_research(
    request: ResearchRequest
):

    project_id = str(uuid.uuid4())

    with db() as conn:

        conn.execute(
            """
            INSERT INTO projects
            (id, question, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                project_id,
                request.question,
                "running",
                datetime.now(timezone.utc).isoformat()
            )
        )

    await run_research(
        project_id,
        request.question,
        request.max_papers
    )

    return get_research(project_id)


@app.get("/research/{project_id}")
def get_research(project_id: str):

    with db() as conn:

        row = conn.execute(
            "SELECT * FROM projects WHERE id=?",
            (project_id,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Project not found"
        )

    result = dict(row)

    result["report"] = (
        json.loads(result["report"])
        if result["report"]
        else None
    )

    return result