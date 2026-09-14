import json
import time
from pathlib import Path

import requests


API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DATA_DIR / "dsai_corpus_raw.json"

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "publicationDate",
    "fieldsOfStudy",
    "citationCount",
    "referenceCount",
    "isOpenAccess",
    "url",
])


QUERIES = [
    '"machine learning"',
    '"deep learning"',
    '"representation learning"',
    '"natural language processing"',
    '"computer vision"',
    '"reinforcement learning"',
    '"generative AI"',
    '"data mining"',
]


MAX_PAPERS_PER_QUERY = 1500
MIN_YEAR = 2018

MAX_RETRIES = 5
RETRY_DELAY = 10


def fetch_query(query: str, max_papers: int):
    """Fetch papers for one query using Semantic Scholar bulk pagination."""

    papers = []
    token = None

    print()
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    while len(papers) < max_papers:

        params = {
            "query": query,
            "fields": FIELDS,
            "year": f"{MIN_YEAR}-",
        }

        if token:
            params["token"] = token

        for attempt in range(MAX_RETRIES):

            response = requests.get(
                API_URL,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1)

                print(
                    f"Rate limited (429). "
                    f"Waiting {wait_time}s before retry..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()
            break

        else:
            raise RuntimeError(
                "Semantic Scholar API rate limit persisted "
                f"after {MAX_RETRIES} retries."
            )

        batch = result.get("data", [])

        if not batch:
            break

        papers.extend(batch)

        print(
            f"Received batch: {len(batch)} | "
            f"Query total collected: {len(papers)}"
        )

        token = result.get("token")

        if not token:
            break

        time.sleep(2)

    return papers[:max_papers]


def deduplicate_papers(papers):
    """Remove duplicate papers using paperId."""

    unique = {}

    for paper in papers:
        paper_id = paper.get("paperId")

        if paper_id:
            unique[paper_id] = paper

    return list(unique.values())


def save_json(data, output_file: Path):
    """Save papers as formatted JSON."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "data": data,
        "metadata": {
            "total_papers": len(data),
            "queries": QUERIES,
            "min_year": MIN_YEAR,
        },
    }

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():

    print()
    print("=" * 60)
    print("ARI DSAI CORPUS COLLECTION")
    print("=" * 60)

    all_papers = []

    for query in QUERIES:

        papers = fetch_query(
            query=query,
            max_papers=MAX_PAPERS_PER_QUERY,
        )

        print(
            f"Finished query: {query}"
        )

        print(
            f"Papers collected for query: {len(papers)}"
        )

        all_papers.extend(papers)

        print(
            f"Total papers before deduplication: "
            f"{len(all_papers)}"
        )

    print()
    print("=" * 60)
    print("DEDUPLICATION")
    print("=" * 60)

    before = len(all_papers)

    unique_papers = deduplicate_papers(
        all_papers
    )

    after = len(unique_papers)

    print(
        f"Before deduplication: {before}"
    )

    print(
        f"After deduplication:  {after}"
    )

    print(
        f"Duplicates removed:   {before - after}"
    )

    save_json(
        unique_papers,
        OUTPUT_FILE,
    )

    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"Final corpus size: {len(unique_papers)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()