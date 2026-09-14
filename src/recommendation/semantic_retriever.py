from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_results.parquet"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUERY = "generative AI in higher education"

CANDIDATE_K = 100

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load corpus
# ---------------------------------------------------------

papers = pd.read_parquet(
    PAPERS_FILE
)

embeddings = np.load(
    EMBEDDINGS_FILE
)


print("=" * 70)
print("SEMANTIC RESEARCH RETRIEVAL")
print("=" * 70)

print(
    f"Query: {QUERY}"
)

print(
    f"Corpus size: {len(papers)}"
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print(
    f"\nLoading model: {MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)


# ---------------------------------------------------------
# Encode query
# ---------------------------------------------------------

query_embedding = model.encode(
    QUERY,
    normalize_embeddings=True,
)


# ---------------------------------------------------------
# Calculate semantic similarity
# ---------------------------------------------------------

similarities = (
    embeddings @ query_embedding
)


# ---------------------------------------------------------
# Get top candidates
# ---------------------------------------------------------

candidate_count = min(
    CANDIDATE_K,
    len(papers)
)


top_indices = np.argsort(
    similarities
)[::-1][
    :candidate_count
]


results = papers.iloc[
    top_indices
].copy()


results["similarity"] = (
    similarities[top_indices]
)


# ---------------------------------------------------------
# Sort by semantic similarity
# ---------------------------------------------------------

results = results.sort_values(
    by="similarity",
    ascending=False,
).reset_index(drop=True)


results["retrieval_rank"] = (
    range(
        1,
        len(results) + 1
    )
)


# ---------------------------------------------------------
# Select output columns
# ---------------------------------------------------------

preferred_columns = [
    "retrieval_rank",
    "paperId",
    "title",
    "year",
    "citationCount",
    "semantic_quality",
    "cluster",
    "similarity",
    "url",
]


output_columns = [
    column
    for column in preferred_columns
    if column in results.columns
]


results = results[
    output_columns
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

results.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display top results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print(
    f"TOP {candidate_count} SEMANTIC CANDIDATES"
)
print("=" * 70)


for _, row in results.head(20).iterrows():

    print(
        f"{row['retrieval_rank']}. "
        f"{row['title']}"
    )

    print(
        f"   Similarity: "
        f"{row['similarity']:.4f}"
    )

    print(
        f"   Year: "
        f"{row['year']}"
    )

    print(
        f"   Cluster: "
        f"{row['cluster']}"
    )

    print("-" * 70)


# ---------------------------------------------------------
# Candidate diversity
# ---------------------------------------------------------

unique_clusters = (
    results["cluster"]
    .nunique()
)


print("\n" + "=" * 70)
print("CANDIDATE POOL ANALYSIS")
print("=" * 70)

print(
    f"Candidates: {len(results)}"
)

print(
    f"Unique clusters: {unique_clusters}"
)

print(
    f"Total clusters in corpus: "
    f"{papers['cluster'].nunique()}"
)

print(
    f"Cluster coverage: "
    f"{unique_clusters / papers['cluster'].nunique() * 100:.2f}%"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print(
    "\nSemantic retrieval complete."
)