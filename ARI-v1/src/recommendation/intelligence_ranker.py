from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_results.parquet"
)

RESEARCH_INTELLIGENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_intelligence.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_results.parquet"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Final ranking weights
SEMANTIC_WEIGHT = 0.40
INTELLIGENCE_WEIGHT = 0.30
CITATION_WEIGHT = 0.15
EMERGENCE_WEIGHT = 0.15


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

retrieval = pd.read_parquet(
    RETRIEVAL_FILE
)

intelligence = pd.read_parquet(
    RESEARCH_INTELLIGENCE_FILE
)


print("=" * 70)
print("INTELLIGENCE-AWARE RECOMMENDATION RANKING")
print("=" * 70)

print(
    f"Semantic candidates: {len(retrieval)}"
)

print(
    f"Research intelligence corpus: "
    f"{len(intelligence)}"
)


# ---------------------------------------------------------
# Validate retrieval
# ---------------------------------------------------------

required_retrieval_columns = [
    "paperId",
    "similarity",
]


missing_retrieval = [
    column
    for column in required_retrieval_columns
    if column not in retrieval.columns
]


if missing_retrieval:

    raise ValueError(
        "Missing retrieval columns: "
        f"{missing_retrieval}"
    )


# ---------------------------------------------------------
# Validate intelligence data
# ---------------------------------------------------------

required_intelligence_columns = [
    "paperId",
    "intelligence_score",
    "citation_score",
    "topic_emergence_score",
]


missing_intelligence = [
    column
    for column in required_intelligence_columns
    if column not in intelligence.columns
]


if missing_intelligence:

    raise ValueError(
        "Missing intelligence columns: "
        f"{missing_intelligence}"
    )


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

merged = retrieval.merge(
    intelligence[
        [
            "paperId",
            "intelligence_score",
            "citation_score",
            "topic_emergence_score",
        ]
    ],
    on="paperId",
    how="left",
)


# ---------------------------------------------------------
# Handle missing values
# ---------------------------------------------------------

score_columns = [
    "intelligence_score",
    "citation_score",
    "topic_emergence_score",
]


for column in score_columns:

    merged[column] = (
        merged[column]
        .fillna(0)
    )


# ---------------------------------------------------------
# Normalize semantic relevance
# ---------------------------------------------------------

semantic_min = (
    merged["similarity"].min()
)

semantic_max = (
    merged["similarity"].max()
)


if semantic_max > semantic_min:

    merged["semantic_score"] = (
        (
            merged["similarity"]
            - semantic_min
        )
        /
        (
            semantic_max
            - semantic_min
        )
    )

else:

    merged["semantic_score"] = 1.0


# ---------------------------------------------------------
# Normalize intelligence
# ---------------------------------------------------------

intelligence_min = (
    merged["intelligence_score"].min()
)

intelligence_max = (
    merged["intelligence_score"].max()
)


if intelligence_max > intelligence_min:

    merged["normalized_intelligence"] = (
        (
            merged["intelligence_score"]
            - intelligence_min
        )
        /
        (
            intelligence_max
            - intelligence_min
        )
    )

else:

    merged["normalized_intelligence"] = 1.0


# ---------------------------------------------------------
# Normalize citation impact
# ---------------------------------------------------------

citation_min = (
    merged["citation_score"].min()
)

citation_max = (
    merged["citation_score"].max()
)


if citation_max > citation_min:

    merged["normalized_citation"] = (
        (
            merged["citation_score"]
            - citation_min
        )
        /
        (
            citation_max
            - citation_min
        )
    )

else:

    merged["normalized_citation"] = 1.0


# ---------------------------------------------------------
# Normalize topic emergence
# ---------------------------------------------------------

emergence_min = (
    merged["topic_emergence_score"].min()
)

emergence_max = (
    merged["topic_emergence_score"].max()
)


if emergence_max > emergence_min:

    merged["normalized_emergence"] = (
        (
            merged["topic_emergence_score"]
            - emergence_min
        )
        /
        (
            emergence_max
            - emergence_min
        )
    )

else:

    merged["normalized_emergence"] = 1.0


# ---------------------------------------------------------
# Calculate recommendation score
# ---------------------------------------------------------

merged["recommendation_score"] = (

    SEMANTIC_WEIGHT
    * merged["semantic_score"]

    +

    INTELLIGENCE_WEIGHT
    * merged["normalized_intelligence"]

    +

    CITATION_WEIGHT
    * merged["normalized_citation"]

    +

    EMERGENCE_WEIGHT
    * merged["normalized_emergence"]
)


# ---------------------------------------------------------
# Sort candidates
# ---------------------------------------------------------

merged = merged.sort_values(
    by="recommendation_score",
    ascending=False,
).reset_index(drop=True)


# ---------------------------------------------------------
# Assign ranking
# ---------------------------------------------------------

merged["recommendation_rank"] = (
    range(
        1,
        len(merged) + 1
    )
)


# ---------------------------------------------------------
# Select output columns
# ---------------------------------------------------------

preferred_columns = [
    "recommendation_rank",
    "paperId",
    "title",
    "year",
    "citationCount",
    "semantic_quality",
    "cluster",
    "similarity",
    "url",
    "intelligence_score",
    "citation_score",
    "topic_emergence_score",
    "semantic_score",
    "normalized_intelligence",
    "normalized_citation",
    "normalized_emergence",
    "recommendation_score",
]


output_columns = [
    column
    for column in preferred_columns
    if column in merged.columns
]


results = merged[
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
# Display top 20
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 INTELLIGENCE-AWARE CANDIDATES")
print("=" * 70)


for _, row in results.head(20).iterrows():

    print(
        f"{row['recommendation_rank']}. "
        f"{row['title']}"
    )

    print(
        f"   Recommendation score: "
        f"{row['recommendation_score']:.4f}"
    )

    print(
        f"   Semantic similarity: "
        f"{row['similarity']:.4f}"
    )

    print(
        f"   Intelligence score: "
        f"{row['intelligence_score']:.4f}"
    )

    print(
        f"   Citation score: "
        f"{row['citation_score']:.4f}"
    )

    print(
        f"   Emergence score: "
        f"{row['topic_emergence_score']:.4f}"
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
print("CANDIDATE POOL SUMMARY")
print("=" * 70)

print(
    f"Candidates ranked: {len(results)}"
)

print(
    f"Unique clusters: {unique_clusters}"
)

print(
    f"Cluster coverage: "
    f"{unique_clusters / 5 * 100:.2f}%"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print(
    "\nIntelligence-aware ranking complete."
)