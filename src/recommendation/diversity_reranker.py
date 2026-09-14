from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_results.parquet"
)

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
    / "diverse_recommendations.parquet"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 10

LAMBDA = 0.75


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

recommendations = pd.read_parquet(
    RECOMMENDATION_FILE
)

papers = pd.read_parquet(
    PAPERS_FILE
)

embeddings = np.load(
    EMBEDDINGS_FILE
)


print("=" * 70)
print("DIVERSITY-AWARE RECOMMENDATION RERANKING")
print("=" * 70)

print(
    f"Candidate recommendations: "
    f"{len(recommendations)}"
)

print(
    f"Requested top-K: {TOP_K}"
)

print(
    f"MMR lambda: {LAMBDA}"
)


# ---------------------------------------------------------
# Get cluster mapping
# ---------------------------------------------------------

cluster_mapping = (
    papers[
        ["paperId", "cluster"]
    ]
    .drop_duplicates(
        subset="paperId"
    )
)


# ---------------------------------------------------------
# Remove any existing cluster column
# ---------------------------------------------------------

if "cluster" in recommendations.columns:

    recommendations = recommendations.drop(
        columns=["cluster"]
    )


# ---------------------------------------------------------
# Merge cluster information
# ---------------------------------------------------------

recommendations = recommendations.merge(
    cluster_mapping,
    on="paperId",
    how="left",
)


# ---------------------------------------------------------
# Validate columns
# ---------------------------------------------------------

required_columns = [
    "paperId",
    "title",
    "recommendation_score",
    "similarity",
    "cluster",
]


missing_columns = [
    column
    for column in required_columns
    if column not in recommendations.columns
]


if missing_columns:

    print(
        "\nAvailable columns:"
    )

    print(
        recommendations.columns.tolist()
    )

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# ---------------------------------------------------------
# Map paper IDs to embedding indices
# ---------------------------------------------------------

paper_to_index = {
    paper_id: index
    for index, paper_id
    in enumerate(
        papers["paperId"]
    )
}


candidate_indices = []

valid_rows = []


for index, row in recommendations.iterrows():

    paper_id = row["paperId"]

    if paper_id in paper_to_index:

        candidate_indices.append(
            paper_to_index[paper_id]
        )

        valid_rows.append(index)


candidates = recommendations.loc[
    valid_rows
].reset_index(drop=True)


candidate_embeddings = embeddings[
    candidate_indices
]


# ---------------------------------------------------------
# Normalize embeddings
# ---------------------------------------------------------

norms = np.linalg.norm(
    candidate_embeddings,
    axis=1,
    keepdims=True,
)

norms[norms == 0] = 1


candidate_embeddings = (
    candidate_embeddings / norms
)


# ---------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------

similarity_matrix = (
    candidate_embeddings
    @ candidate_embeddings.T
)


# ---------------------------------------------------------
# MMR selection
# ---------------------------------------------------------

selected = []

mmr_scores = []

remaining = list(
    range(len(candidates))
)


while (
    remaining
    and len(selected) < TOP_K
):

    best_candidate = None

    best_score = -np.inf


    for candidate in remaining:

        relevance = float(
            candidates.loc[
                candidate,
                "recommendation_score"
            ]
        )


        if not selected:

            redundancy = 0.0

        else:

            redundancy = max(
                similarity_matrix[
                    candidate,
                    selected
                ]
            )


        mmr_score = (
            LAMBDA * relevance
            - (1 - LAMBDA) * redundancy
        )


        if mmr_score > best_score:

            best_score = mmr_score

            best_candidate = candidate


    selected.append(
        best_candidate
    )

    mmr_scores.append(
        best_score
    )

    remaining.remove(
        best_candidate
    )


# ---------------------------------------------------------
# Build final recommendations
# ---------------------------------------------------------

diverse_df = candidates.loc[
    selected
].copy()


diverse_df = diverse_df.reset_index(
    drop=True
)


diverse_df["mmr_score"] = (
    mmr_scores
)


diverse_df["diverse_rank"] = (
    range(
        1,
        len(diverse_df) + 1
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

diverse_df.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display recommendations
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIVERSE RECOMMENDATIONS")
print("=" * 70)


for _, row in diverse_df.iterrows():

    print(
        f"{row['diverse_rank']}. "
        f"{row['title']}"
    )

    print(
        f"   Original score: "
        f"{row['recommendation_score']:.4f}"
    )

    print(
        f"   Semantic similarity: "
        f"{row['similarity']:.4f}"
    )

    print(
        f"   Cluster: "
        f"{row['cluster']}"
    )

    print(
        f"   MMR score: "
        f"{row['mmr_score']:.4f}"
    )

    print("-" * 70)


# ---------------------------------------------------------
# Diversity summary
# ---------------------------------------------------------

unique_clusters = (
    diverse_df["cluster"]
    .nunique()
)


total_clusters = (
    papers["cluster"]
    .nunique()
)


cluster_coverage = (
    unique_clusters
    / total_clusters
    * 100
)


# ---------------------------------------------------------
# Semantic diversity
# ---------------------------------------------------------

selected_indices = [
    paper_to_index[paper_id]
    for paper_id
    in diverse_df["paperId"]
]


selected_embeddings = embeddings[
    selected_indices
]


if len(selected_embeddings) > 1:

    selected_norms = np.linalg.norm(
        selected_embeddings,
        axis=1,
        keepdims=True,
    )

    selected_norms[
        selected_norms == 0
    ] = 1

    selected_embeddings = (
        selected_embeddings
        / selected_norms
    )

    selected_similarity = (
        selected_embeddings
        @ selected_embeddings.T
    )

    upper_triangle = (
        selected_similarity[
            np.triu_indices(
                len(selected_embeddings),
                k=1,
            )
        ]
    )

    mean_pairwise_similarity = (
        upper_triangle.mean()
    )

    semantic_diversity = (
        1
        - mean_pairwise_similarity
    )

else:

    mean_pairwise_similarity = 0.0

    semantic_diversity = 0.0


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIVERSITY SUMMARY")
print("=" * 70)

print(
    f"Candidates considered: "
    f"{len(candidates)}"
)

print(
    f"Final recommendations: "
    f"{len(diverse_df)}"
)

print(
    f"Unique clusters: "
    f"{unique_clusters}"
)

print(
    f"Total clusters: "
    f"{total_clusters}"
)

print(
    f"Cluster coverage: "
    f"{cluster_coverage:.2f}%"
)

print(
    f"Mean pairwise semantic similarity: "
    f"{mean_pairwise_similarity:.4f}"
)

print(
    f"Semantic diversity score: "
    f"{semantic_diversity:.4f}"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)

print(
    "\nDiversity-aware reranking complete."
)