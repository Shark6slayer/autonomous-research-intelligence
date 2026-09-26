from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_recommendations.parquet"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)

PAPERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_evaluation.parquet"
)


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
print("COMPLETE-SYSTEM RECOMMENDATION EVALUATION")
print("=" * 70)

print(
    f"Recommendations: {len(recommendations)}"
)

print(
    f"Corpus papers: {len(papers)}"
)

print(
    f"Embedding matrix: {embeddings.shape}"
)


# ---------------------------------------------------------
# Required columns
# ---------------------------------------------------------

required_columns = [
    "paperId",
    "final_rank",
    "recommendation_score",
    "similarity",
    "intelligence_score",
    "year",
]


missing_columns = [
    column
    for column in required_columns
    if column not in recommendations.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# 1. Ranking quality
# ---------------------------------------------------------

evaluation = {}

evaluation["recommendation_count"] = (
    len(recommendations)
)

evaluation["top_recommendation_score"] = (
    recommendations["recommendation_score"].max()
)

evaluation["bottom_recommendation_score"] = (
    recommendations["recommendation_score"].min()
)

evaluation["mean_recommendation_score"] = (
    recommendations["recommendation_score"].mean()
)

evaluation["recommendation_score_std"] = (
    recommendations["recommendation_score"].std()
)


# ---------------------------------------------------------
# 2. Semantic relevance
# ---------------------------------------------------------

evaluation["mean_semantic_similarity"] = (
    recommendations["similarity"].mean()
)

evaluation["top_semantic_similarity"] = (
    recommendations["similarity"].max()
)

evaluation["minimum_semantic_similarity"] = (
    recommendations["similarity"].min()
)


# ---------------------------------------------------------
# 3. Intelligence quality
# ---------------------------------------------------------

evaluation["mean_intelligence_score"] = (
    recommendations["intelligence_score"].mean()
)

evaluation["maximum_intelligence_score"] = (
    recommendations["intelligence_score"].max()
)


# ---------------------------------------------------------
# 4. Cluster / topic coverage
# ---------------------------------------------------------

# The final recommendation file contains cluster,
# while human-readable topic labels are stored in
# the clustered corpus.

cluster_mapping = (
    papers[
        ["paperId", "cluster"]
    ]
    .drop_duplicates("paperId")
)


recommendations_with_cluster = (
    recommendations.merge(
        cluster_mapping,
        on="paperId",
        how="left",
        suffixes=("", "_corpus"),
    )
)


if "cluster" in recommendations_with_cluster.columns:

    unique_clusters = (
        recommendations_with_cluster["cluster"]
        .nunique()
    )

    total_clusters = (
        papers["cluster"]
        .nunique()
    )

    evaluation["unique_clusters_recommended"] = (
        unique_clusters
    )

    evaluation["total_clusters_in_corpus"] = (
        total_clusters
    )

    evaluation["cluster_coverage_percent"] = (
        unique_clusters
        / total_clusters
        * 100
    )


# ---------------------------------------------------------
# 5. Year diversity
# ---------------------------------------------------------

evaluation["unique_years"] = (
    recommendations["year"].nunique()
)


evaluation["year_range"] = (
    f"{recommendations['year'].min()}-"
    f"{recommendations['year'].max()}"
)


# ---------------------------------------------------------
# 6. Citation coverage
# ---------------------------------------------------------

if "citationCount" in recommendations.columns:

    evaluation["papers_with_citations"] = (
        (
            recommendations["citationCount"] > 0
        ).sum()
    )

    evaluation["citation_coverage_percent"] = (
        (
            recommendations["citationCount"] > 0
        ).mean()
        * 100
    )

    evaluation["mean_citations"] = (
        recommendations["citationCount"].mean()
    )

    evaluation["maximum_citations"] = (
        recommendations["citationCount"].max()
    )


# ---------------------------------------------------------
# 7. Semantic diversity
# ---------------------------------------------------------

paper_to_index = {
    paper_id: index
    for index, paper_id
    in enumerate(papers["paperId"])
}


recommended_indices = []

for paper_id in recommendations["paperId"]:

    if paper_id in paper_to_index:

        recommended_indices.append(
            paper_to_index[paper_id]
        )


recommended_embeddings = embeddings[
    recommended_indices
]


if len(recommended_embeddings) > 1:

    similarity_matrix = cosine_similarity(
        recommended_embeddings
    )

    upper_triangle = similarity_matrix[
        np.triu_indices(
            len(recommended_embeddings),
            k=1,
        )
    ]

    mean_pairwise_similarity = (
        upper_triangle.mean()
    )

else:

    mean_pairwise_similarity = 0.0


evaluation[
    "mean_pairwise_semantic_similarity"
] = mean_pairwise_similarity


evaluation[
    "semantic_diversity_score"
] = (
    1.0 - mean_pairwise_similarity
)


# ---------------------------------------------------------
# 8. Knowledge graph coverage
# ---------------------------------------------------------

if "graph_degree" in recommendations.columns:

    graph_connected = (
        recommendations["graph_degree"] > 0
    )

    evaluation["graph_connected_papers"] = (
        graph_connected.sum()
    )

    evaluation["graph_coverage_percent"] = (
        graph_connected.mean() * 100
    )


# ---------------------------------------------------------
# 9. Related-paper coverage
# ---------------------------------------------------------

if "related_paper_count" in recommendations.columns:

    related_available = (
        recommendations["related_paper_count"] > 0
    )

    evaluation[
        "papers_with_related_graph_papers"
    ] = related_available.sum()

    evaluation[
        "related_paper_coverage_percent"
    ] = (
        related_available.mean()
        * 100
    )


# ---------------------------------------------------------
# 10. Collaboration coverage
# ---------------------------------------------------------

if "coauthor_count" in recommendations.columns:

    collaboration_connected = (
        recommendations["coauthor_count"] > 0
    )

    evaluation[
        "papers_with_coauthor_connections"
    ] = collaboration_connected.sum()

    evaluation[
        "coauthor_connection_percent"
    ] = (
        collaboration_connected.mean()
        * 100
    )


# ---------------------------------------------------------
# 11. Ranking consistency
# ---------------------------------------------------------

expected_ranks = list(
    range(
        1,
        len(recommendations) + 1
    )
)


actual_ranks = (
    recommendations["final_rank"]
    .tolist()
)


evaluation["ranking_order_valid"] = (
    actual_ranks == expected_ranks
)


# ---------------------------------------------------------
# Evaluation table
# ---------------------------------------------------------

evaluation_df = pd.DataFrame(
    [evaluation]
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

evaluation_df.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)


for metric, value in evaluation.items():

    if isinstance(
        value,
        (float, np.floating)
    ):

        print(
            f"{metric}: {value:.4f}"
        )

    else:

        print(
            f"{metric}: {value}"
        )


# ---------------------------------------------------------
# Cluster distribution
# ---------------------------------------------------------

if "cluster" in recommendations_with_cluster.columns:

    print("\n" + "=" * 70)
    print("RECOMMENDED CLUSTER DISTRIBUTION")
    print("=" * 70)

    print(
        recommendations_with_cluster[
            "cluster"
        ].value_counts().sort_index()
    )


# ---------------------------------------------------------
# Final status
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RECOMMENDATION EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)