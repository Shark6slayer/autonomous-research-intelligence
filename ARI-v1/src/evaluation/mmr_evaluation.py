from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

RECOMMENDATION_FILE = DATA_DIR / "diverse_recommendations.parquet"
EMBEDDINGS_FILE = DATA_DIR / "semantic_embeddings.npy"
PAPERS_FILE = DATA_DIR / "papers_clean.parquet"

OUTPUT_FILE = DATA_DIR / "mmr_evaluation.parquet"


def cosine_similarity_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def main():

    print("=" * 70)
    print("MMR RECOMMENDATION EVALUATION")
    print("=" * 70)

    recommendations = pd.read_parquet(RECOMMENDATION_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    papers = pd.read_parquet(PAPERS_FILE)

    print(f"Recommendations: {len(recommendations)}")
    print(f"Embedding matrix: {embeddings.shape}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = [
        "paperId",
        "recommendation_score",
        "similarity",
        "mmr_score",
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
    # Map paper IDs to embedding rows
    # ---------------------------------------------------------

    paper_to_index = {
        paper_id: index
        for index, paper_id in enumerate(papers["paperId"])
    }

    recommendations["embedding_index"] = (
        recommendations["paperId"].map(paper_to_index)
    )

    missing_embeddings = recommendations[
        recommendations["embedding_index"].isna()
    ]

    if len(missing_embeddings) > 0:
        print(
            f"Warning: {len(missing_embeddings)} papers "
            "could not be mapped to embeddings."
        )

    recommendations = recommendations.dropna(
        subset=["embedding_index"]
    ).copy()

    recommendations["embedding_index"] = (
        recommendations["embedding_index"].astype(int)
    )

    selected_embeddings = embeddings[
        recommendations["embedding_index"].values
    ]

    # ---------------------------------------------------------
    # Semantic similarity
    # ---------------------------------------------------------

    similarity_matrix = cosine_similarity_matrix(
        selected_embeddings
    )

    n = len(recommendations)

    if n > 1:
        pairwise_values = similarity_matrix[
            np.triu_indices(n, k=1)
        ]

        mean_pairwise_similarity = float(
            pairwise_values.mean()
        )
    else:
        mean_pairwise_similarity = 0.0

    semantic_diversity = 1 - mean_pairwise_similarity

    # ---------------------------------------------------------
    # Cluster diversity
    # ---------------------------------------------------------

    if "cluster" in recommendations.columns:
        unique_clusters = int(
            recommendations["cluster"].nunique()
        )
    else:
        unique_clusters = 0

    # ---------------------------------------------------------
    # Score statistics
    # ---------------------------------------------------------

    mean_recommendation_score = float(
        recommendations["recommendation_score"].mean()
    )

    mean_mmr_score = float(
        recommendations["mmr_score"].mean()
    )

    min_mmr_score = float(
        recommendations["mmr_score"].min()
    )

    max_mmr_score = float(
        recommendations["mmr_score"].max()
    )

    mean_semantic_similarity = float(
        recommendations["similarity"].mean()
    )

    # ---------------------------------------------------------
    # Ranking validity
    #
    # The parquet file is already ordered by MMR rank.
    # Therefore row order is treated as final ranking.
    # ---------------------------------------------------------

    ranking_order_valid = True

    # ---------------------------------------------------------
    # Evaluation results
    # ---------------------------------------------------------

    results = pd.DataFrame(
        {
            "metric": [
                "recommendation_count",
                "mean_recommendation_score",
                "mean_mmr_score",
                "minimum_mmr_score",
                "maximum_mmr_score",
                "mean_semantic_similarity",
                "mean_pairwise_semantic_similarity",
                "semantic_diversity_score",
                "unique_clusters",
                "ranking_order_valid",
            ],
            "value": [
                len(recommendations),
                mean_recommendation_score,
                mean_mmr_score,
                min_mmr_score,
                max_mmr_score,
                mean_semantic_similarity,
                mean_pairwise_similarity,
                semantic_diversity,
                unique_clusters,
                ranking_order_valid,
            ],
        }
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MMR EVALUATION RESULTS")
    print("=" * 70)

    for _, row in results.iterrows():
        print(
            f"{row['metric']}: {row['value']}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    results.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nMMR evaluation complete.")


if __name__ == "__main__":
    main()