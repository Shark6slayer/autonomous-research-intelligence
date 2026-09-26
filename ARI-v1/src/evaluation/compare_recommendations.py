from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

BASELINE_FILE = DATA_DIR / "recommendation_results.parquet"
MMR_FILE = DATA_DIR / "diverse_recommendations.parquet"
PAPERS_FILE = DATA_DIR / "papers_clean.parquet"
EMBEDDINGS_FILE = DATA_DIR / "semantic_embeddings.npy"

OUTPUT_FILE = DATA_DIR / "recommendation_comparison.parquet"


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def calculate_metrics(
    recommendations,
    paper_to_index,
    normalized_embeddings,
):
    recommendations = recommendations.copy()

    # Map papers to embedding rows
    recommendations["embedding_index"] = (
        recommendations["paperId"].map(paper_to_index)
    )

    recommendations = recommendations.dropna(
        subset=["embedding_index"]
    ).copy()

    recommendations["embedding_index"] = (
        recommendations["embedding_index"].astype(int)
    )

    # Semantic similarity to query
    mean_semantic_similarity = float(
        recommendations["similarity"].mean()
    )

    # Recommendation score
    mean_recommendation_score = float(
        recommendations["recommendation_score"].mean()
    )

    # Cluster coverage
    if "cluster" in recommendations.columns:
        unique_clusters = int(
            recommendations["cluster"].nunique()
        )
    else:
        unique_clusters = 0

    # Pairwise semantic similarity
    selected_embeddings = normalized_embeddings[
        recommendations["embedding_index"].values
    ]

    similarity_matrix = (
        selected_embeddings @ selected_embeddings.T
    )

    n = len(recommendations)

    if n > 1:
        pairwise_values = similarity_matrix[
            np.triu_indices(n, k=1)
        ]

        mean_pairwise_similarity = float(
            pairwise_values.mean()
        )

        semantic_diversity = (
            1 - mean_pairwise_similarity
        )
    else:
        mean_pairwise_similarity = 0.0
        semantic_diversity = 0.0

    return {
        "recommendation_count": len(recommendations),
        "mean_recommendation_score": mean_recommendation_score,
        "mean_semantic_similarity": mean_semantic_similarity,
        "mean_pairwise_semantic_similarity": mean_pairwise_similarity,
        "semantic_diversity_score": semantic_diversity,
        "unique_clusters": unique_clusters,
    }


def main():

    print("=" * 70)
    print("BASELINE VS MMR RECOMMENDATION COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    baseline = pd.read_parquet(BASELINE_FILE)
    mmr = pd.read_parquet(MMR_FILE)

    papers = pd.read_parquet(PAPERS_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Baseline recommendations: {len(baseline)}")
    print(f"MMR recommendations: {len(mmr)}")
    print(f"Embeddings: {embeddings.shape}")

    # ---------------------------------------------------------
    # Prepare embedding lookup
    # ---------------------------------------------------------

    paper_to_index = {
        paper_id: index
        for index, paper_id in enumerate(
            papers["paperId"]
        )
    }

    normalized_embeddings = normalize_embeddings(
        embeddings
    )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    baseline_metrics = calculate_metrics(
        baseline,
        paper_to_index,
        normalized_embeddings,
    )

    mmr_metrics = calculate_metrics(
        mmr,
        paper_to_index,
        normalized_embeddings,
    )

    # ---------------------------------------------------------
    # Print comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    metrics = [
        "recommendation_count",
        "mean_recommendation_score",
        "mean_semantic_similarity",
        "mean_pairwise_semantic_similarity",
        "semantic_diversity_score",
        "unique_clusters",
    ]

    print(
        f"{'Metric':<40}"
        f"{'Baseline':>15}"
        f"{'MMR':>15}"
    )

    print("-" * 70)

    rows = []

    for metric in metrics:

        baseline_value = baseline_metrics[metric]
        mmr_value = mmr_metrics[metric]

        print(
            f"{metric:<40}"
            f"{baseline_value:>15.4f}"
            f"{mmr_value:>15.4f}"
        )

        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "mmr": mmr_value,
                "change": mmr_value - baseline_value,
            }
        )

    comparison = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Percentage improvement for diversity
    # ---------------------------------------------------------

    baseline_diversity = baseline_metrics[
        "semantic_diversity_score"
    ]

    mmr_diversity = mmr_metrics[
        "semantic_diversity_score"
    ]

    if baseline_diversity != 0:
        diversity_improvement = (
            (mmr_diversity - baseline_diversity)
            / baseline_diversity
        ) * 100
    else:
        diversity_improvement = 0.0

    print("\n" + "=" * 70)
    print("KEY RESULT")
    print("=" * 70)

    print(
        f"Semantic diversity improvement: "
        f"{diversity_improvement:.2f}%"
    )

    # ---------------------------------------------------------
    # Recommendation overlap
    # ---------------------------------------------------------

    baseline_ids = set(baseline["paperId"])
    mmr_ids = set(mmr["paperId"])

    overlap = baseline_ids.intersection(mmr_ids)

    overlap_percent = (
        len(overlap) / max(len(baseline_ids), 1)
    ) * 100

    print(
        f"Recommendation overlap: "
        f"{len(overlap)}/{len(baseline_ids)} "
        f"({overlap_percent:.2f}%)"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    comparison.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nComparison complete.")


if __name__ == "__main__":
    main()