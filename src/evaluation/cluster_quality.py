from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)

CLUSTERED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cluster_quality.parquet"
)

BEST_K = 5
SAMPLE_SIZE = 3000


def main():

    print()
    print("========== ARI CLUSTER QUALITY ==========")
    print()

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    df = pd.read_parquet(
        CLUSTERED_FILE
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"Papers: {len(df)}"
    )

    if len(embeddings) != len(df):
        raise ValueError(
            "Embedding and dataset counts do not match."
        )

    print(
        f"\nEvaluating k={BEST_K}..."
    )

    model = KMeans(
        n_clusters=BEST_K,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(
        embeddings
    )

    # Use the same sample size for all metrics
    rng = np.random.default_rng(42)

    if len(embeddings) > SAMPLE_SIZE:
        sample_indices = rng.choice(
            len(embeddings),
            size=SAMPLE_SIZE,
            replace=False,
        )
    else:
        sample_indices = np.arange(
            len(embeddings)
        )

    sampled_embeddings = embeddings[
        sample_indices
    ]

    sampled_labels = labels[
        sample_indices
    ]

    print(
        f"Evaluation samples: "
        f"{len(sampled_embeddings)}"
    )

    print()
    print("Calculating metrics...")

    silhouette = silhouette_score(
        sampled_embeddings,
        sampled_labels,
    )

    davies_bouldin = davies_bouldin_score(
        sampled_embeddings,
        sampled_labels,
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            sampled_embeddings,
            sampled_labels,
        )
    )

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    largest_cluster = (
        cluster_counts.max()
    )

    smallest_cluster = (
        cluster_counts.min()
    )

    balance_ratio = (
        smallest_cluster
        / largest_cluster
    )

    results = pd.DataFrame(
        [
            {
                "k": BEST_K,
                "silhouette_score": silhouette,
                "davies_bouldin_index": davies_bouldin,
                "calinski_harabasz_score": calinski_harabasz,
                "largest_cluster": largest_cluster,
                "smallest_cluster": smallest_cluster,
                "balance_ratio": balance_ratio,
            }
        ]
    )

    print()
    print("========== QUALITY METRICS ==========")

    print(
        f"Silhouette Score: "
        f"{silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{davies_bouldin:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{calinski_harabasz:.2f}"
    )

    print(
        f"Largest cluster: "
        f"{largest_cluster}"
    )

    print(
        f"Smallest cluster: "
        f"{smallest_cluster}"
    )

    print(
        f"Balance ratio: "
        f"{balance_ratio:.4f}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved quality metrics to:"
    )
    print(OUTPUT_FILE)

    print()
    print("========================================")
    print()


if __name__ == "__main__":
    main()