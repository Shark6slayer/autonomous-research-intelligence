from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_metadata.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_cluster_results.parquet"
)


CLUSTER_COUNTS = [5, 8, 10, 12, 15]


def main():

    print()
    print("========== ARI SEMANTIC CLUSTERING ==========")
    print()

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding and metadata row counts do not match."
        )

    results = []

    for k in CLUSTER_COUNTS:

        print(
            f"Training K-Means with k={k}..."
        )

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(
            embeddings
        )

        score = silhouette_score(
            embeddings,
            labels,
            sample_size=3000,
            random_state=42,
        )

        results.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

        print(
            f"Silhouette score: {score:.4f}"
        )

        print()

    results_df = pd.DataFrame(
        results
    )

    print("========== RESULTS ==========")

    print(
        results_df.to_string(
            index=False
        )
    )

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    best_k = int(
        best_row["k"]
    )

    best_score = float(
        best_row["silhouette_score"]
    )

    print()
    print(
        f"Best k: {best_k}"
    )

    print(
        f"Best silhouette score: "
        f"{best_score:.4f}"
    )

    # Save evaluation results
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved results to:"
    )
    print(OUTPUT_FILE)

    print()
    print("============================================")
    print()


if __name__ == "__main__":
    main()