from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TFIDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_features.npz"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_metadata.parquet"
)


CLUSTER_COUNTS = [5, 8, 10, 12, 15]


def main():

    print()
    print("========== ARI K-MEANS BASELINE ==========")
    print()

    matrix = load_npz(TFIDF_FILE)

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    print(
        f"TF-IDF matrix: {matrix.shape}"
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    print()

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

        labels = model.fit_predict(matrix)

        score = silhouette_score(
            matrix,
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

    results_df = pd.DataFrame(results)

    print("========== RESULTS ==========")

    print(
        results_df.to_string(
            index=False
        )
    )

    best_row = results_df.loc[
        results_df["silhouette_score"].idxmax()
    ]

    print()
    print(
        f"Best k: {int(best_row['k'])}"
    )

    print(
        f"Best silhouette score: "
        f"{best_row['silhouette_score']:.4f}"
    )

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()