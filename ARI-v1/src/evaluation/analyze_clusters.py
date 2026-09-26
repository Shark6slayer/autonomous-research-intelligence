from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "papers_clean.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

BEST_K = 5


def main():

    print()
    print("========== ARI CLUSTER ANALYSIS ==========")
    print()

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    df = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Papers: {len(df)}"
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    if len(df) != len(embeddings):
        raise ValueError(
            "Dataset and embedding counts do not match."
        )

    print(
        f"\nTraining final K-Means model with k={BEST_K}..."
    )

    model = KMeans(
        n_clusters=BEST_K,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(
        embeddings
    )

    df["cluster"] = labels

    print()
    print("========== CLUSTER SIZES ==========")

    cluster_sizes = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster_id, count in cluster_sizes.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"Cluster {cluster_id}: "
            f"{count} papers "
            f"({percentage:.2f}%)"
        )

    print()
    print("========== CLUSTER YEARS ==========")

    year_summary = (
        df.groupby("cluster")["year"]
        .agg(["min", "max", "mean"])
        .round(2)
    )

    print(
        year_summary
    )

    print()
    print("========== CLUSTER CITATIONS ==========")

    citation_summary = (
        df.groupby("cluster")["citationCount"]
        .agg(["mean", "median", "max"])
        .round(2)
    )

    print(
        citation_summary
    )

    print()
    print("========== DOMINANT FIELDS ==========")

    for cluster_id in sorted(
        df["cluster"].unique()
    ):

        cluster_df = df[
            df["cluster"] == cluster_id
        ]

        fields = (
            cluster_df["fieldsOfStudy"]
            .explode()
            .dropna()
            .value_counts()
            .head(5)
        )

        print()
        print(
            f"Cluster {cluster_id}:"
        )

        if len(fields) == 0:
            print("  No field data available.")
        else:
            for field, count in fields.items():

                print(
                    f"  {field}: {count}"
                )

    print()
    print("========== REPRESENTATIVE PAPERS ==========")

    for cluster_id in sorted(
        df["cluster"].unique()
    ):

        cluster_df = df[
            df["cluster"] == cluster_id
        ]

        representatives = (
            cluster_df
            .sort_values(
                "citationCount",
                ascending=False,
            )
            .head(5)
        )

        print()
        print(
            f"Cluster {cluster_id}:"
        )

        for _, row in representatives.iterrows():

            print(
                f"  - {row['title']} "
                f"({row['year']}, "
                f"{row['citationCount']} citations)"
            )

    print()
    print("Saving clustered dataset...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()