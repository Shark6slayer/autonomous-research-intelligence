from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TFIDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_features.npz"
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
    / "discovered_topics.parquet"
)


N_TOP_TERMS = 15


def main():

    print()
    print("========== ARI TOPIC DISCOVERY ==========")
    print()

    matrix = load_npz(
        TFIDF_FILE
    )

    df = pd.read_parquet(
        CLUSTERED_FILE
    )

    print(
        f"TF-IDF matrix: {matrix.shape}"
    )

    print(
        f"Papers: {len(df)}"
    )

    if matrix.shape[0] != len(df):
        raise ValueError(
            "TF-IDF and dataset row counts do not match."
        )

    # Recover the TF-IDF vocabulary
    #
    # The original vectorizer vocabulary is not stored,
    # so we rebuild it using the same configuration.
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )

    print("Rebuilding TF-IDF vocabulary...")

    vectorizer.fit(
        df["text"].fillna("").tolist()
    )

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    print(
        f"Vocabulary size: {len(feature_names)}"
    )

    results = []

    print()
    print("Discovering topics...")

    for cluster_id in sorted(
        df["cluster"].unique()
    ):

        cluster_mask = (
            df["cluster"].values
            == cluster_id
        )

        cluster_matrix = matrix[
            cluster_mask
        ]

        # Average TF-IDF weight for each term
        mean_scores = np.asarray(
            cluster_matrix.mean(
                axis=0
            )
        ).ravel()

        top_indices = np.argsort(
            mean_scores
        )[::-1][:N_TOP_TERMS]

        print()
        print(
            f"Cluster {cluster_id}"
        )

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            term = feature_names[index]
            score = mean_scores[index]

            results.append(
                {
                    "cluster": int(cluster_id),
                    "rank": rank,
                    "term": term,
                    "tfidf_score": float(score),
                }
            )

            print(
                f"  {rank:2d}. "
                f"{term:<35} "
                f"{score:.5f}"
            )

    results_df = pd.DataFrame(
        results
    )

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
        f"Saved discovered topics to:"
    )
    print(OUTPUT_FILE)

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()