from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "distinctive_topics.parquet"
)

N_TOP_TERMS = 15


GENERIC_TERMS = {
    "learning",
    "model",
    "models",
    "data",
    "using",
    "based",
    "method",
    "methods",
    "approach",
    "study",
    "results",
    "analysis",
    "paper",
    "research",
    "proposed",
    "used",
    "use",
    "new",
    "different",
    "performance",
}


def main():

    print()
    print("========== ARI DISTINCTIVE TOPICS ==========")
    print()

    df = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Papers loaded: {len(df)}"
    )

    if "cluster" not in df.columns:
        raise ValueError(
            "Cluster column not found."
        )

    # --------------------------------------------------
    # Create one document per cluster
    # --------------------------------------------------

    cluster_documents = (
        df.groupby("cluster")["text"]
        .apply(
            lambda texts: " ".join(
                texts.fillna("")
            )
        )
    )

    cluster_ids = (
        cluster_documents.index
        .tolist()
    )

    print(
        f"Clusters: {len(cluster_ids)}"
    )

    # --------------------------------------------------
    # TF-IDF over cluster documents
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        cluster_documents.tolist()
    )

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    print(
        f"Cluster-term matrix: {matrix.shape}"
    )

    print()
    print(
        "Calculating cluster-distinctive terms..."
    )

    results = []

    # --------------------------------------------------
    # Calculate distinctive terms
    # --------------------------------------------------

    for row_index, cluster_id in enumerate(
        cluster_ids
    ):

        scores = matrix[
            row_index
        ].toarray().ravel()

        # Remove generic terms
        for index, term in enumerate(
            feature_names
        ):

            if term in GENERIC_TERMS:
                scores[index] = 0.0

        top_indices = np.argsort(
            scores
        )[::-1][:N_TOP_TERMS]

        print()
        print(
            f"Cluster {cluster_id}"
        )

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            term = str(
                feature_names[index]
            )

            score = float(
                scores[index]
            )

            results.append(
                {
                    "cluster": int(cluster_id),
                    "rank": rank,
                    "term": term,
                    "distinctiveness_score": score,
                }
            )

            print(
                f"  {rank:2d}. "
                f"{term:<35} "
                f"{score:.4f}"
            )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

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
        "Saved distinctive topics to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("============================================")
    print()


if __name__ == "__main__":
    main()