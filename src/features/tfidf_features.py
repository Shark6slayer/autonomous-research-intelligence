from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "papers_clean.parquet"
)

OUTPUT_MATRIX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_features.npz"
)

OUTPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_metadata.parquet"
)

OUTPUT_VOCAB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tfidf_vocabulary.npy"
)


def main():

    print()
    print("========== ARI TF-IDF FEATURES ==========")
    print()

    df = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Papers loaded: {len(df)}"
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )

    print(
        "Building TF-IDF matrix..."
    )

    matrix = vectorizer.fit_transform(
        df["text"].fillna("").tolist()
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    print(
        f"Matrix shape: {matrix.shape}"
    )

    print(
        f"Non-zero values: {matrix.nnz}"
    )

    print(
        f"Vocabulary size: {len(feature_names)}"
    )

    # --------------------------------------------------
    # Save TF-IDF matrix
    # --------------------------------------------------

    save_npz(
        OUTPUT_MATRIX,
        matrix
    )

    print()
    print(
        f"Saved matrix to:"
    )
    print(
        OUTPUT_MATRIX
    )

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    metadata_columns = [
        "paperId",
        "title",
        "year",
        "citationCount",
        "semantic_quality",
    ]

    metadata = df[
        metadata_columns
    ].copy()

    metadata.to_parquet(
        OUTPUT_METADATA,
        index=False,
    )

    print(
        "Saved metadata to:"
    )
    print(
        OUTPUT_METADATA
    )

    # --------------------------------------------------
    # Save vocabulary
    # --------------------------------------------------

    np.save(
        OUTPUT_VOCAB,
        feature_names
    )

    print(
        "Saved vocabulary to:"
    )
    print(
        OUTPUT_VOCAB
    )

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()