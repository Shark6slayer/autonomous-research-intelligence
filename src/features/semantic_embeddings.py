from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "papers_clean.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_embeddings.npy"
)


MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32


def main():

    print()
    print("========== ARI SEMANTIC EMBEDDINGS ==========")
    print()

    df = pd.read_parquet(INPUT_FILE)

    print(f"Papers loaded: {len(df)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = (
        df["text"]
        .fillna("")
        .tolist()
    )

    print(
        f"Generating embeddings "
        f"with batch size {BATCH_SIZE}..."
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print()
    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    print("Saving embeddings...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_FILE,
        embeddings,
    )

    print()
    print(
        f"Saved embeddings to:"
    )
    print(OUTPUT_FILE)

    print()
    print("==============================================")
    print()


if __name__ == "__main__":
    main()