import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dsai_corpus_raw.json"


def load_data():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["data"]


def main():
    papers = load_data()

    df = pd.DataFrame(papers)

    print("\n========== ARI DATASET PROFILE ==========\n")

    print(f"Total papers: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate paper IDs:")
    print(df["paperId"].duplicated().sum())

    print("\nDuplicate titles:")
    print(df["title"].duplicated().sum())

    print("\nPublication years:")
    print(df["year"].value_counts(dropna=False).sort_index())

    print("\nFields of study:")
    print(
        df["fieldsOfStudy"]
        .explode()
        .value_counts()
        .head(15)
    )

    print("\nCitation statistics:")
    print(df["citationCount"].describe())

    print("\nAbstract availability:")
    print(
        df["abstract"]
        .notna()
        .value_counts()
    )

    print("\nTop 10 papers by citation count:")

    top_papers = df[
        ["title", "year", "citationCount"]
    ].sort_values(
        "citationCount",
        ascending=False,
    ).head(10)

    print(top_papers.to_string(index=False))

    print("\n==========================================\n")


if __name__ == "__main__":
    main()