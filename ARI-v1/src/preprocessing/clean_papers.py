import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dsai_corpus_raw.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "papers_clean.parquet"


def load_raw_data():
    """Load papers from the raw JSON file."""

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["data"]


def clean_text(text):
    """Normalize text for downstream NLP processing."""

    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_dataframe(papers):
    """Convert raw records into a clean research DataFrame."""

    df = pd.DataFrame(papers)

    # Remove duplicate paper IDs
    before_ids = len(df)

    df = df.drop_duplicates(
        subset="paperId",
        keep="first",
    )

    removed_id_duplicates = before_ids - len(df)

    # Clean title
    df["title"] = df["title"].apply(clean_text)

    # Clean abstract
    df["abstract"] = df["abstract"].apply(clean_text)

    # Track abstract availability
    df["has_abstract"] = (
        df["abstract"].str.len() > 0
    )

    # Create combined research text
    df["text"] = (
        df["title"] + ". " + df["abstract"]
    ).str.strip()

    # Calculate text length
    df["text_length"] = (
        df["text"]
        .str.split()
        .str.len()
    )

    # Assign text-quality categories
    df["semantic_quality"] = pd.cut(
        df["text_length"],
        bins=[-1, 29, 99, 199, float("inf")],
        labels=[
            "Very Short",
            "Short",
            "Medium",
            "Long",
        ],
    )

    # Remove records without meaningful text
    before_text_filter = len(df)

    df = df[df["text_length"] >= 5]

    removed_text_records = (
        before_text_filter - len(df)
    )

    # Remove duplicate titles
    before_titles = len(df)

    df = df.drop_duplicates(
        subset="title",
        keep="first",
    )

    removed_title_duplicates = (
        before_titles - len(df)
    )

    # Reset index
    df = df.reset_index(drop=True)

    print()
    print("Cleaning summary:")
    print(
        f"  Paper ID duplicates removed: "
        f"{removed_id_duplicates}"
    )
    print(
        f"  Records with insufficient text removed: "
        f"{removed_text_records}"
    )
    print(
        f"  Duplicate titles removed: "
        f"{removed_title_duplicates}"
    )

    return df


def save_processed_data(df):
    """Save the processed dataset as Parquet."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )


def main():

    print()
    print("========== ARI PREPROCESSING ==========")
    print()

    papers = load_raw_data()

    print(
        f"Raw papers: {len(papers)}"
    )

    df = prepare_dataframe(papers)

    print(
        f"Processed papers: {len(df)}"
    )

    print(
        f"Total removed: "
        f"{len(papers) - len(df)}"
    )

    print()
    print("Final columns:")

    for column in df.columns:
        print(f"  - {column}")

    print()
    print("Abstract availability:")

    print(
        df["has_abstract"]
        .value_counts()
    )

    print()
    print("Semantic quality:")

    print(
        df["semantic_quality"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Text length statistics:")

    print(
        df["text_length"]
        .describe()
    )

    save_processed_data(df)

    print()
    print(
        f"Saved processed dataset to:"
    )
    print(OUTPUT_FILE)

    print()
    print("========================================")
    print()


if __name__ == "__main__":
    main()