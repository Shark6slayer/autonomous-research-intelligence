from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustered_papers.parquet"
)

TRENDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_trends.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_intelligence.parquet"
)


REFERENCE_YEAR = 2025


def min_max_normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def main():

    print()
    print("========== ARI RESEARCH INTELLIGENCE ==========")
    print()

    papers = pd.read_parquet(
        PAPERS_FILE
    )

    trends = pd.read_parquet(
        TRENDS_FILE
    )

    print(
        f"Papers loaded: {len(papers)}"
    )

    print(
        f"Topics loaded: {len(trends)}"
    )

    required_columns = {
        "paperId",
        "title",
        "year",
        "citationCount",
        "cluster",
        "semantic_quality",
    }

    missing = (
        required_columns
        - set(papers.columns)
    )

    if missing:

        raise ValueError(
            f"Missing paper columns: {missing}"
        )

    # --------------------------------------------------
    # Merge topic intelligence
    # --------------------------------------------------

    topic_columns = [
        "cluster",
        "topic",
        "emerging_score",
        "recent_growth_percent",
        "latest_year_share_percent",
    ]

    papers = papers.merge(
        trends[topic_columns],
        on="cluster",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------
    # Recency
    # --------------------------------------------------

    papers["age"] = (
        REFERENCE_YEAR
        - papers["year"]
    )

    papers["age"] = (
        papers["age"]
        .clip(lower=0)
    )

    papers["recency_score"] = np.exp(
        -0.35
        * papers["age"]
    )

    # --------------------------------------------------
    # Age-aware citation impact
    # --------------------------------------------------

    papers["citations_per_year"] = (
        papers["citationCount"]
        / (
            papers["age"] + 1
        )
    )

    papers["citation_score"] = np.log1p(
        papers["citations_per_year"]
        .clip(lower=0)
    )

    papers["citation_score"] = (
        min_max_normalize(
            papers["citation_score"]
        )
    )

    # --------------------------------------------------
    # Topic emergence
    # --------------------------------------------------

    papers["topic_emergence_score"] = (
        papers["emerging_score"]
        .fillna(0.0)
    )

    # --------------------------------------------------
    # Semantic / text quality
    # --------------------------------------------------

    quality_mapping = {
        "Very Short": 0.25,
        "Short": 0.50,
        "Medium": 0.80,
        "Long": 1.00,
    }

    papers["quality_score"] = (
        papers["semantic_quality"]
        .astype(str)
        .map(quality_mapping)
        .fillna(0.50)
        .astype(float)
    )

    # --------------------------------------------------
    # Final intelligence score
    # --------------------------------------------------

    papers["intelligence_score"] = (
        0.30
        * papers["recency_score"]
        + 0.30
        * papers["citation_score"]
        + 0.20
        * papers["topic_emergence_score"]
        + 0.20
        * papers["quality_score"]
    )

    # --------------------------------------------------
    # Rank
    # --------------------------------------------------

    papers = (
        papers
        .sort_values(
            "intelligence_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    papers["intelligence_rank"] = (
        papers.index + 1
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    papers.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Display top papers
    # --------------------------------------------------

    print()
    print(
        "========== TOP RESEARCH INTELLIGENCE =========="
    )

    for _, row in papers.head(15).iterrows():

        print()
        print(
            f"Rank {int(row['intelligence_rank'])}"
        )

        print(
            f"Title: {row['title']}"
        )

        print(
            f"Topic: {row['topic']}"
        )

        print(
            f"Year: {int(row['year'])}"
        )

        print(
            f"Citations: "
            f"{int(row['citationCount'])}"
        )

        print(
            f"Citations/year: "
            f"{row['citations_per_year']:.2f}"
        )

        print(
            f"Intelligence score: "
            f"{row['intelligence_score']:.4f}"
        )

    print()
    print(
        "Saved research intelligence to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "================================================"
    )
    print()


if __name__ == "__main__":
    main()