from pathlib import Path

import numpy as np
import pandas as pd


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
    / "research_trends.parquet"
)


CLUSTER_LABELS = {
    0: "Computer Vision & Deep Learning",
    1: "Generative AI & AI in Education",
    2: "Data Mining & Machine Learning",
    3: "Reinforcement Learning & Control",
    4: "Natural Language Processing",
}


GENERIC_GROWTH_CAP = 300.0


def calculate_growth(
    previous_value,
    current_value,
):
    if previous_value == 0:

        if current_value > 0:
            return np.inf

        return 0.0

    return (
        (current_value - previous_value)
        / previous_value
    ) * 100


def main():

    print()
    print("========== ARI RESEARCH TREND DETECTION ==========")
    print()

    df = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Papers loaded: {len(df)}"
    )

    required_columns = {
        "cluster",
        "year",
        "citationCount",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[
        df["year"].notna()
    ].copy()

    df["year"] = (
        df["year"]
        .astype(int)
    )

    years = sorted(
        df["year"].unique()
    )

    print(
        f"Available years: "
        f"{years[0]} - {years[-1]}"
    )

    # --------------------------------------------------
    # Use only completed years for annual trend analysis
    # --------------------------------------------------

    analysis_years = [
        year
        for year in years
        if year < 2026
    ]

    first_year = analysis_years[0]
    latest_year = analysis_years[-1]
    previous_year = latest_year - 1

    print(
        f"Trend analysis period: "
        f"{first_year} - {latest_year}"
    )

    print(
        f"Latest complete year: "
        f"{latest_year}"
    )

    print(
        f"2026 papers retained in dataset "
        f"but excluded from annual growth ranking."
    )

    # --------------------------------------------------
    # Overall publication counts
    # --------------------------------------------------

    yearly_total = (
        df[
            df["year"].isin(
                analysis_years
            )
        ]
        .groupby("year")
        .size()
        .to_dict()
    )

    results = []

    print()
    print(
        "Calculating topic trends..."
    )

    # --------------------------------------------------
    # Topic analysis
    # --------------------------------------------------

    for cluster_id in sorted(
        df["cluster"].unique()
    ):

        cluster_df = df[
            df["cluster"] == cluster_id
        ].copy()

        analysis_df = cluster_df[
            cluster_df["year"].isin(
                analysis_years
            )
        ]

        topic_name = CLUSTER_LABELS.get(
            int(cluster_id),
            f"Cluster {cluster_id}",
        )

        yearly_counts = (
            analysis_df
            .groupby("year")
            .size()
            .reindex(
                analysis_years,
                fill_value=0
            )
        )

        # --------------------------------------------------
        # Long-term growth
        # --------------------------------------------------

        first_count = int(
            yearly_counts.loc[
                first_year
            ]
        )

        latest_count = int(
            yearly_counts.loc[
                latest_year
            ]
        )

        overall_growth = calculate_growth(
            first_count,
            latest_count,
        )

        # --------------------------------------------------
        # Latest complete-year growth
        # --------------------------------------------------

        previous_count = int(
            yearly_counts.loc[
                previous_year
            ]
        )

        recent_growth = calculate_growth(
            previous_count,
            latest_count,
        )

        # --------------------------------------------------
        # Topic share
        # --------------------------------------------------

        latest_total = yearly_total[
            latest_year
        ]

        latest_share = (
            latest_count
            / latest_total
        ) * 100

        previous_total = yearly_total[
            previous_year
        ]

        previous_share = (
            previous_count
            / previous_total
        ) * 100

        share_change = (
            latest_share
            - previous_share
        )

        # --------------------------------------------------
        # Citation impact
        # --------------------------------------------------

        mean_citations = float(
            analysis_df[
                "citationCount"
            ].mean()
        )

        median_citations = float(
            analysis_df[
                "citationCount"
            ].median()
        )

        # --------------------------------------------------
        # Growth normalization
        # --------------------------------------------------

        if np.isinf(
            recent_growth
        ):

            growth_component = 1.0

        else:

            growth_component = min(
                max(
                    recent_growth,
                    0.0
                ),
                GENERIC_GROWTH_CAP,
            ) / GENERIC_GROWTH_CAP

        # --------------------------------------------------
        # Share component
        # --------------------------------------------------

        share_component = min(
            latest_share,
            30.0,
        ) / 30.0

        # --------------------------------------------------
        # Share momentum
        # --------------------------------------------------

        share_momentum = min(
            max(
                share_change,
                0.0
            ),
            10.0,
        ) / 10.0

        # --------------------------------------------------
        # Emerging score
        # --------------------------------------------------

        emerging_score = (
            0.50 * growth_component
            + 0.30 * share_momentum
            + 0.20 * share_component
        )

        result = {
            "cluster": int(cluster_id),
            "topic": topic_name,
            "first_year": first_year,
            "latest_complete_year": latest_year,
            "first_year_papers": first_count,
            "latest_year_papers": latest_count,
            "overall_growth_percent": float(
                overall_growth
            ),
            "previous_year": previous_year,
            "previous_year_papers": previous_count,
            "recent_growth_percent": float(
                recent_growth
            ),
            "previous_year_share_percent": float(
                previous_share
            ),
            "latest_year_share_percent": float(
                latest_share
            ),
            "share_change_percentage_points": float(
                share_change
            ),
            "mean_citations": mean_citations,
            "median_citations": median_citations,
            "emerging_score": float(
                emerging_score
            ),
        }

        results.append(
            result
        )

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print()
        print(
            f"Cluster {cluster_id}: "
            f"{topic_name}"
        )

        print(
            f"  {first_year}: "
            f"{first_count} papers"
        )

        print(
            f"  {latest_year}: "
            f"{latest_count} papers"
        )

        print(
            f"  Long-term growth: "
            f"{overall_growth:.2f}%"
        )

        print(
            f"  {previous_year} -> "
            f"{latest_year}: "
            f"{recent_growth:.2f}%"
        )

        print(
            f"  Latest share: "
            f"{latest_share:.2f}%"
        )

        print(
            f"  Share change: "
            f"{share_change:+.2f} percentage points"
        )

        print(
            f"  Mean citations: "
            f"{mean_citations:.2f}"
        )

        print(
            f"  Emerging score: "
            f"{emerging_score:.4f}"
        )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    trends_df = pd.DataFrame(
        results
    )

    trends_df = (
        trends_df
        .sort_values(
            "emerging_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    trends_df[
        "trend_rank"
    ] = (
        trends_df.index + 1
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trends_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "========== TOP EMERGING TOPICS =========="
    )

    for _, row in trends_df.iterrows():

        print(
            f"{int(row['trend_rank'])}. "
            f"{row['topic']} "
            f"(score={row['emerging_score']:.4f})"
        )

    print()
    print(
        "Saved research trends to:"
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