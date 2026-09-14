from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_results.parquet"
)

GRAPH_CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_graph_context.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_recommendations.parquet"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

recommendations = pd.read_parquet(
    RECOMMENDATION_FILE
)

graph_context = pd.read_parquet(
    GRAPH_CONTEXT_FILE
)


print("Recommendation columns:")
print(recommendations.columns.tolist())

print("\nGraph context columns:")
print(graph_context.columns.tolist())


# ---------------------------------------------------------
# Select graph information
# ---------------------------------------------------------

graph_columns = [
    "paperId",
    "author_count",
    "coauthor_count",
    "related_paper_count",
    "graph_degree",
]

available_graph_columns = [
    column
    for column in graph_columns
    if column in graph_context.columns
]

graph_context = graph_context[
    available_graph_columns
]


# ---------------------------------------------------------
# Merge recommendation + knowledge graph
# ---------------------------------------------------------

final_df = recommendations.merge(
    graph_context,
    on="paperId",
    how="left",
)


# ---------------------------------------------------------
# Fill missing graph values
# ---------------------------------------------------------

for column in [
    "author_count",
    "coauthor_count",
    "related_paper_count",
    "graph_degree",
]:

    if column in final_df.columns:
        final_df[column] = final_df[column].fillna(0)


# ---------------------------------------------------------
# Generate explanation
# ---------------------------------------------------------

def generate_explanation(row):

    reasons = []

    similarity = row.get(
        "similarity",
        0
    )

    intelligence = row.get(
        "intelligence_score",
        0
    )

    citation = row.get(
        "citation_score",
        0
    )

    emergence = row.get(
        "topic_emergence_score",
        0
    )

    related = row.get(
        "related_paper_count",
        0
    )

    coauthors = row.get(
        "coauthor_count",
        0
    )


    # Semantic relevance

    if similarity >= 0.80:

        reasons.append(
            "very strong semantic relevance"
        )

    elif similarity >= 0.70:

        reasons.append(
            "strong semantic relevance"
        )

    else:

        reasons.append(
            "moderate semantic relevance"
        )


    # Research intelligence

    if intelligence >= 0.60:

        reasons.append(
            "high research-intelligence score"
        )

    elif intelligence >= 0.45:

        reasons.append(
            "good research-intelligence score"
        )


    # Citation impact

    if citation >= 0.50:

        reasons.append(
            "strong citation impact"
        )


    # Emerging topic

    if emergence >= 0.40:

        reasons.append(
            "emerging research area"
        )


    # Knowledge graph context

    if coauthors > 0:

        reasons.append(
            "connected through the collaboration graph"
        )

    if related > 0:

        reasons.append(
            f"{int(related)} related papers found in the knowledge graph"
        )


    return "; ".join(reasons)


final_df["explanation"] = final_df.apply(
    generate_explanation,
    axis=1,
)


# ---------------------------------------------------------
# Final ranking
# ---------------------------------------------------------

final_df = final_df.sort_values(
    by="recommendation_score",
    ascending=False,
).reset_index(drop=True)


final_df["final_rank"] = (
    range(1, len(final_df) + 1)
)


# ---------------------------------------------------------
# Select final output
# ---------------------------------------------------------

preferred_columns = [

    "final_rank",

    "paperId",

    "title",

    "year",

    "venue",

    "authors",

    "topic",

    "recommendation_score",

    "similarity",

    "intelligence_score",

    "citation_score",

    "topic_emergence_score",

    "citationCount",

    "author_count",

    "coauthor_count",

    "related_paper_count",

    "graph_degree",

    "explanation",
]


final_columns = [
    column
    for column in preferred_columns
    if column in final_df.columns
]


final_df = final_df[
    final_columns
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

final_df.to_parquet(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\n" + "=" * 70)

print(
    "FINAL RESEARCH RECOMMENDATIONS"
)

print("=" * 70)

print(
    f"Recommendations generated: "
    f"{len(final_df)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("\nTop recommendations:\n")


for _, row in final_df.head(10).iterrows():

    print(
        f"{row['final_rank']}. "
        f"{row['title']}"
    )

    print(
        f"   Score: "
        f"{row['recommendation_score']:.4f}"
    )

    if "similarity" in row:

        print(
            f"   Semantic relevance: "
            f"{row['similarity']:.4f}"
        )

    if "topic" in row:

        print(
            f"   Topic: {row['topic']}"
        )

    if "year" in row:

        print(
            f"   Year: {row['year']}"
        )

    if "citationCount" in row:

        print(
            f"   Citations: "
            f"{row['citationCount']}"
        )

    print(
        f"   Explanation: "
        f"{row['explanation']}"
    )

    print("-" * 70)