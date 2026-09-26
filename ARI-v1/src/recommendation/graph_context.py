from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "research_knowledge_graph.graphml"
)

RECOMMENDATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "diverse_recommendations.parquet"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "recommendation_graph_context.parquet"
)

TOP_K = 10


# ============================================================
# LOAD
# ============================================================

print("\n========== ARI GRAPH CONTEXT ==========\n")

print("Loading knowledge graph...")

graph = nx.read_graphml(GRAPH_PATH)

print(
    f"Graph loaded: "
    f"{graph.number_of_nodes()} nodes, "
    f"{graph.number_of_edges()} edges"
)

print("\nLoading recommendation results...")

recommendations = (
    pd.read_parquet(RECOMMENDATION_PATH)
    .head(TOP_K)
    .copy()
)

recommendations["recommendation_rank"] = (
    range(1, len(recommendations) + 1)
)

print(
    f"Recommendations loaded: "
    f"{len(recommendations)}"
)


# ============================================================
# IDENTIFY PAPER NODES
# ============================================================

paper_nodes = {
    node
    for node, data in graph.nodes(data=True)
    if data.get("node_type") == "paper"
}

print(
    f"\nPaper nodes available: "
    f"{len(paper_nodes)}"
)


# ============================================================
# EXTRACT CONTEXT
# ============================================================

context_rows = []

print("\nExtracting graph context...\n")


for _, recommendation in recommendations.iterrows():

    paper_id = str(
        recommendation["paperId"]
    )

    graph_paper_id = f"paper:{paper_id}"

    if graph_paper_id not in graph:

        print(
            f"Warning: paper not found: {paper_id}"
        )

        continue

    # --------------------------------------------------------
    # DIRECT PAPER CONNECTIONS
    # --------------------------------------------------------

    neighbors = list(
        graph.neighbors(
            graph_paper_id
        )
    )

    authors = []
    topics = []
    venues = []

    author_nodes = []

    for neighbor in neighbors:

        node_data = graph.nodes[
            neighbor
        ]

        node_type = node_data.get(
            "node_type"
        )

        node_name = node_data.get(
            "name"
        )

        if node_type == "author":

            author_nodes.append(
                neighbor
            )

            authors.append(
                node_name or neighbor
            )

        elif node_type == "topic":

            topics.append(
                node_name or neighbor
            )

        elif node_type == "venue":

            venues.append(
                node_name or neighbor
            )

    # --------------------------------------------------------
    # RELATED PAPERS
    #
    # Paper → Author → Co-author → Paper
    # --------------------------------------------------------

    related_papers = set()

    for author in author_nodes:

        coauthors = [
            neighbor
            for neighbor in graph.neighbors(
                author
            )
            if graph.nodes[neighbor].get(
                "node_type"
            ) == "author"
        ]

        for coauthor in coauthors:

            for connected_node in graph.neighbors(
                coauthor
            ):

                if (
                    connected_node
                    != graph_paper_id
                    and connected_node
                    in paper_nodes
                ):

                    related_papers.add(
                        connected_node
                    )

    # --------------------------------------------------------
    # DIRECT CO-AUTHOR COUNT
    # --------------------------------------------------------

    coauthor_nodes = set()

    for author in author_nodes:

        for neighbor in graph.neighbors(
            author
        ):

            if (
                graph.nodes[neighbor].get(
                    "node_type"
                ) == "author"
            ):

                coauthor_nodes.add(
                    neighbor
                )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    context_rows.append(
        {
            "paperId": paper_id,
            "recommendation_rank": int(
                recommendation[
                    "recommendation_rank"
                ]
            ),
            "recommendation_score": float(
                recommendation[
                    "recommendation_score"
                ]
            ),
            "title": recommendation[
                "title"
            ],
            "author_count": len(
                authors
            ),
            "authors": authors,
            "topics": topics,
            "venues": venues,
            "related_paper_count": len(
                related_papers
            ),
            "related_papers": list(
                related_papers
            ),
            "coauthor_count": len(
                coauthor_nodes
            ),
            "graph_degree": int(
                graph.degree(
                    graph_paper_id
                )
            ),
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

context_df = pd.DataFrame(
    context_rows
)

context_df = context_df.sort_values(
    "recommendation_rank"
).reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

print(
    "========== GRAPH CONTEXT SUMMARY ==========\n"
)

print(
    f"Successfully processed: "
    f"{len(context_df)}/{len(recommendations)} papers"
)

print()

for _, row in context_df.iterrows():

    print(
        f"Rank {int(row['recommendation_rank'])}: "
        f"{row['title']}"
    )

    print(
        f"  Authors: {row['author_count']}"
    )

    print(
        f"  Co-authors in graph: "
        f"{row['coauthor_count']}"
    )

    print(
        f"  Topics: {row['topics']}"
    )

    print(
        f"  Venues: {row['venues']}"
    )

    print(
        f"  Related papers: "
        f"{row['related_paper_count']}"
    )

    print(
        f"  Graph degree: "
        f"{row['graph_degree']}"
    )

    print()


# ============================================================
# SAVE
# ============================================================

context_df.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print(
    "Saved graph context to:"
)

print(
    OUTPUT_PATH
)

print(
    "\n===========================================\n"
)