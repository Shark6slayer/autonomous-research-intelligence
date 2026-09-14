from pathlib import Path
from itertools import combinations
import json

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_intelligence.parquet"
)

OUTPUT_GRAPH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_knowledge_graph.graphml"
)

OUTPUT_STATS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "knowledge_graph_stats.json"
)


def parse_authors(value):
    """
    Convert the authors column into a Python list of dictionaries.
    Handles JSON strings, Python lists, numpy arrays, and null values.
    """

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, str):

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except (json.JSONDecodeError, TypeError):
            return []

        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return value

    return []


def main():

    print()
    print("========== ARI KNOWLEDGE GRAPH ==========")
    print()

    df = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Papers loaded: {len(df)}"
    )

    required_columns = {
        "paperId",
        "title",
        "authors",
        "topic",
        "venue",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    graph = nx.Graph()

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    print()
    print(
        "Building knowledge graph..."
    )

    for _, row in df.iterrows():

        paper_id = str(
            row["paperId"]
        )

        paper_node = (
            f"paper:{paper_id}"
        )

        # --------------------------------------------------
        # Paper node
        # --------------------------------------------------

        graph.add_node(
            paper_node,
            node_type="paper",
            label=str(row["title"]),
            year=int(row["year"]),
            citations=int(
                row["citationCount"]
            ),
        )

        # --------------------------------------------------
        # Topic node
        # --------------------------------------------------

        topic = str(
            row["topic"]
        ).strip()

        if topic:

            topic_node = (
                f"topic:{topic}"
            )

            graph.add_node(
                topic_node,
                node_type="topic",
                label=topic,
            )

            graph.add_edge(
                paper_node,
                topic_node,
                relationship="BELONGS_TO",
            )

        # --------------------------------------------------
        # Venue node
        # --------------------------------------------------

        venue = row["venue"]

        if pd.notna(venue):

            venue = str(
                venue
            ).strip()

            if venue:

                venue_node = (
                    f"venue:{venue}"
                )

                graph.add_node(
                    venue_node,
                    node_type="venue",
                    label=venue,
                )

                graph.add_edge(
                    paper_node,
                    venue_node,
                    relationship="PUBLISHED_IN",
                )

        # --------------------------------------------------
        # Author nodes
        # --------------------------------------------------

        authors = parse_authors(
            row["authors"]
        )

        author_nodes = []

        for author in authors:

            if not isinstance(
                author,
                dict,
            ):
                continue

            author_id = author.get(
                "authorId"
            )

            author_name = author.get(
                "name"
            )

            if not author_id:
                continue

            author_node = (
                f"author:{author_id}"
            )

            graph.add_node(
                author_node,
                node_type="author",
                label=str(
                    author_name
                    or author_id
                ),
            )

            graph.add_edge(
                author_node,
                paper_node,
                relationship="WROTE",
            )

            author_nodes.append(
                author_node
            )

        # --------------------------------------------------
        # Co-author relationships
        # --------------------------------------------------

        unique_authors = sorted(
            set(author_nodes)
        )

        for author_a, author_b in combinations(
            unique_authors,
            2,
        ):

            graph.add_edge(
                author_a,
                author_b,
                relationship="COAUTHORED_WITH",
            )

    # --------------------------------------------------
    # Count node types
    # --------------------------------------------------

    node_types = {}

    for _, attributes in graph.nodes(
        data=True
    ):

        node_type = attributes.get(
            "node_type",
            "unknown",
        )

        node_types[node_type] = (
            node_types.get(
                node_type,
                0,
            )
            + 1
        )

    # --------------------------------------------------
    # Count relationships
    # --------------------------------------------------

    relationship_types = {}

    for _, _, attributes in graph.edges(
        data=True
    ):

        relationship = attributes.get(
            "relationship",
            "unknown",
        )

        relationship_types[
            relationship
        ] = (
            relationship_types.get(
                relationship,
                0,
            )
            + 1
        )

    # --------------------------------------------------
    # Graph statistics
    # --------------------------------------------------

    stats = {
        "papers": int(
            node_types.get(
                "paper",
                0,
            )
        ),
        "authors": int(
            node_types.get(
                "author",
                0,
            )
        ),
        "topics": int(
            node_types.get(
                "topic",
                0,
            )
        ),
        "venues": int(
            node_types.get(
                "venue",
                0,
            )
        ),
        "nodes": int(
            graph.number_of_nodes()
        ),
        "edges": int(
            graph.number_of_edges()
        ),
        "relationships": relationship_types,
    }

    # --------------------------------------------------
    # Save GraphML
    # --------------------------------------------------

    OUTPUT_GRAPH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        OUTPUT_GRAPH,
    )

    # --------------------------------------------------
    # Save statistics
    # --------------------------------------------------

    with open(
        OUTPUT_STATS,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stats,
            file,
            indent=2,
        )

    # --------------------------------------------------
    # Display statistics
    # --------------------------------------------------

    print()
    print(
        "========== GRAPH STATISTICS =========="
    )

    print(
        f"Paper nodes:  {stats['papers']}"
    )

    print(
        f"Author nodes: {stats['authors']}"
    )

    print(
        f"Topic nodes:  {stats['topics']}"
    )

    print(
        f"Venue nodes:  {stats['venues']}"
    )

    print(
        f"Total nodes:  {stats['nodes']}"
    )

    print(
        f"Total edges:  {stats['edges']}"
    )

    print()
    print(
        "Relationships:"
    )

    for relationship, count in (
        relationship_types.items()
    ):

        print(
            f"  {relationship}: {count}"
        )

    print()
    print(
        "Saved graph to:"
    )

    print(
        OUTPUT_GRAPH
    )

    print()
    print(
        "Saved graph statistics to:"
    )

    print(
        OUTPUT_STATS
    )

    print()
    print(
        "======================================"
    )
    print()


if __name__ == "__main__":
    main()