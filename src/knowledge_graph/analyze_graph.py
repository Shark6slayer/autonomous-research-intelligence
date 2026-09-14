from pathlib import Path
import json

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

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_graph_analysis.parquet"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\n========== ARI KNOWLEDGE GRAPH ANALYSIS ==========\n")

print("Loading knowledge graph...")

graph = nx.read_graphml(GRAPH_PATH)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")


# ============================================================
# NODE TYPE COUNTS
# ============================================================

node_types = {}

for _, data in graph.nodes(data=True):
    node_type = data.get("node_type", "unknown")
    node_types[node_type] = node_types.get(node_type, 0) + 1

print("\nNode types:")

for node_type, count in sorted(node_types.items()):
    print(f"  {node_type}: {count}")


# ============================================================
# DEGREE CENTRALITY
# ============================================================

print("\nCalculating degree centrality...")

degree_centrality = nx.degree_centrality(graph)


# ============================================================
# AUTHOR ANALYSIS
# ============================================================

print("\nAnalyzing authors...")

author_rows = []

for node, data in graph.nodes(data=True):

    if data.get("node_type") != "author":
        continue

    author_name = data.get("name", node)

    degree = graph.degree(node)

    centrality = degree_centrality.get(node, 0.0)

    paper_count = sum(
        1
        for neighbor in graph.neighbors(node)
        if graph.nodes[neighbor].get("node_type") == "paper"
    )

    coauthor_count = sum(
        1
        for neighbor in graph.neighbors(node)
        if graph.nodes[neighbor].get("node_type") == "author"
    )

    author_rows.append(
        {
            "author_id": node,
            "author_name": author_name,
            "paper_count": paper_count,
            "coauthor_count": coauthor_count,
            "degree": degree,
            "degree_centrality": centrality,
        }
    )


authors_df = pd.DataFrame(author_rows)

authors_df = authors_df.sort_values(
    ["paper_count", "coauthor_count", "degree_centrality"],
    ascending=False,
)

print("\nTop 15 authors by publication count:")

print(
    authors_df[
        [
            "author_name",
            "paper_count",
            "coauthor_count",
            "degree_centrality",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# VENUE ANALYSIS
# ============================================================

print("\nAnalyzing venues...")

venue_rows = []

for node, data in graph.nodes(data=True):

    if data.get("node_type") != "venue":
        continue

    venue_name = data.get("name", node)

    degree = graph.degree(node)

    paper_count = sum(
        1
        for neighbor in graph.neighbors(node)
        if graph.nodes[neighbor].get("node_type") == "paper"
    )

    venue_rows.append(
        {
            "venue_id": node,
            "venue_name": venue_name,
            "paper_count": paper_count,
            "degree": degree,
            "degree_centrality": degree_centrality.get(node, 0.0),
        }
    )


venues_df = pd.DataFrame(venue_rows)

venues_df = venues_df.sort_values(
    ["paper_count", "degree_centrality"],
    ascending=False,
)

print("\nTop 15 venues by paper count:")

print(
    venues_df[
        [
            "venue_name",
            "paper_count",
            "degree_centrality",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# TOPIC ANALYSIS
# ============================================================

print("\nAnalyzing topics...")

topic_rows = []

for node, data in graph.nodes(data=True):

    if data.get("node_type") != "topic":
        continue

    topic_name = data.get("name", node)

    connected_papers = [
        neighbor
        for neighbor in graph.neighbors(node)
        if graph.nodes[neighbor].get("node_type") == "paper"
    ]

    paper_count = len(connected_papers)

    connected_authors = set()

    for paper in connected_papers:

        for neighbor in graph.neighbors(paper):

            if graph.nodes[neighbor].get("node_type") == "author":
                connected_authors.add(neighbor)

    topic_rows.append(
        {
            "topic_id": node,
            "topic_name": topic_name,
            "paper_count": paper_count,
            "author_count": len(connected_authors),
            "degree": graph.degree(node),
            "degree_centrality": degree_centrality.get(node, 0.0),
        }
    )


topics_df = pd.DataFrame(topic_rows)

topics_df = topics_df.sort_values(
    ["paper_count", "author_count"],
    ascending=False,
)

print("\nTopic analysis:")

print(
    topics_df[
        [
            "topic_name",
            "paper_count",
            "author_count",
            "degree_centrality",
        ]
    ].to_string(index=False)
)


# ============================================================
# COAUTHOR NETWORK
# ============================================================

print("\nAnalyzing co-author network...")

author_nodes = [
    node
    for node, data in graph.nodes(data=True)
    if data.get("node_type") == "author"
]

coauthor_graph = graph.subgraph(author_nodes).copy()

print(f"Author network nodes: {coauthor_graph.number_of_nodes()}")
print(f"Author network edges: {coauthor_graph.number_of_edges()}")


# ============================================================
# TOP COAUTHOR CONNECTIONS
# ============================================================

coauthor_edges = []

for author_a, author_b in coauthor_graph.edges():

    name_a = graph.nodes[author_a].get("name", author_a)
    name_b = graph.nodes[author_b].get("name", author_b)

    coauthor_edges.append(
        {
            "author_a": name_a,
            "author_b": name_b,
        }
    )


coauthor_df = pd.DataFrame(coauthor_edges)


print("\nTop co-author connections:")

if len(coauthor_df) > 0:
    print(coauthor_df.head(15).to_string(index=False))
else:
    print("No co-author connections found.")


# ============================================================
# SAVE COMBINED ANALYSIS
# ============================================================

print("\nSaving analysis...")

analysis_rows = []

for _, row in authors_df.iterrows():

    analysis_rows.append(
        {
            "entity_type": "author",
            "entity_id": row["author_id"],
            "entity_name": row["author_name"],
            "paper_count": row["paper_count"],
            "author_count": row["coauthor_count"],
            "degree": row["degree"],
            "degree_centrality": row["degree_centrality"],
        }
    )


for _, row in venues_df.iterrows():

    analysis_rows.append(
        {
            "entity_type": "venue",
            "entity_id": row["venue_id"],
            "entity_name": row["venue_name"],
            "paper_count": row["paper_count"],
            "author_count": None,
            "degree": row["degree"],
            "degree_centrality": row["degree_centrality"],
        }
    )


for _, row in topics_df.iterrows():

    analysis_rows.append(
        {
            "entity_type": "topic",
            "entity_id": row["topic_id"],
            "entity_name": row["topic_name"],
            "paper_count": row["paper_count"],
            "author_count": row["author_count"],
            "degree": row["degree"],
            "degree_centrality": row["degree_centrality"],
        }
    )


analysis_df = pd.DataFrame(analysis_rows)

analysis_df.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print(f"\nSaved analysis to:")
print(OUTPUT_PATH)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "nodes": graph.number_of_nodes(),
    "edges": graph.number_of_edges(),
    "node_types": node_types,
    "author_network_nodes": coauthor_graph.number_of_nodes(),
    "author_network_edges": coauthor_graph.number_of_edges(),
    "top_author": (
        authors_df.iloc[0]["author_name"]
        if len(authors_df) > 0
        else None
    ),
    "top_topic": (
        topics_df.iloc[0]["topic_name"]
        if len(topics_df) > 0
        else None
    ),
}

summary_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_graph_analysis_summary.json"
)

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)


print("\nSaved summary to:")
print(summary_path)

print("\n===============================================\n")