from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# ARI LIVE RESEARCH RECOMMENDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# FILES
# ============================================================

PAPERS_FILE = (
    DATA_DIR
    / "clustered_papers.parquet"
)

EMBEDDINGS_FILE = (
    DATA_DIR
    / "semantic_embeddings.npy"
)

INTELLIGENCE_FILE = (
    DATA_DIR
    / "research_intelligence.parquet"
)

GRAPH_FILE = (
    DATA_DIR
    / "research_knowledge_graph.graphml"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

CANDIDATE_K = 100

TOP_K = 10

MMR_LAMBDA = 0.75

SEMANTIC_WEIGHT = 0.40

INTELLIGENCE_WEIGHT = 0.30

CITATION_WEIGHT = 0.15

EMERGENCE_WEIGHT = 0.15


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


def get_model():

    global _model

    if _model is None:

        print(
            f"Loading semantic model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


# ============================================================
# LOAD ARI DATA
# ============================================================

_papers = None
_embeddings = None
_intelligence = None


def load_data():

    global _papers
    global _embeddings
    global _intelligence

    if _papers is None:

        print("Loading ARI paper corpus...")

        _papers = pd.read_parquet(
            PAPERS_FILE
        )

    if _embeddings is None:

        print("Loading semantic embeddings...")

        _embeddings = np.load(
            EMBEDDINGS_FILE
        )

    if _intelligence is None:

        print(
            "Loading research intelligence..."
        )

        _intelligence = pd.read_parquet(
            INTELLIGENCE_FILE
        )

    return (
        _papers,
        _embeddings,
        _intelligence,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(series):

    minimum = series.min()

    maximum = series.max()

    if maximum > minimum:

        return (
            series - minimum
        ) / (
            maximum - minimum
        )

    return pd.Series(
        1.0,
        index=series.index,
    )


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(row):

    reasons = []

    similarity = float(
        row.get(
            "similarity",
            0,
        )
    )

    intelligence = float(
        row.get(
            "intelligence_score",
            0,
        )
    )

    citation = float(
        row.get(
            "citation_score",
            0,
        )
    )

    emergence = float(
        row.get(
            "topic_emergence_score",
            0,
        )
    )

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


    if intelligence >= 0.60:

        reasons.append(
            "high research-intelligence score"
        )

    elif intelligence >= 0.45:

        reasons.append(
            "good research-intelligence score"
        )


    if citation >= 0.50:

        reasons.append(
            "strong citation impact"
        )


    if emergence >= 0.40:

        reasons.append(
            "emerging research area"
        )


    return "; ".join(
        reasons
    )


# ============================================================
# LIVE RECOMMENDATION
# ============================================================

def recommend(
    query: str,
    candidate_k: int = CANDIDATE_K,
    top_k: int = TOP_K,
):

    query = query.strip()

    if not query:

        raise ValueError(
            "Research query cannot be empty."
        )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        papers,
        embeddings,
        intelligence,
    ) = load_data()


    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    model = get_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )


    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    similarities = (
        embeddings
        @ query_embedding
    )


    candidate_count = min(
        candidate_k,
        len(papers),
    )


    top_indices = np.argsort(
        similarities
    )[::-1][
        :candidate_count
    ]


    candidates = papers.iloc[
        top_indices
    ].copy()


    candidates["similarity"] = (
        similarities[
            top_indices
        ]
    )


    candidates = candidates.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Intelligence data
    # --------------------------------------------------------

    intelligence_columns = [

        "paperId",

        "intelligence_score",

        "citation_score",

        "topic_emergence_score",
    ]


    available_columns = [

        column

        for column in intelligence_columns

        if column in intelligence.columns
    ]


    candidates = candidates.merge(

        intelligence[
            available_columns
        ],

        on="paperId",

        how="left",
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in [

        "intelligence_score",

        "citation_score",

        "topic_emergence_score",

    ]:

        if column in candidates.columns:

            candidates[column] = (
                candidates[column]
                .fillna(0)
            )


    # --------------------------------------------------------
    # Normalize signals
    # --------------------------------------------------------

    candidates[
        "semantic_score"
    ] = min_max_normalize(
        candidates["similarity"]
    )


    candidates[
        "normalized_intelligence"
    ] = min_max_normalize(
        candidates["intelligence_score"]
    )


    candidates[
        "normalized_citation"
    ] = min_max_normalize(
        candidates["citation_score"]
    )


    candidates[
        "normalized_emergence"
    ] = min_max_normalize(
        candidates["topic_emergence_score"]
    )


    # --------------------------------------------------------
    # Recommendation score
    # --------------------------------------------------------

    candidates[
        "recommendation_score"
    ] = (

        SEMANTIC_WEIGHT
        * candidates[
            "semantic_score"
        ]

        +

        INTELLIGENCE_WEIGHT
        * candidates[
            "normalized_intelligence"
        ]

        +

        CITATION_WEIGHT
        * candidates[
            "normalized_citation"
        ]

        +

        EMERGENCE_WEIGHT
        * candidates[
            "normalized_emergence"
        ]
    )


    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    candidates = candidates.sort_values(

        "recommendation_score",

        ascending=False,
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # MMR
    # --------------------------------------------------------

    candidate_paper_ids = (
        candidates["paperId"]
        .tolist()
    )


    paper_to_index = {

        paper_id: index

        for index, paper_id

        in enumerate(
            papers["paperId"]
        )
    }


    valid_positions = []

    embedding_indices = []


    for position, paper_id in enumerate(
        candidate_paper_ids
    ):

        if paper_id in paper_to_index:

            valid_positions.append(
                position
            )

            embedding_indices.append(
                paper_to_index[
                    paper_id
                ]
            )


    mmr_candidates = candidates.iloc[
        valid_positions
    ].reset_index(
        drop=True
    )


    candidate_embeddings = embeddings[
        embedding_indices
    ].astype(
        np.float32
    )


    # Normalize embeddings

    norms = np.linalg.norm(
        candidate_embeddings,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1


    candidate_embeddings = (
        candidate_embeddings
        / norms
    )


    similarity_matrix = (
        candidate_embeddings
        @ candidate_embeddings.T
    )


    # --------------------------------------------------------
    # MMR selection
    # --------------------------------------------------------

    selected = []

    mmr_scores = []

    remaining = list(
        range(
            len(mmr_candidates)
        )
    )


    while (

        remaining

        and len(selected) < top_k

    ):

        best_candidate = None

        best_score = -np.inf


        for candidate in remaining:

            relevance = float(
                mmr_candidates.loc[
                    candidate,
                    "recommendation_score",
                ]
            )


            if not selected:

                redundancy = 0.0

            else:

                redundancy = max(

                    similarity_matrix[
                        candidate,
                        selected,
                    ]
                )


            mmr_score = (

                MMR_LAMBDA
                * relevance

                -

                (1 - MMR_LAMBDA)
                * redundancy
            )


            if mmr_score > best_score:

                best_score = mmr_score

                best_candidate = candidate


        selected.append(
            best_candidate
        )

        mmr_scores.append(
            best_score
        )

        remaining.remove(
            best_candidate
        )


    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = mmr_candidates.loc[
        selected
    ].copy()


    results = results.reset_index(
        drop=True
    )


    results[
        "mmr_score"
    ] = mmr_scores


    results[
        "recommendation_rank"
    ] = range(
        1,
        len(results) + 1
    )


    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    results[
        "explanation"
    ] = results.apply(
        generate_explanation,
        axis=1,
    )


    # --------------------------------------------------------
    # Final JSON-friendly output
    # --------------------------------------------------------

    output_columns = [

        "recommendation_rank",

        "paperId",

        "title",

        "year",

        "citationCount",

        "semantic_quality",

        "cluster",

        "similarity",

        "url",

        "intelligence_score",

        "citation_score",

        "topic_emergence_score",

        "recommendation_score",

        "mmr_score",

        "explanation",
    ]


    output_columns = [

        column

        for column in output_columns

        if column in results.columns
    ]


    results = results[
        output_columns
    ]


    return results


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter research query: "
    ).strip()


    recommendations = recommend(
        query
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ARI LIVE RESEARCH RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )


    for _, row in recommendations.iterrows():

        print(
            f"\n"
            f"{row['recommendation_rank']}. "
            f"{row['title']}"
        )

        print(
            f"   Similarity: "
            f"{row['similarity']:.4f}"
        )

        print(
            f"   Intelligence: "
            f"{row['intelligence_score']:.4f}"
        )

        print(
            f"   Recommendation: "
            f"{row['recommendation_score']:.4f}"
        )

        print(
            f"   MMR: "
            f"{row['mmr_score']:.4f}"
        )

        print(
            f"   Explanation: "
            f"{row['explanation']}"
        )

    print(
        "\n"
        + "=" * 70
    )