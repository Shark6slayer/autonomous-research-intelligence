from pathlib import Path
import subprocess
import sys
import time
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


STAGES = [
    (
        "Data Collection",
        PROJECT_ROOT / "src" / "data" / "collect_papers.py",
    ),
    (
        "Data Profiling",
        PROJECT_ROOT / "src" / "data" / "profile_dataset.py",
    ),
    (
        "Data Preprocessing",
        PROJECT_ROOT / "src" / "preprocessing" / "clean_papers.py",
    ),
    (
        "TF-IDF Feature Extraction",
        PROJECT_ROOT / "src" / "features" / "tfidf_features.py",
    ),
    (
        "Semantic Embeddings",
        PROJECT_ROOT / "src" / "features" / "semantic_embeddings.py",
    ),
    (
        "Semantic Clustering",
        PROJECT_ROOT / "src" / "evaluation" / "semantic_cluster.py",
    ),
    (
        "Cluster Analysis",
        PROJECT_ROOT / "src" / "evaluation" / "analyze_clusters.py",
    ),
    (
        "Cluster Quality Evaluation",
        PROJECT_ROOT / "src" / "evaluation" / "cluster_quality.py",
    ),
    (
        "Topic Discovery",
        PROJECT_ROOT / "src" / "classification" / "distinctive_topics.py",
    ),
    (
        "Trend Detection",
        PROJECT_ROOT / "src" / "classification" / "trend_detection.py",
    ),
    (
        "Research Intelligence",
        PROJECT_ROOT / "src" / "pipeline" / "research_intelligence.py",
    ),
    (
        "Knowledge Graph Construction",
        PROJECT_ROOT / "src" / "knowledge_graph" / "build_graph.py",
    ),
    (
        "Knowledge Graph Analysis",
        PROJECT_ROOT / "src" / "knowledge_graph" / "analyze_graph.py",
    ),
    (
        "Semantic Retrieval",
        PROJECT_ROOT / "src" / "recommendation" / "semantic_retriever.py",
    ),
    (
        "Intelligence Ranking",
        PROJECT_ROOT / "src" / "recommendation" / "intelligence_ranker.py",
    ),
    (
        "MMR Diversity Reranking",
        PROJECT_ROOT / "src" / "recommendation" / "diversity_reranker.py",
    ),
    (
        "Graph Context",
        PROJECT_ROOT / "src" / "recommendation" / "graph_context.py",
    ),
    (
        "MMR Evaluation",
        PROJECT_ROOT / "src" / "evaluation" / "mmr_evaluation.py",
    ),
    (
        "Recommendation Comparison",
        PROJECT_ROOT / "src" / "evaluation" / "compare_recommendations.py",
    ),
]


def run_stage(stage_number, total_stages, stage_name, script_path):

    print()
    print("=" * 80)
    print(f"STAGE {stage_number}/{total_stages}: {stage_name}")
    print("=" * 80)
    print(f"Script: {script_path}")

    if not script_path.exists():
        raise FileNotFoundError(
            f"Stage script not found: {script_path}"
        )

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print()
        print("=" * 80)
        print(f"FAILED: {stage_name}")
        print(f"Exit code: {result.returncode}")
        print(f"Time: {elapsed:.2f} seconds")
        print("=" * 80)
        return False

    print()
    print(f"COMPLETED: {stage_name}")
    print(f"Time: {elapsed:.2f} seconds")

    return True


def main():

    parser = argparse.ArgumentParser(
        description="Run the Autonomous Research Intelligence pipeline."
    )

    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip the Semantic Scholar data collection stage and use the existing corpus.",
    )

    args = parser.parse_args()

    stages_to_run = STAGES.copy()

    if args.skip_collection:
        stages_to_run = [
            stage
            for stage in stages_to_run
            if stage[0] != "Data Collection"
        ]

    total_stages = len(stages_to_run)

    print()
    print("=" * 80)
    print("AUTONOMOUS RESEARCH INTELLIGENCE")
    print("MASTER PIPELINE")
    print("=" * 80)

    if args.skip_collection:
        print("Mode: SKIP DATA COLLECTION")
    else:
        print("Mode: FULL PIPELINE")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Total stages: {total_stages}")

    pipeline_start = time.time()

    for stage_number, (stage_name, script_path) in enumerate(
        stages_to_run,
        start=1,
    ):

        success = run_stage(
            stage_number,
            total_stages,
            stage_name,
            script_path,
        )

        if not success:
            total_time = time.time() - pipeline_start

            print()
            print("=" * 80)
            print("PIPELINE FAILED")
            print("=" * 80)
            print(f"Failed stage: {stage_name}")
            print(f"Total time: {total_time:.2f} seconds")
            print("=" * 80)

            sys.exit(1)

    total_time = time.time() - pipeline_start

    print()
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Stages completed: {total_stages}/{total_stages}")
    print(f"Total time: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()