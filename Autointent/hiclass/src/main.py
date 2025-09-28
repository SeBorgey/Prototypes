from orchestrator import ExperimentOrchestrator

DATASET_DIRS = [
    "unified_datasets/custom_intents",
    # "unified_datasets/dbpedia_classes",
    # "unified_datasets/wiki_academic_subjects",
]
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    orchestrator = ExperimentOrchestrator(
        dataset_dirs=DATASET_DIRS, embedder_model=EMBEDDER_MODEL
    )
    orchestrator.run_experiments()
    orchestrator.report_results()
