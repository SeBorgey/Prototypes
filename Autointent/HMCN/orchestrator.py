import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from autointent import Embedder
from autointent.configs import EmbedderConfig
from model_training import AutoIntentMulticlassTrainer, HMCNTrainer
from preprocessor import DatasetPreprocessor


def _leaf(sample: Dict) -> str:
    return sample["labels"][-1] if sample.get("labels") else ""


class ExperimentOrchestrator:
    def __init__(
        self,
        dataset_dirs: List[str],
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_train_freq: int = 2,
        val_size: float = 0.2,
        random_state: int = 42,
        hmcn_hidden_size: int = 384,
        hmcn_lr: float = 0.0005,
        hmcn_epochs: int = 100,
        hmcn_batch_size: int = 64,
        device: str = "cpu",
    ):
        self.dataset_dirs = dataset_dirs
        self.embedder_model = embedder_model
        self.min_train_freq = min_train_freq
        self.val_size = val_size
        self.random_state = random_state

        self.hmcn_hidden_size = hmcn_hidden_size
        self.hmcn_lr = hmcn_lr
        self.hmcn_epochs = hmcn_epochs
        self.hmcn_batch_size = hmcn_batch_size
        self.device = device

        self.embedder = Embedder(EmbedderConfig(model_name=self.embedder_model))
        self.embedder_config_autointent = EmbedderConfig(model_name=self.embedder_model)

    @staticmethod
    def _load_raw_data(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
        train_path = Path(dataset_path) / "train.json"
        test_path = Path(dataset_path) / "test.json"
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        return train_data, test_data

    @staticmethod
    def _build_leaf_to_id(common_leaves: List[str]) -> Dict[str, int]:
        return {leaf: i for i, leaf in enumerate(common_leaves)}

    @staticmethod
    def _calculate_accuracy(y_true: List, y_pred: List) -> float:
        if len(y_true) == 0:
            return 0.0
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    def run(self) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        for dataset_dir in self.dataset_dirs:
            print(f"\n{'=' * 80}")
            print(f"Processing dataset: {dataset_dir}")
            print(f"{'=' * 80}")

            train_raw, test_raw = self._load_raw_data(dataset_dir)

            preproc = DatasetPreprocessor(
                train_data=train_raw,
                test_data=test_raw,
                min_train_freq=self.min_train_freq,
            )
            preproc.run_preprocessing()

            train_p, test_p = preproc.get_processed_data()
            if len(train_p) == 0 or len(test_p) == 0:
                print("[WARN] Empty splits after preprocessing — skipping dataset.")
                continue

            leaf_to_id = self._build_leaf_to_id(preproc.common_leaves)
            n_leaves = len(leaf_to_id)

            print(
                f"Dataset stats: train={len(train_p)}, test={len(test_p)}, "
                f"leaf_classes={n_leaves}"
            )

            y_true_paths = [s["labels"] for s in test_p]
            y_true_leaves = [leaf_to_id[_leaf(s)] for s in test_p]

            print("\n--- Training HMCN ---")
            hmcn_trainer = HMCNTrainer(
                embedder=self.embedder,
                hidden_size=self.hmcn_hidden_size,
                lr=self.hmcn_lr,
                epochs=self.hmcn_epochs,
                batch_size=self.hmcn_batch_size,
                device=self.device,
            )

            prepared_hmcn = hmcn_trainer.prepare(train_p, test_p)
            output_hmcn = hmcn_trainer.run(prepared_hmcn)

            y_pred_hmcn = output_hmcn["predictions"]
            hmcn_accuracy = self._calculate_accuracy(y_true_paths, y_pred_hmcn)

            results.append(
                {
                    "dataset": Path(dataset_dir).name,
                    "model": "HMCN",
                    "n_classes": n_leaves,
                    "accuracy": f"{hmcn_accuracy:.4f}",
                }
            )
            print(f"HMCN Accuracy: {hmcn_accuracy:.4f}")

            print("\n--- Training AutoIntent Multiclass ---")
            ai_mc = AutoIntentMulticlassTrainer(
                embedder_config=self.embedder_config_autointent,
                val_size=self.val_size,
                random_state=self.random_state,
            )

            prepared_mc = ai_mc.prepare(
                train_p,
                test_p,
                final_labels=preproc.final_labels,
                leaf_to_id=leaf_to_id,
            )
            output_mc = ai_mc.run(prepared_mc)

            y_pred_mc = output_mc["predictions"]
            mc_accuracy = self._calculate_accuracy(y_true_leaves, y_pred_mc)

            results.append(
                {
                    "dataset": Path(dataset_dir).name,
                    "model": "AutoIntent_Multiclass",
                    "n_classes": n_leaves,
                    "accuracy": f"{mc_accuracy:.4f}",
                }
            )
            print(f"AutoIntent Multiclass Accuracy: {mc_accuracy:.4f}")

        df_results = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(df_results.to_string(index=False))
        return df_results


if __name__ == "__main__":
    DATASET_DIRS = [
        "unified_datasets/custom_intents",
        "unified_datasets/dbpedia_classes",
        "unified_datasets/wiki_academic_subjects",
    ]

    orchestrator = ExperimentOrchestrator(
        dataset_dirs=DATASET_DIRS,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        min_train_freq=2,
        val_size=0.2,
        random_state=42,
        hmcn_hidden_size=384,
        hmcn_lr=0.0005,
        hmcn_epochs=150,
        hmcn_batch_size=32,
        device="cpu",
    )

    results_df = orchestrator.run()
    results_df.to_csv("experiment_results.csv", index=False)
    print("\nResults saved to experiment_results.csv")