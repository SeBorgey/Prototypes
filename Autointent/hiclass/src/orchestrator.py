import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from autointent import Embedder
from autointent.configs import EmbedderConfig
from hiclass import (LocalClassifierPerLevel, LocalClassifierPerNode,
                     LocalClassifierPerParentNode)
from model_training import (AutoIntentMulticlassTrainer,
                            AutoIntentMultilabelTrainer, HiClassTrainer)
from metrics import MetricsCalculator
from preprocessor import DatasetPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


def _get_leaf(sample: Dict) -> str:
    return sample["labels"][-1]


class ExperimentOrchestrator:
    def __init__(self, dataset_dirs: List[str], embedder_model: str):
        self.dataset_dirs = dataset_dirs
        self.embedder_model = embedder_model
        self.results = []
        self.embedder_config = EmbedderConfig(model_name=self.embedder_model)
        self.embedder = Embedder(self.embedder_config)

    @staticmethod
    def _load_raw_data(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
        train_path = Path(dataset_path) / "train.json"
        test_path = Path(dataset_path) / "test.json"
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        return train_data, test_data

    def _get_models_to_run(self) -> Dict[str, Any]:
        base_classifier = LogisticRegression(max_iter=500, random_state=42)
        hiclass_variants = {
            "LCPN_siblings": {"binary_policy": "siblings"},
            "LCPPN": {},
            "LCPL": {},
        }
        models = {}
        for name, params in hiclass_variants.items():
            model_class = LocalClassifierPerNode
            if "LCPPN" in name:
                model_class = LocalClassifierPerParentNode
            elif "LCPL" in name:
                model_class = LocalClassifierPerLevel

            models[f"hiclass_{name}"] = HiClassTrainer(
                model_class=model_class,
                embedder=self.embedder,
                local_classifier=base_classifier,
                **params,
            )

        models["autointent_multiclass"] = AutoIntentMulticlassTrainer(
            embedder_config=self.embedder_config
        )
        models["autointent_multilabel"] = AutoIntentMultilabelTrainer(
            embedder_config=self.embedder_config
        )
        return models

    def _run_single_dataset(self, dataset_dir: str):
        print(f"--- Processing dataset: {dataset_dir} ---")
        train_raw, test_raw = self._load_raw_data(dataset_dir)

        preprocessor = DatasetPreprocessor(train_raw, test_raw, min_train_freq=2)
        preprocessor.run_preprocessing()
        train_f, test_f = preprocessor.get_processed_data()

        if not train_f or not test_f:
            print("[WARN] Empty splits after filtering, skipping dataset.\n")
            return

        leaf_to_id = {
            leaf: i for i, leaf in enumerate(preprocessor.common_leaves)
        }
        mlb = MultiLabelBinarizer(classes=preprocessor.final_labels)
        mlb.fit([set(s["labels"]) for s in train_f])
        
        num_leaves = len(preprocessor.common_leaves)
        k = math.ceil(0.2 * num_leaves) if num_leaves > 0 else 1
        print(f"Leaf count: {num_leaves}. Using k={k} for large-scale metrics.")


        models = self._get_models_to_run()

        for name, trainer in models.items():
            print(f"Running model: {name}...")

            y_true_raw, context = None, {}
            prepare_kwargs = {
                "final_labels": preprocessor.final_labels,
                "mlb": mlb,
                "leaf_to_id": leaf_to_id,
            }
            prepared_data = trainer.prepare(train_f, test_f, **prepare_kwargs)
            
            predictions = trainer.run(prepared_data)

            if "hiclass" in name:
                y_true_raw = [s["labels"] for s in test_f]
                context = {"mlb": mlb}
            elif "multiclass" in name:
                y_true_raw = [leaf_to_id[_get_leaf(s)] for s in test_f]
                context = {"mlb": mlb, "leaf_to_id": leaf_to_id}
            elif "multilabel" in name:
                y_true_raw = mlb.transform([set(s["labels"]) for s in test_f]).tolist()
            
            if "scores" in predictions:
                context["y_pred_scores"] = predictions["scores"]

            calculator = MetricsCalculator(y_true_raw, predictions, **context)
            if num_leaves > 50:
                metrics = calculator.calculate_large_scale_metrics(k)
            else:
                metrics = calculator.calculate_small_scale_metrics()
            
            print(f"Results for {name}: {metrics}")
            self.results.append({"dataset": dataset_dir, "model": name, **metrics})
        print("-" * 50)


    def run_experiments(self):
        for dataset_dir in self.dataset_dirs:
            self._run_single_dataset(dataset_dir)

    def report_results(self):
        if not self.results:
            print("No results to report.")
            return
        df_results = pd.DataFrame(self.results)
        print("\n--- Final Comparison Results ---")
        print(df_results.to_string())
        df_results.to_csv("final_results.csv", index=False)
        print("\nResults saved to final_results.csv")