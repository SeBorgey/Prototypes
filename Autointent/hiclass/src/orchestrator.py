import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from autointent import Embedder
from autointent.configs import EmbedderConfig
from hiclass import (
    LocalClassifierPerLevel,
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
)
from metrics import MetricsCalculator
from model_training import (
    AutoIntentMulticlassTrainer,
    AutoIntentMultilabelTrainer,
    HiClassTrainer,
)
from preprocessor import DatasetPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


def _leaf(sample: Dict) -> str:
    return sample["labels"][-1] if sample.get("labels") else ""


class ExperimentOrchestrator:
    def __init__(
        self,
        dataset_dirs: List[str],
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_train_freq: int = 2,
        val_size: float = 0.5,
        random_state: int = 42,
        large_space_threshold: int = 50,
    ):
        self.dataset_dirs = dataset_dirs
        self.embedder_model = embedder_model
        self.min_train_freq = min_train_freq
        self.val_size = val_size
        self.random_state = random_state
        self.large_space_threshold = large_space_threshold

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
    def _build_mlb(
        final_labels: List[str], train_processed: List[Dict]
    ) -> MultiLabelBinarizer:
        mlb = MultiLabelBinarizer(classes=final_labels)
        mlb.fit([set(s["labels"]) for s in train_processed])
        return mlb

    @staticmethod
    def _build_leaf_to_id(common_leaves: List[str]) -> Dict[str, int]:
        return {leaf: i for i, leaf in enumerate(common_leaves)}

    def _choose_metrics(
        self, n_classes: int, calc: MetricsCalculator
    ) -> Dict[str, Any]:
        if n_classes > self.large_space_threshold:
            k = max(1, math.ceil(0.2 * n_classes))
            return {
                "n_classes": n_classes,
                "k": k,
                **calc.calculate_all_metrics_large_space(k),
            }
        else:
            return {"n_classes": n_classes, **calc.calculate_all_metrics_small_space()}

    def run(self) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        for dataset_dir in self.dataset_dirs:
            print(f"--- Processing dataset: {dataset_dir} ---")

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

            mlb = self._build_mlb(preproc.final_labels, train_p)
            leaf_to_id = self._build_leaf_to_id(preproc.common_leaves)

            n_leaves = len(leaf_to_id)
            print(
                f"After preprocessing: train={len(train_p)}, test={len(test_p)}, "
                f"leaf_classes={n_leaves}, all_final_labels={len(preproc.final_labels)}"
            )

            y_true_hiclass = [s["labels"] for s in test_p]

            y_true_multiclass = [leaf_to_id[_leaf(s)] for s in test_p]

            y_true_multilabel = mlb.transform([set(s["labels"]) for s in test_p])

            base_classifier = LogisticRegression(max_iter=500)
            hiclass_models = {
                "LCPN exclusive": (
                    LocalClassifierPerNode,
                    {"local_classifier": base_classifier, "binary_policy": "exclusive"},
                ),
                "LCPN less exclusive": (
                    LocalClassifierPerNode,
                    {
                        "local_classifier": base_classifier,
                        "binary_policy": "less_exclusive",
                    },
                ),
                "LCPN less inclusive": (
                    LocalClassifierPerNode,
                    {
                        "local_classifier": base_classifier,
                        "binary_policy": "less_inclusive",
                    },
                ),
                "LCPN inclusive": (
                    LocalClassifierPerNode,
                    {"local_classifier": base_classifier, "binary_policy": "inclusive"},
                ),
                "LCPN siblings": (
                    LocalClassifierPerNode,
                    {"local_classifier": base_classifier, "binary_policy": "siblings"},
                ),
                "LCPN exclusive siblings": (
                    LocalClassifierPerNode,
                    {
                        "local_classifier": base_classifier,
                        "binary_policy": "exclusive_siblings",
                    },
                ),
                "LCPPN": (
                    LocalClassifierPerParentNode,
                    {"local_classifier": base_classifier},
                ),
                "LCPL": (
                    LocalClassifierPerLevel,
                    {"local_classifier": base_classifier},
                ),
            }

            for name, (model_class, kwargs) in hiclass_models.items():
                print(f"Running hiclass: {name}...")
                trainer = HiClassTrainer(
                    model_class=model_class, embedder=self.embedder, **kwargs
                )

                prepared = trainer.prepare(train_p, test_p, mlb=mlb)
                output = trainer.run(prepared)

                calc = MetricsCalculator(
                    y_true_raw=y_true_hiclass,
                    y_pred_raw=output["predictions"],
                    mlb=mlb,
                    y_pred_scores=output.get("scores"),
                )
                metrics = self._choose_metrics(n_leaves, calc)

                results.append(
                    {"dataset": dataset_dir, "model": f"hiclass_{name}", **metrics}
                )
                print(f"Results for {name}: {metrics}")

            print("Running autointent: Multiclass LogReg...")
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

            calc_mc = MetricsCalculator(
                y_true_raw=y_true_multiclass,
                y_pred_raw=output_mc["predictions"],
                mlb=mlb,
                leaf_to_id=leaf_to_id,
                y_pred_scores=output_mc.get("scores"),
            )
            metrics_mc = self._choose_metrics(n_leaves, calc_mc)
            results.append(
                {
                    "dataset": dataset_dir,
                    "model": "autointent_multiclass_logreg",
                    **metrics_mc,
                }
            )
            print(f"Results for Autointent Multiclass LogReg: {metrics_mc}")

            print("Running autointent: Multilabel LogReg...")
            ai_ml = AutoIntentMultilabelTrainer(
                embedder_config=self.embedder_config_autointent,
                val_size=self.val_size,
                random_state=self.random_state,
            )
            prepared_ml = ai_ml.prepare(
                train_p,
                test_p,
                final_labels=preproc.final_labels,
                mlb=mlb,
            )
            output_ml = ai_ml.run(prepared_ml)

            calc_ml = MetricsCalculator(
                y_true_raw=y_true_multilabel.tolist(),
                y_pred_raw=output_ml["predictions"],
                y_pred_scores=output_ml.get("scores"),
            )
            metrics_ml = self._choose_metrics(n_leaves, calc_ml)
            results.append(
                {
                    "dataset": dataset_dir,
                    "model": "autointent_multilabel_logreg",
                    **metrics_ml,
                }
            )
            print(f"Results for Autointent Multilabel LogReg: {metrics_ml}")

        df_results = pd.DataFrame(results)
        print("\n--- Final Comparison Results ---")
        print(df_results.to_string(index=False))
        return df_results


if __name__ == "__main__":
    DATASET_DIRS = [
        # "unified_datasets/custom_intents",
        "unified_datasets/dbpedia_classes",
        "unified_datasets/wiki_academic_subjects",
    ]
    orchestrator = ExperimentOrchestrator(
        dataset_dirs=DATASET_DIRS,
        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
        min_train_freq=2,
        val_size=0.5,
        random_state=42,
        large_space_threshold=50,
    )
    orchestrator.run()
