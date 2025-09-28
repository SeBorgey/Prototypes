import copy
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path
import json

class DatasetPreprocessor:
    def __init__(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        min_train_freq: int = 2,
    ):
        self.train_data = copy.deepcopy(train_data)
        self.test_data = copy.deepcopy(test_data)
        self.min_train_freq = min_train_freq
        self.common_leaves: List[str] = []
        self.final_labels: List[str] = []

    @staticmethod
    def _get_leaf(sample: Dict) -> str:
        return sample["labels"][-1] if sample["labels"] else ""

    def _filter_by_common_leaves(self):
        train_leaf_counts = Counter(self._get_leaf(s) for s in self.train_data)
        train_leaves_sufficient = {
            lbl for lbl, cnt in train_leaf_counts.items() if cnt >= self.min_train_freq
        }
        test_leaves = {self._get_leaf(s) for s in self.test_data}

        self.common_leaves = sorted(
            list(train_leaves_sufficient.intersection(test_leaves))
        )

        def keep(sample: Dict) -> bool:
            return self._get_leaf(sample) in self.common_leaves

        self.train_data = [s for s in self.train_data if keep(s)]
        self.test_data = [s for s in self.test_data if keep(s)]

    def _determine_final_labels(self):
        if not self.train_data:
            return

        counts = Counter(lbl for s in self.train_data for lbl in s["labels"])
        test_labels_set = {lbl for s in self.test_data for lbl in s["labels"]}
        labels_to_keep = sorted(
            [
                lbl
                for lbl, cnt in counts.items()
                if cnt >= self.min_train_freq and lbl in test_labels_set
            ]
        )
    
        leaves_in_train = {self._get_leaf(s) for s in self.train_data}
        missing_leaves = sorted(list(leaves_in_train - set(labels_to_keep)))
        if missing_leaves:
            labels_to_keep = sorted(set(labels_to_keep).union(missing_leaves))

        self.final_labels = labels_to_keep

    def _log_state(self, stage: str):
        train_size = len(self.train_data)
        test_size = len(self.test_data)
        leaves_size = len(self.common_leaves)
        labels_size = len(self.final_labels)
        print(
            f"  - {stage:<35} -> "
            f"Train: {train_size}, Test: {test_size}, "
            f"CommonLeaves: {leaves_size}, FinalLabels: {labels_size}"
        )

    def run_preprocessing(self):
        print("Starting preprocessing...")
        self._log_state("Initial state")

        self._filter_by_common_leaves()
        self._log_state("After filtering by common leaves")

        self._determine_final_labels()
        self._log_state("After determining final labels")

        print("Preprocessing finished.\n")


    def get_processed_data(self) -> Tuple[List[Dict], List[Dict]]:
        return self.train_data, self.test_data


def main():
    DATASET_DIRS = [
        "unified_datasets/custom_intents",
        "unified_datasets/dbpedia_classes",
        "unified_datasets/wiki_academic_subjects",
    ]

    for dataset_dir in DATASET_DIRS:
        print(f"--- Processing dataset: {dataset_dir} ---")
        try:
            train_path = Path(dataset_dir) / "train.json"
            test_path = Path(dataset_dir) / "test.json"
            with open(train_path, "r", encoding="utf-8") as f:
                train_raw = json.load(f)
            with open(test_path, "r", encoding="utf-8") as f:
                test_raw = json.load(f)

            preprocessor = DatasetPreprocessor(train_raw, test_raw)
            preprocessor.run_preprocessing()

        except FileNotFoundError:
            print(f"[ERROR] Directory or files not found for: {dataset_dir}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred with {dataset_dir}: {e}")



if __name__ == "__main__":
    main()
