import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from autointent import Dataset, Embedder, Pipeline
from autointent.configs import EmbedderConfig,DataConfig
from hiclass import (
    LocalClassifierPerLevel,
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
)

# --- Конфигурация ---
DATASET_DIRS = ["unified_datasets/custom_intents"] 
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# --- Конец конфигурации ---


def load_raw_data(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
    train_path = Path(dataset_path) / "train.json"
    test_path = Path(dataset_path) / "test.json"
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return train_data, test_data


def preprocess_for_hiclass(
    train_raw: List[Dict], test_raw: List[Dict]
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    x_train = [item["text"] for item in train_raw]
    y_train = [item["labels"][0] for item in train_raw]

    x_test = [item["text"] for item in test_raw]
    y_test = [item["labels"][0] for item in test_raw]

    max_depth = max(max(len(i) for i in y_train), max(len(i) for i in y_test))

    def pad(labels, depth):
        return np.array(
            [row + [""] * (depth - len(row)) for row in labels], dtype=object
        )

    y_train_padded = pad(y_train, max_depth)
    y_test_padded = pad(y_test, max_depth)

    return x_train, y_train_padded, x_test, y_test_padded


def preprocess_for_autointent_multiclass(
    train_raw: List[Dict], test_raw: List[Dict]
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    train_samples = [
        {"utterance": item["text"], "label": item["labels"][0][-1]} for item in train_raw
    ]
    test_samples = [
        {"utterance": item["text"], "label": item["labels"][0][-1]} for item in test_raw
    ]

    all_leaf_labels = sorted(list(set(s["label"] for s in train_samples)))
    label_to_id = {label: i for i, label in enumerate(all_leaf_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    for sample in train_samples:
        sample["label"] = label_to_id[sample["label"]]
    for sample in test_samples:
        sample["label"] = label_to_id.get(sample["label"], -1) # Use -1 for unseen test labels

    intents = [{"id": i, "name": name} for name, i in label_to_id.items()]

    return train_samples, test_samples, {"intents": intents, "id_to_label": id_to_label}


def preprocess_for_autointent_multilabel(
    train_raw: List[Dict], test_raw: List[Dict]
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    
    train_labels_sets = [set(item["labels"][0]) for item in train_raw]
    
    mlb = MultiLabelBinarizer()
    y_train_binarized = mlb.fit_transform(train_labels_sets)
    
    train_samples = [
        {"utterance": item["text"], "label": y_train_binarized[i].tolist()}
        for i, item in enumerate(train_raw)
    ]

    test_samples = []
    for item in test_raw:
        label_set = set(item["labels"][0])
        # Only binarize known classes
        known_labels = [lbl for lbl in label_set if lbl in mlb.classes_]
        binarized_label = mlb.transform([known_labels])[0].tolist() if known_labels else [0] * len(mlb.classes_)
        test_samples.append({"utterance": item["text"], "label": binarized_label})
    
    intents = [{"id": i, "name": name} for i, name in enumerate(mlb.classes_)]
    
    return train_samples, test_samples, {"intents": intents, "mlb": mlb}


def calculate_hiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    total_samples = len(y_true)
    if total_samples == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    total_true_positives = 0
    total_predicted_positives = 0
    total_actual_positives = 0
    
    for true_path, pred_path in zip(y_true, y_pred):
        true_set = set(filter(None, true_path))
        pred_set = set(filter(None, pred_path))
        
        total_true_positives += len(true_set.intersection(pred_set))
        total_predicted_positives += len(pred_set)
        total_actual_positives += len(true_set)
        
    precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0
    recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_autointent_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    total_true_positives = 0
    total_predicted_positives = 0
    total_actual_positives = 0

    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])

        total_true_positives += len(true_set.intersection(pred_set))
        total_predicted_positives += len(pred_set)
        total_actual_positives += len(true_set)
        
    precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0
    recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def run_hiclass_experiment(
    model_class, x_train_embed, y_train, x_test_embed, y_test, **kwargs
) -> Dict:
    model = model_class(**kwargs)
    model.fit(x_train_embed, y_train)
    y_pred = model.predict(x_test_embed)
    return calculate_hiclass_metrics(y_test, y_pred)


def run_autointent_multiclass_experiment(
    train_samples, test_samples, metadata, embedder_config
) -> Dict:
    dataset = Dataset.from_dict(
        {"train": train_samples, "test": test_samples, "intents": metadata["intents"]}
    )

    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_f1",
            "search_space": [{"module_name": "sklearn", "clf_name": ["LogisticRegression"]}],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_f1",
            "search_space": [{"module_name": "argmax"}],
        },
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(embedder_config)

    data_config = DataConfig(separation_ratio=None, validation_size=0.5)
    pipeline.set_config(data_config)
    pipeline.fit(dataset)

    test_utterances = [s["utterance"] for s in test_samples]
    y_pred = pipeline.predict(test_utterances)

    y_true = [s["label"] for s in test_samples]

    # Filter out samples with unseen labels during testing
    y_true_filtered, y_pred_filtered = [], []
    for true, pred in zip(y_true, y_pred):
        if true != -1:
            y_true_filtered.append(true)
            y_pred_filtered.append(pred)

    return {
        "precision": precision_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0),
        "recall": recall_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0),
        "f1": f1_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0),
    }


def run_autointent_multilabel_experiment(
    train_samples, test_samples, metadata, embedder_config
) -> Dict:
    dataset = Dataset.from_dict(
        {"train": train_samples, "test": test_samples, "intents": metadata["intents"]}
    ).to_multilabel()

    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_f1",
            "search_space": [{"module_name": "sklearn", "clf_name": ["LogisticRegression"]}],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_f1",
            "search_space": [{"module_name": "threshold", "thresh": [0.5]}],
        },
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(embedder_config)

    data_config = DataConfig(separation_ratio=None, validation_size=0.5)
    pipeline.set_config(data_config)

    pipeline.fit(dataset)

    test_utterances = [s["utterance"] for s in test_samples]
    y_pred = np.array(pipeline.predict(test_utterances))
    
    y_true = np.array([s["label"] for s in test_samples])

    return calculate_autointent_multilabel_metrics(y_true, y_pred)


def main():
    results = []
    embedder = Embedder(EmbedderConfig(model_name=EMBEDDER_MODEL))
    embedder_config_autointent = EmbedderConfig(model_name=EMBEDDER_MODEL)

    for dataset_dir in DATASET_DIRS:
        print(f"--- Processing dataset: {dataset_dir} ---")
        train_raw, test_raw = load_raw_data(dataset_dir)

        # 1. Hiclass experiments
        print("Preparing data for hiclass...")
        x_train_text, y_train_h, x_test_text, y_test_h = preprocess_for_hiclass(
            train_raw, test_raw
        )
        x_train_embed = embedder.embed(x_train_text)
        x_test_embed = embedder.embed(x_test_text)
        
        base_classifier = LogisticRegression()

        hiclass_models = {
            "LCPN": (
                LocalClassifierPerNode,
                {"local_classifier": base_classifier, "binary_policy": "siblings"},
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
            metrics = run_hiclass_experiment(
                model_class, x_train_embed, y_train_h, x_test_embed, y_test_h, **kwargs
            )
            results.append({"dataset": dataset_dir, "model": f"hiclass_{name}", **metrics})
            print(f"Results for {name}: {metrics}")

        # 2. Autointent Multiclass experiment
        print("Running autointent: Multiclass LogReg...")
        train_mc, test_mc, meta_mc = preprocess_for_autointent_multiclass(
            train_raw, test_raw
        )
        metrics_mc = run_autointent_multiclass_experiment(
            train_mc, test_mc, meta_mc, embedder_config_autointent
        )
        results.append(
            {"dataset": dataset_dir, "model": "autointent_multiclass_logreg", **metrics_mc}
        )
        print(f"Results for Autointent Multiclass LogReg: {metrics_mc}")


        # 3. Autointent Multilabel experiment
        print("Running autointent: Multilabel LogReg...")
        train_ml, test_ml, meta_ml = preprocess_for_autointent_multilabel(
            train_raw, test_raw
        )
        metrics_ml = run_autointent_multilabel_experiment(
            train_ml, test_ml, meta_ml, embedder_config_autointent
        )
        results.append(
            {"dataset": dataset_dir, "model": "autointent_multilabel_logreg", **metrics_ml}
        )
        print(f"Results for Autointent Multilabel LogReg: {metrics_ml}")

    df_results = pd.DataFrame(results)
    print("\n--- Final Comparison Results ---")
    print(df_results.to_string())


if __name__ == "__main__":
    main()