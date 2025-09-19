import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from autointent import Dataset, Embedder, Pipeline
from autointent.configs import DataConfig, EmbedderConfig
from hiclass import (
    LocalClassifierPerLevel,
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_DIRS = [
    "unified_datasets/custom_intents",
    "unified_datasets/dbpedia_classes",
    "unified_datasets/wiki_academic_subjects",
]
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TRAIN_FREQ = 2
VAL_SIZE = 0.5
RANDOM_STATE = 42


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


def calculate_hiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] == 0:
        raise ValueError(
            "Cannot calculate metrics on empty ground truth data (y_true)."
        )
    correct_predictions = 0
    for true_path, pred_path in zip(y_true, y_pred):
        clean_true = list(filter(None, true_path))
        clean_pred = list(filter(None, pred_path))
        if clean_true == clean_pred:
            correct_predictions += 1
    return correct_predictions / len(y_true)


def run_hiclass_experiment(
    model_class, x_train_embed, y_train, x_test_embed, y_test, **kwargs
) -> Dict:
    model = model_class(**kwargs)
    model.fit(x_train_embed, y_train)
    y_pred = model.predict(x_test_embed)
    accuracy = calculate_hiclass_accuracy(y_test, y_pred)
    return {"accuracy": accuracy}


def _leaf(sample: Dict) -> str:
    return sample["labels"][0][-1]


def filter_by_common_leaves(
    train_raw: List[Dict],
    test_raw: List[Dict],
    min_train_freq: int = 2,
) -> Tuple[List[Dict], List[Dict], List[str], Dict[str, int]]:
    # Листовые метки в train
    train_leaf_counts = Counter(_leaf(s) for s in train_raw)
    train_leaves_sufficient = {
        lbl for lbl, cnt in train_leaf_counts.items() if cnt >= min_train_freq
    }
    # Листовые метки в test
    test_leaves = {_leaf(s) for s in test_raw}
    # Итоговые листья: есть в test и достаточно частотны в train
    final_common_leaves = sorted(
        list(train_leaves_sufficient.intersection(test_leaves))
    )
    if not final_common_leaves:
        print(
            "[WARN] After leaf filtering no common leaves remain. Check data/min_train_freq."
        )
    leaf_to_id = {leaf: i for i, leaf in enumerate(final_common_leaves)}

    def keep(sample: Dict) -> bool:
        return _leaf(sample) in leaf_to_id

    train_f = [s for s in train_raw if keep(s)]
    test_f = [s for s in test_raw if keep(s)]
    return train_f, test_f, final_common_leaves, leaf_to_id


def build_mlb_on_filtered(
    train_f: List[Dict], test_f: List[Dict], min_train_freq: int = 2
) -> MultiLabelBinarizer:
    # Частоты по всем уровням в train после листовой фильтрации
    counts = Counter(lbl for s in train_f for lbl in s["labels"][0])
    test_labels = {lbl for s in test_f for lbl in s["labels"][0]}
    labels_to_keep = sorted(
        [
            lbl
            for lbl, cnt in counts.items()
            if cnt >= min_train_freq and lbl in test_labels
        ]
    )

    # Гарантируем, что все листья в классы тоже попали
    leaves_in_train = {_leaf(s) for s in train_f}
    missing_leaves = sorted(list(leaves_in_train - set(labels_to_keep)))
    if missing_leaves:
        # Если такое вдруг произойдет — добавляем их принудительно
        labels_to_keep = sorted(set(labels_to_keep).union(missing_leaves))

    mlb = MultiLabelBinarizer(classes=labels_to_keep)
    mlb.fit([set(s["labels"][0]) for s in train_f])
    return mlb


def multilabel_y_for(samples: List[Dict], mlb: MultiLabelBinarizer) -> np.ndarray:
    return mlb.transform([set(s["labels"][0]) for s in samples])


def _ensure_label_coverage(
    y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, max_iter: int = 10000
):
    # Гарантируем, что каждый класс встречается и в train, и в val
    y = np.asarray(y)
    train_idx, val_idx = list(train_idx), list(val_idx)

    def present(idxs):
        if len(idxs) == 0:
            return set()
        return set(np.where(y[idxs].sum(axis=0) > 0)[0])

    all_labels = set(np.where(y.sum(axis=0) > 0)[0])

    it = 0
    while it < max_iter:
        it += 1
        miss_train = all_labels - present(train_idx)
        miss_val = all_labels - present(val_idx)
        if not miss_train and not miss_val:
            break

        # Чиним train
        for lbl in list(miss_train):
            candidates = [i for i in val_idx if y[i, lbl] == 1]
            if not candidates:
                continue
            candidates.sort(key=lambda i: int(y[i].sum()))
            chosen = candidates[0]
            val_idx.remove(chosen)
            train_idx.append(chosen)

        # Чиним val
        for lbl in list(miss_val):
            candidates = [i for i in train_idx if y[i, lbl] == 1]
            if not candidates:
                continue
            candidates.sort(key=lambda i: int(y[i].sum()))
            chosen = candidates[0]
            train_idx.remove(chosen)
            val_idx.append(chosen)

    # Финальная проверка
    miss_train = all_labels - present(train_idx)
    miss_val = all_labels - present(val_idx)
    if miss_train or miss_val:
        raise RuntimeError(
            f"Не удалось обеспечить покрытие всех меток: "
            f"train_missing={len(miss_train)}, val_missing={len(miss_val)}"
        )
    return np.array(train_idx), np.array(val_idx)


def build_external_multilabel_split_indices(
    train_f: List[Dict],
    mlb: MultiLabelBinarizer,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Строим Y по множеству меток (включая листья) на уже отфильтрованном train
    y = multilabel_y_for(train_f, mlb)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as e:
        raise ImportError("Please install iterstrat: pip install iterstrat") from e

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    idx_train, idx_val = next(msss.split(np.zeros(len(train_f)), y))

    # Для честности — тот же ensure_label_coverage, что раньше использовался только в autointent
    idx_train, idx_val = _ensure_label_coverage(y, idx_train, idx_val)
    return idx_train, idx_val, y


def run_autointent_multiclass_experiment(
    train_split, val_split, test_samples, metadata, embedder_config
) -> Dict:
    if not train_split or not test_samples:
        print("Skipping multiclass experiment due to empty train/test after filtering.")
        return {"accuracy": 0.0}

    dataset = Dataset.from_dict(
        {
            "train": train_split,
            "validation": val_split if val_split else [],
            "test": test_samples,
            "intents": metadata["intents"],
        }
    )

    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_f1",
            "search_space": [
                {
                    "module_name": "sklearn",
                    "clf_name": ["LogisticRegression"],
                    "max_iter": [500],
                }
            ],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_f1",
            "search_space": [{"module_name": "argmax"}],
        },
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(embedder_config)
    # Валидация подана явно — отключаем внутренние сплиты
    pipeline.set_config(DataConfig(separation_ratio=None, validation_size=0))
    pipeline.fit(dataset)

    test_utterances = [s["utterance"] for s in test_samples]
    y_pred = pipeline.predict(test_utterances)
    y_true = [s["label"] for s in test_samples]

    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy}


def calculate_autointent_multilabel_accuracy(y_true: np.ndarray, y_pred: List) -> float:
    if y_true.shape[0] == 0:
        raise ValueError(
            "Cannot calculate metrics on empty ground truth data (y_true)."
        )
    num_classes = y_true.shape[1]
    y_pred_processed = np.array(
        [pred if pred is not None else [0] * num_classes for pred in y_pred]
    )
    if y_true.shape != y_pred_processed.shape:
        print(
            f"Warning: Shape mismatch in multilabel accuracy calculation. y_true: {y_true.shape}, y_pred_processed: {y_pred_processed.shape}"
        )
        return 0.0
    correct_rows = np.all(y_true == y_pred_processed, axis=1)
    return np.mean(correct_rows)


def run_autointent_multilabel_experiment(
    train_split, val_split, test_samples, metadata, embedder_config
) -> Dict:
    if not train_split or not test_samples:
        print("Skipping multilabel experiment due to empty train/test after filtering.")
        return {"accuracy": 0.0}

    dataset = Dataset.from_dict(
        {
            "train": train_split,
            "validation": val_split if val_split else [],
            "test": test_samples,
            "intents": metadata["intents"],
        }
    ).to_multilabel()

    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_f1",
            "search_space": [
                {
                    "module_name": "sklearn",
                    "clf_name": ["LogisticRegression"],
                    "max_iter": [500],
                }
            ],
        },
        {
            "node_type": "decision",
            "target_metric": "decision_f1",
            "search_space": [{"module_name": "adaptive"}],
        },
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(embedder_config)
    pipeline.set_config(DataConfig(separation_ratio=None))  # валидация дана явно
    pipeline.fit(dataset)

    test_utterances = [s["utterance"] for s in test_samples]
    y_pred = pipeline.predict(test_utterances)
    y_true = np.array([s["label"] for s in test_samples])

    accuracy = calculate_autointent_multilabel_accuracy(y_true, y_pred)
    return {"accuracy": accuracy}


def build_autointent_multiclass_samples(
    train_f: List[Dict], test_f: List[Dict], leaf_to_id: Dict[str, int]
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    train_samples = [
        {"utterance": s["text"], "label": leaf_to_id[_leaf(s)]} for s in train_f
    ]
    test_samples = [
        {"utterance": s["text"], "label": leaf_to_id[_leaf(s)]} for s in test_f
    ]
    intents = [{"id": i, "name": name} for name, i in leaf_to_id.items()]
    return train_samples, test_samples, {"intents": intents}


def build_autointent_multilabel_samples(
    train_f: List[Dict], test_f: List[Dict], mlb: MultiLabelBinarizer
) -> Tuple[List[Dict], List[Dict], Dict[str, Any], np.ndarray, np.ndarray]:
    y_train = multilabel_y_for(train_f, mlb)
    y_test = multilabel_y_for(test_f, mlb)

    # Благодаря листовой фильтрации у каждого примера точно есть хотя бы листовая метка
    assert (y_train.sum(axis=1) > 0).all(), (
        "Unexpected empty multilabel rows in train after common filtering"
    )
    assert (y_test.sum(axis=1) > 0).all(), (
        "Unexpected empty multilabel rows in test after common filtering"
    )

    train_samples = [
        {"utterance": s["text"], "label": y_train[i].tolist()}
        for i, s in enumerate(train_f)
    ]
    test_samples = [
        {"utterance": s["text"], "label": y_test[i].tolist()}
        for i, s in enumerate(test_f)
    ]

    intents = [{"id": i, "name": name} for i, name in enumerate(mlb.classes_)]
    return train_samples, test_samples, {"intents": intents}, y_train, y_test


def main():
    results = []
    embedder = Embedder(EmbedderConfig(model_name=EMBEDDER_MODEL))
    embedder_config_autointent = EmbedderConfig(model_name=EMBEDDER_MODEL)

    for dataset_dir in DATASET_DIRS:
        print(f"--- Processing dataset: {dataset_dir} ---")
        train_raw, test_raw = load_raw_data(dataset_dir)

        # 0) Общая фильтрация по листьям для всех сценариев
        train_f, test_f, final_leaves, leaf_to_id = filter_by_common_leaves(
            train_raw, test_raw, min_train_freq=MIN_TRAIN_FREQ
        )
        print(
            f"After common leaf filtering: train={len(train_f)}, test={len(test_f)}, leaves={len(final_leaves)}"
        )
        if len(train_f) == 0 or len(test_f) == 0:
            print("[WARN] Empty splits after filtering — skipping dataset.")
            continue

        # 0.1) Мультилейбл бинaризатор на отфильтрованных данных (с гарантированными листьями)
        mlb = build_mlb_on_filtered(train_f, test_f, min_train_freq=MIN_TRAIN_FREQ)

        # 0.2) Единый внешний сплит по мультилейблу (те же индексы для всех моделей)
        idx_train, idx_val, y_train_ml = build_external_multilabel_split_indices(
            train_f, mlb, val_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        print(
            f"External split: train={len(idx_train)}, val={len(idx_val)}, test={len(test_f)}"
        )

        # 1) Hiclass на тех же train/test, что и у autointent (train берём idx_train)
        print("Preparing data for hiclass...")
        x_train_text_all, y_train_h_all, x_test_text, y_test_h = preprocess_for_hiclass(
            train_f, test_f
        )
        x_train_embed_all = embedder.embed(x_train_text_all)
        x_test_embed = embedder.embed(x_test_text)

        x_train_embed = x_train_embed_all[idx_train]
        y_train_h = y_train_h_all[idx_train]

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
            metrics = run_hiclass_experiment(
                model_class, x_train_embed, y_train_h, x_test_embed, y_test_h, **kwargs
            )
            results.append(
                {"dataset": dataset_dir, "model": f"hiclass_{name}", **metrics}
            )
            print(f"Results for {name}: {metrics}")

        # 2) Autointent Multiclass — на тех же индексах train/val
        print("Running autointent: Multiclass LogReg...")
        train_mc, test_mc, meta_mc = build_autointent_multiclass_samples(
            train_f, test_f, leaf_to_id
        )
        train_mc_split = [train_mc[i] for i in idx_train]
        val_mc_split = [train_mc[i] for i in idx_val]

        metrics_mc = run_autointent_multiclass_experiment(
            train_mc_split, val_mc_split, test_mc, meta_mc, embedder_config_autointent
        )
        results.append(
            {
                "dataset": dataset_dir,
                "model": "autointent_multiclass_logreg",
                **metrics_mc,
            }
        )
        print(f"Results for Autointent Multiclass LogReg: {metrics_mc}")

        # 3) Autointent Multilabel — тот же самый внешний сплит (idx_train/idx_val)
        print("Running autointent: Multilabel LogReg...")
        train_ml, test_ml, meta_ml, y_train_ml_all, y_test_ml = (
            build_autointent_multilabel_samples(train_f, test_f, mlb)
        )
        train_ml_split = [train_ml[i] for i in idx_train]
        val_ml_split = [train_ml[i] for i in idx_val]

        metrics_ml = run_autointent_multilabel_experiment(
            train_ml_split, val_ml_split, test_ml, meta_ml, embedder_config_autointent
        )
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
    print(df_results.to_string())


if __name__ == "__main__":
    main()
