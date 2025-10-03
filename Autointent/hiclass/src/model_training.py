import abc
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from autointent import Dataset, Embedder, Pipeline
from autointent.configs import DataConfig, EmbedderConfig
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



def _leaf(sample: Dict) -> str:
    return sample["labels"][-1]


class BaseModelTrainer(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass


class HiClassTrainer(BaseModelTrainer):
    def __init__(self, model_class: Any, embedder: Embedder, **model_kwargs: Any):
        self.model_class, self.embedder, self.model_kwargs = (
            model_class,
            embedder,
            model_kwargs,
        )

    def prepare(
        self, train_data: List[Dict], test_data: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        mlb = kwargs.get("mlb")
        if not mlb:
            raise ValueError(
                "HiClassTrainer requires 'mlb' in prepare() kwargs to build the scores matrix."
            )

        x_train_text = [i["text"] for i in train_data]
        y_train_labels = [i["labels"] for i in train_data]
        x_test_text = [i["text"] for i in test_data]

        max_depth = max((len(i) for i in y_train_labels), default=0)
        max_depth_test = max((len(i["labels"]) for i in test_data), default=0)
        max_depth = max(max_depth, max_depth_test)

        def pad(labels, depth):
            return np.array([r + [None] * (depth - len(r)) for r in labels], dtype=object)

        return {
            "x_train_embed": self.embedder.embed(x_train_text),
            "y_train": pad(y_train_labels, max_depth),
            "x_test_embed": self.embedder.embed(x_test_text),
            "mlb": mlb,
        }

    def _calculate_leaf_probabilities(
        self, clf, X_test: np.ndarray, mlb: MultiLabelBinarizer
    ) -> np.ndarray: 
        n_samples = X_test.shape[0]
        leaf_names = list(mlb.classes_)
        n_leaves = len(leaf_names)
        leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

        def one_hot_fallback() -> np.ndarray:
            out = np.zeros((n_samples, n_leaves), dtype=float)
            try:
                paths = clf.predict(X_test)
            except Exception:
                return out
            for i, path in enumerate(paths):
                if isinstance(path, (list, tuple, np.ndarray)):
                    for node in reversed(list(path)):
                        if node in leaf_to_idx:
                            out[i, leaf_to_idx[node]] = 1.0
                            break
            return out

        probas_per_level = clf.predict_proba(X_test)
        if not isinstance(probas_per_level, (list, tuple)):
            probas_per_level = [probas_per_level]

        global_maps = getattr(clf, "global_class_to_index_mapping_", None)
        if not (isinstance(global_maps, (list, tuple)) and all(isinstance(m, dict) for m in global_maps)):
            return one_hot_fallback()

        hierarchy = getattr(clf, "hierarchy_", None)
        root = getattr(clf, "root_", None)
        if hierarchy is None or root is None:
            return one_hot_fallback()

        def coerce_level_matrix(P) -> np.ndarray | None:
            if isinstance(P, np.ndarray):
                if P.ndim == 2:
                    if P.shape[0] == n_samples:
                        return P
                    if P.shape[1] == n_samples:
                        return P.T
                    return None
                return None
            if isinstance(P, (list, tuple)):
                if len(P) == 0:
                    return None
                arrs = [np.asarray(a) for a in P]
                if all(a.ndim == 1 and a.shape[0] == n_samples for a in arrs):
                    return np.column_stack(arrs)
                if all(a.ndim == 2 and a.shape[0] == n_samples for a in arrs):
                    return np.concatenate(arrs, axis=1)
                return None
            return None

        level_mats: list[np.ndarray] = []
        for lvl, P in enumerate(probas_per_level):
            M = coerce_level_matrix(P)
            if M is None:
                return one_hot_fallback()

            lvl_map = global_maps[lvl] if lvl < len(global_maps) else {}
            valid_indices = [idx for node, idx in lvl_map.items() if node is not None]
            needed_cols = (max(valid_indices) + 1) if valid_indices else 0
            if needed_cols and M.shape[1] < needed_cols:
                return one_hot_fallback()
            level_mats.append(M)

        leaf_paths: dict[str, list] = {}
        for leaf in leaf_names:
            try:
                path = nx.shortest_path(hierarchy, source=root, target=leaf)
            except Exception:
                leaf_paths[leaf] = None
                continue
            path = [n for n in path if n is not None and n != root]
            leaf_paths[leaf] = path if path else None

        scores = np.zeros((n_samples, n_leaves), dtype=float)

        for i in range(n_samples):
            for leaf_idx, leaf in enumerate(leaf_names):
                path = leaf_paths.get(leaf)
                if not path:
                    continue

                prob = 1.0
                ok = True
                for level, node in enumerate(path):
                    if level >= len(level_mats):
                        ok = False
                        break
                    P = level_mats[level]
                    if i >= P.shape[0]:
                        ok = False
                        break

                    lvl_map = global_maps[level] if level < len(global_maps) else {}
                    col_idx = lvl_map.get(node, None)

                    if col_idx is None or col_idx >= P.shape[1]:
                        ok = False
                        break

                    prob *= float(P[i, col_idx])
                    if prob == 0.0:
                        ok = False
                        break

                if ok:
                    scores[i, leaf_idx] = prob

        zero_rows = np.where(np.all(scores == 0, axis=1))[0]
        if len(zero_rows) > 0:
            oh = one_hot_fallback()
            scores[zero_rows] = oh[zero_rows]
        return scores



    def run(self, prepared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        mlb = prepared_data.get("mlb")
        if mlb is None:
            raise ValueError("HiClassTrainer did not find 'mlb' in prepared_data.")

        model_kwargs = self.model_kwargs.copy()
        model_kwargs["return_all_probabilities"] = True
        model = self.model_class(**model_kwargs)
        model.fit(prepared_data["x_train_embed"], prepared_data["y_train"])
        predictions = model.predict(prepared_data["x_test_embed"])
        scores = self._calculate_leaf_probabilities(
            model, prepared_data["x_test_embed"], mlb
        )
        return {"predictions": predictions, "scores": scores}


class AutoIntentBaseTrainer(BaseModelTrainer):
    def __init__(
        self,
        embedder_config: EmbedderConfig,
        val_size: float = 0.5,
        random_state: int = 42,
    ):
        self.embedder_config = embedder_config
        self.val_size = val_size
        self.random_state = random_state
        self.pipeline = None

    @staticmethod
    def _ensure_label_coverage(
        y, train_idx, val_idx, max_iter=1000
    ):
        train_idx, val_idx = list(train_idx), list(val_idx)
        all_labels = set(np.where(y.sum(axis=0) > 0)[0])
        for _ in range(max_iter):
            train_labels = (
                set(np.where(y[train_idx].sum(axis=0) > 0)[0]) if train_idx else set()
            )
            val_labels = (
                set(np.where(y[val_idx].sum(axis=0) > 0)[0]) if val_idx else set()
            )
            miss_train, miss_val = all_labels - train_labels, all_labels - val_labels
            if not miss_train and not miss_val:
                break
            for lbl in miss_train:
                cands = [i for i in val_idx if y[i, lbl] == 1]
                if cands:
                    val_idx.remove(cands[0])
                    train_idx.append(cands[0])
            for lbl in miss_val:
                cands = [i for i in train_idx if y[i, lbl] == 1]
                if cands:
                    train_idx.remove(cands[0])
                    val_idx.append(cands[0])
        return np.array(train_idx), np.array(val_idx)

    def _split_data(
        self, train_data: List[Dict], all_labels: List[str]
    ) -> tuple[List[Dict], List[Dict]]:
        if not train_data:
            return [], []
        mlb = MultiLabelBinarizer(classes=all_labels).fit(
            [set(s["labels"]) for s in train_data]
        )
        y = mlb.transform([set(s["labels"]) for s in train_data])
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=self.val_size, random_state=self.random_state
        )
        train_idx, val_idx = next(msss.split(np.zeros(len(train_data)), y))
        train_idx, val_idx = self._ensure_label_coverage(y, train_idx, val_idx)
        train_arr = np.array(train_data, dtype=object)
        return train_arr[train_idx].tolist(), train_arr[val_idx].tolist()


class AutoIntentMulticlassTrainer(AutoIntentBaseTrainer):
    def prepare(
        self, train_data: List[Dict], test_data: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        train_split, val_split = self._split_data(train_data, kwargs["final_labels"])
        leaf_to_id = kwargs["leaf_to_id"]
        def formatter(s):
            return {"utterance": s["text"], "label": leaf_to_id[_leaf(s)]}
        return {
            "train": [formatter(s) for s in train_split],
            "validation": [formatter(s) for s in val_split],
            "test": [formatter(s) for s in test_data],
            "intents": [{"id": i, "name": name} for name, i in leaf_to_id.items()],
        }

    def run(self, prepared_data: Dict[str, Any], **kwargs) -> List:
        dataset = Dataset.from_dict(prepared_data)
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
        self.pipeline = Pipeline.from_search_space(search_space)
        self.pipeline.set_config(self.embedder_config)
        self.pipeline.set_config(DataConfig(separation_ratio=None, validation_size=0))
        self.pipeline.fit(dataset)
        test_utterances = [s["utterance"] for s in prepared_data["test"]]
        output = self.pipeline.predict_with_metadata(test_utterances)
        predictions = output.predictions
        scores = [utterance_output.score for utterance_output in output.utterances]
        return {"predictions": predictions, "scores": scores}


class AutoIntentMultilabelTrainer(AutoIntentBaseTrainer):
    def prepare(
        self, train_data: List[Dict], test_data: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        train_split, val_split = self._split_data(train_data, kwargs["final_labels"])
        mlb: MultiLabelBinarizer = kwargs["mlb"]
        def formatter(s):
            return {
                    "utterance": s["text"],
                    "label": mlb.transform([set(s["labels"])])[0].tolist(),
                }
        return {
            "train": [formatter(s) for s in train_split],
            "validation": [formatter(s) for s in val_split],
            "test": [formatter(s) for s in test_data],
            "intents": [{"id": i, "name": name} for i, name in enumerate(mlb.classes_)],
        }

    def run(self, prepared_data: Dict[str, Any], **kwargs) -> List:
        dataset = Dataset.from_dict(prepared_data).to_multilabel()
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
        self.pipeline = Pipeline.from_search_space(search_space)
        self.pipeline.set_config(self.embedder_config)
        self.pipeline.set_config(DataConfig(separation_ratio=None))
        self.pipeline.fit(dataset)
        test_utterances = [s["utterance"] for s in prepared_data["test"]]
        output = self.pipeline.predict_with_metadata(test_utterances)
        predictions = output.predictions
        scores = [utterance_output.score for utterance_output in output.utterances]
        return {"predictions": predictions, "scores": scores}
