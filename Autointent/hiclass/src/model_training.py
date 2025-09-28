import abc
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from autointent import Dataset, Embedder, Pipeline
from autointent.configs import DataConfig, EmbedderConfig
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError as e:
    raise ImportError("Please install iterstrat: pip install iterstrat-fork") from e


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
        probas_per_level = clf.predict_proba(X_test)

        leaf_nodes = {
            node
            for node in clf.hierarchy_.nodes()
            if node is not None and node != clf.root_ and clf.hierarchy_.out_degree(node) == 0
        }

        num_samples = X_test.shape[0]
        leaf_scores = np.zeros((num_samples, len(mlb.classes_)), dtype=float)
        leaf_to_mlb_idx = {name: i for i, name in enumerate(mlb.classes_)}

        global_maps = getattr(clf, "global_class_to_index_mapping_", None)

        if not isinstance(probas_per_level, (list, tuple)):
            probas_per_level = [probas_per_level]

        n_levels = len(probas_per_level)

        for i in range(num_samples):
            for leaf in leaf_nodes:
                try:
                    path = nx.shortest_path(clf.hierarchy_, source=clf.root_, target=leaf)[1:]
                except (nx.NetworkXNoPath, KeyError):
                    continue

                path = [n for n in path if n is not None]
                if not path:
                    continue

                path_prob = 1.0
                ok = True

                for level, node in enumerate(path):
                    if level >= n_levels:
                        ok = False
                        break

                    level_probas = probas_per_level[level]
                    if isinstance(level_probas, list):
                        level_probas = np.asarray(level_probas)

                    if level_probas.ndim == 1:
                        if level_probas.shape[0] <= i:
                            ok = False
                            break
                        node_prob = float(level_probas[i])
                    else:
                        if level_probas.shape[0] <= i:
                            ok = False
                            break

                        class_index = None
                        if global_maps is not None and len(global_maps) > level:
                            class_index = global_maps[level].get(node, None)

                        if class_index is None:
                            node_prob = 0.0
                        else:
                            if 0 <= class_index < level_probas.shape[1]:
                                node_prob = float(level_probas[i, class_index])
                            else:
                                node_prob = 0.0

                    path_prob *= node_prob
                    if path_prob == 0.0:
                        break

                if not ok:
                    continue

                if leaf in leaf_to_mlb_idx:
                    mlb_idx = leaf_to_mlb_idx[leaf]
                    leaf_scores[i, mlb_idx] = path_prob

        return leaf_scores

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
