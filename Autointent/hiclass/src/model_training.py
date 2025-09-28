import abc
from typing import Any, Dict, List

import numpy as np
from autointent import Dataset, Embedder, Pipeline
from autointent.configs import DataConfig, EmbedderConfig
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError as e:
    raise ImportError("Please install iterstrat: pip install iterstrat-fork") from e

# ... [ _leaf и BaseModelTrainer без изменений ] ...
def _leaf(sample: Dict) -> str: return sample["labels"][-1]
class BaseModelTrainer(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs) -> Any: pass
    @abc.abstractmethod
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> np.ndarray | List: pass

class HiClassTrainer(BaseModelTrainer):
    # Этот класс остается без изменений, т.к. он и так не использовал validation
    def __init__(self, model_class: Any, embedder: Embedder, **model_kwargs: Any):
        self.model_class, self.embedder, self.model_kwargs = model_class, embedder, model_kwargs
    def prepare(self, train_data: List[Dict], test_data: List[Dict], **kwargs) -> Dict[str, Any]:
        x_train_text, y_train_labels = [i["text"] for i in train_data], [i["labels"] for i in train_data]
        x_test_text = [i["text"] for i in test_data]
        max_depth = max(len(i) for i in y_train_labels) if y_train_labels else 0
        if test_data and [s['labels'] for s in test_data]:
            max_depth = max(max_depth, max(len(i) for i in [s['labels'] for s in test_data]))
        pad = lambda labels, depth: np.array([r + [""] * (depth - len(r)) for r in labels], dtype=object)
        return {"x_train_embed": self.embedder.embed(x_train_text), "y_train": pad(y_train_labels, max_depth), "x_test_embed": self.embedder.embed(x_test_text)}
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> np.ndarray:
        model = self.model_class(**self.model_kwargs)
        model.fit(prepared_data["x_train_embed"], prepared_data["y_train"])
        return model.predict(prepared_data["x_test_embed"])

class AutoIntentBaseTrainer(BaseModelTrainer):
    def __init__(self, embedder_config: EmbedderConfig, val_size: float = 0.5, random_state: int = 42):
        self.embedder_config = embedder_config
        self.val_size = val_size
        self.random_state = random_state
        self.pipeline = None
    @staticmethod
    def _ensure_label_coverage(y, train_idx, val_idx, max_iter=1000): # ... [без изменений] ...
        train_idx, val_idx = list(train_idx), list(val_idx)
        all_labels = set(np.where(y.sum(axis=0) > 0)[0])
        for _ in range(max_iter):
            train_labels = set(np.where(y[train_idx].sum(axis=0) > 0)[0]) if train_idx else set()
            val_labels = set(np.where(y[val_idx].sum(axis=0) > 0)[0]) if val_idx else set()
            miss_train, miss_val = all_labels - train_labels, all_labels - val_labels
            if not miss_train and not miss_val: break
            for lbl in miss_train:
                cands = [i for i in val_idx if y[i, lbl] == 1]
                if cands: val_idx.remove(cands[0]); train_idx.append(cands[0])
            for lbl in miss_val:
                cands = [i for i in train_idx if y[i, lbl] == 1]
                if cands: train_idx.remove(cands[0]); val_idx.append(cands[0])
        return np.array(train_idx), np.array(val_idx)
    def _split_data(self, train_data: List[Dict], all_labels: List[str]) -> tuple[List[Dict], List[Dict]]:
        if not train_data: return [], []
        mlb = MultiLabelBinarizer(classes=all_labels).fit([set(s["labels"]) for s in train_data])
        y = mlb.transform([set(s["labels"]) for s in train_data])
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        train_idx, val_idx = next(msss.split(np.zeros(len(train_data)), y))
        train_idx, val_idx = self._ensure_label_coverage(y, train_idx, val_idx)
        train_arr = np.array(train_data, dtype=object)
        return train_arr[train_idx].tolist(), train_arr[val_idx].tolist()

class AutoIntentMulticlassTrainer(AutoIntentBaseTrainer):
    def prepare(self, train_data: List[Dict], test_data: List[Dict], **kwargs) -> Dict[str, Any]:
        train_split, val_split = self._split_data(train_data, kwargs["final_labels"])
        leaf_to_id = kwargs["leaf_to_id"]
        formatter = lambda s: {"utterance": s["text"], "label": leaf_to_id[_leaf(s)]}
        return {"train": [formatter(s) for s in train_split], "validation": [formatter(s) for s in val_split], "test": [formatter(s) for s in test_data], "intents": [{"id": i, "name": name} for name, i in leaf_to_id.items()]}
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> List:
        dataset = Dataset.from_dict(prepared_data)
        search_space = [{"node_type": "scoring", "target_metric": "scoring_f1", "search_space": [{"module_name": "sklearn", "clf_name": ["LogisticRegression"], "max_iter": [500]}]}, {"node_type": "decision", "target_metric": "decision_f1", "search_space": [{"module_name": "argmax"}]}]
        self.pipeline = Pipeline.from_search_space(search_space)
        self.pipeline.set_config(self.embedder_config)
        self.pipeline.set_config(DataConfig(separation_ratio=None, validation_size=0))
        self.pipeline.fit(dataset)
        return self.pipeline.predict([s["utterance"] for s in prepared_data["test"]])

class AutoIntentMultilabelTrainer(AutoIntentBaseTrainer):
    def prepare(self, train_data: List[Dict], test_data: List[Dict], **kwargs) -> Dict[str, Any]:
        train_split, val_split = self._split_data(train_data, kwargs["final_labels"])
        mlb: MultiLabelBinarizer = kwargs["mlb"]
        formatter = lambda s: {"utterance": s["text"], "label": mlb.transform([set(s["labels"])])[0].tolist()}
        return {"train": [formatter(s) for s in train_split], "validation": [formatter(s) for s in val_split], "test": [formatter(s) for s in test_data], "intents": [{"id": i, "name": name} for i, name in enumerate(mlb.classes_)]}
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> List:
        dataset = Dataset.from_dict(prepared_data).to_multilabel()
        search_space = [{"node_type": "scoring", "target_metric": "scoring_f1", "search_space": [{"module_name": "sklearn", "clf_name": ["LogisticRegression"], "max_iter": [500]}]}, {"node_type": "decision", "target_metric": "decision_f1", "search_space": [{"module_name": "adaptive"}]}]
        self.pipeline = Pipeline.from_search_space(search_space)
        self.pipeline.set_config(self.embedder_config)
        self.pipeline.set_config(DataConfig(separation_ratio=None))
        self.pipeline.fit(dataset)
        return self.pipeline.predict([s["utterance"] for s in prepared_data["test"]])