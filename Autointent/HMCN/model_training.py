import abc
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from autointent import Dataset, Embedder, Pipeline
from autointent.configs import DataConfig, EmbedderConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset


def _leaf(sample: Dict) -> str:
    return sample["labels"][-1]


class BaseModelTrainer(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def run(self, prepared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pass


class HMCNTrainer(BaseModelTrainer):
    def __init__(
        self,
        embedder: Embedder,
        hidden_size: int = 384,
        lr: float = 0.0005,
        epochs: int = 100,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.embedder = embedder
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _build_hierarchy_info(
        self, train_data: List[Dict], test_data: List[Dict]
    ) -> Dict[str, Any]:
        all_data = train_data + test_data
        max_depth = max(len(s["labels"]) for s in all_data)

        level_classes = [set() for _ in range(max_depth)]
        for sample in all_data:
            for level, label in enumerate(sample["labels"]):
                level_classes[level].add(label)

        level_to_classes = [sorted(list(classes)) for classes in level_classes]

        level_to_idx_map = [
            {cls: idx for idx, cls in enumerate(classes)}
            for classes in level_to_classes
        ]

        num_classes_per_level = [len(classes) for classes in level_to_classes]

        class_has_children = [set() for _ in range(max_depth)]
        for sample in all_data:
            for level in range(len(sample["labels"]) - 1):
                label = sample["labels"][level]
                class_has_children[level].add(label)

        return {
            "max_depth": max_depth,
            "num_classes_per_level": num_classes_per_level,
            "level_to_classes": level_to_classes,
            "level_to_idx_map": level_to_idx_map,
            "class_has_children": class_has_children,
        }

    def _labels_to_indices(
        self, data: List[Dict], hierarchy_info: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        max_depth = hierarchy_info["max_depth"]
        n_samples = len(data)

        indices = np.zeros((n_samples, max_depth), dtype=np.int64)
        masks = np.zeros((n_samples, max_depth), dtype=np.float32)

        for i, sample in enumerate(data):
            for level, label in enumerate(sample["labels"]):
                idx = hierarchy_info["level_to_idx_map"][level][label]
                indices[i, level] = idx
                masks[i, level] = 1.0

        return indices, masks

    def _indices_to_labels(
        self, indices: np.ndarray, hierarchy_info: Dict[str, Any]
    ) -> List[List[str]]:
        result = []
        for sample_indices in indices:
            labels = []
            for level, idx in enumerate(sample_indices):
                if 0 <= idx < len(hierarchy_info["level_to_classes"][level]):
                    label = hierarchy_info["level_to_classes"][level][idx]
                    labels.append(label)
                    if label not in hierarchy_info["class_has_children"][level]:
                        break
                else:
                    break
            result.append(labels)
        return result

    def prepare(
        self, train_data: List[Dict], test_data: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        x_train_text = [s["text"] for s in train_data]
        x_test_text = [s["text"] for s in test_data]

        x_train = self.embedder.embed(x_train_text)
        x_test = self.embedder.embed(x_test_text)

        hierarchy_info = self._build_hierarchy_info(train_data, test_data)

        y_train_indices, train_masks = self._labels_to_indices(train_data, hierarchy_info)
        y_test_indices, test_masks = self._labels_to_indices(test_data, hierarchy_info)

        return {
            "x_train": x_train,
            "y_train": y_train_indices,
            "train_masks": train_masks,
            "x_test": x_test,
            "y_test": y_test_indices,
            "test_masks": test_masks,
            "hierarchy_info": hierarchy_info,
        }

    def run(self, prepared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        from hmcn import HMCNF

        hierarchy_info = prepared_data["hierarchy_info"]

        input_size = prepared_data["x_train"].shape[1]
        model = HMCNF(
            input_size=input_size,
            num_classes_per_level=hierarchy_info["num_classes_per_level"],
            hidden_size=self.hidden_size,
            dropout_rate=0.5,
        ).to(self.device)

        x_train = torch.FloatTensor(prepared_data["x_train"]).to(self.device)
        y_train = torch.LongTensor(prepared_data["y_train"]).to(self.device)
        train_masks = torch.FloatTensor(prepared_data["train_masks"]).to(self.device)

        train_dataset = TensorDataset(x_train, y_train, train_masks)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(reduction="none")

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y, batch_masks in train_loader:
                optimizer.zero_grad()

                local_logits = model(batch_x)

                loss = 0
                for level, logits in enumerate(local_logits):
                    level_targets = batch_y[:, level]
                    level_mask = batch_masks[:, level]

                    level_loss = criterion(logits, level_targets)
                    masked_loss = (level_loss * level_mask).sum() / (
                        level_mask.sum() + 1e-8
                    )
                    loss += masked_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        x_test = torch.FloatTensor(prepared_data["x_test"]).to(self.device)

        with torch.no_grad():
            local_logits = model(x_test)

            predictions_indices = []
            for level, logits in enumerate(local_logits):
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions_indices.append(preds)

            predictions_indices = np.array(predictions_indices).T

        predictions = self._indices_to_labels(predictions_indices, hierarchy_info)

        return {"predictions": predictions}


class AutoIntentBaseTrainer(BaseModelTrainer):
    def __init__(
        self,
        embedder_config: EmbedderConfig,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.embedder_config = embedder_config
        self.val_size = val_size
        self.random_state = random_state
        self.pipeline = None

    @staticmethod
    def _ensure_label_coverage(y, train_idx, val_idx, max_iter=1000):
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

    def run(self, prepared_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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
        return {"predictions": predictions}