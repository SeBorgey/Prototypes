from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

class MetricsCalculator:
    def __init__(self, y_true_raw: Any, y_pred_raw: Any, **context: Any):
        self.y_true_raw = y_true_raw
        self.context = context
        self.model_type = "unknown"

        if isinstance(y_pred_raw, dict) and "predictions" in y_pred_raw:
            self.y_pred_raw = y_pred_raw["predictions"]
            scores = y_pred_raw.get("scores")
            if scores is not None:
                self.context["y_pred_scores"] = scores
        else:
            self.y_pred_raw = y_pred_raw

        self.y_true_multilabel: np.ndarray = np.array([])
        self.y_pred_multilabel: np.ndarray = np.array([])
        self.y_true_multiclass: np.ndarray = np.array([])
        self.y_pred_multiclass: np.ndarray = np.array([])


    @staticmethod
    def _is_ndarray(x) -> bool:
        return isinstance(x, np.ndarray)

    @staticmethod
    def _is_nonempty_ndarray(x) -> bool:
        return isinstance(x, np.ndarray) and x.size > 0

    @staticmethod
    def _is_nonempty_list(x) -> bool:
        return isinstance(x, list) and len(x) > 0

    def _normalize_hiclass(self) -> None:
        mlb = self.context.get("mlb")
        if mlb is None:
            raise ValueError("`mlb` must be provided in context for hiclass metrics")

        allowed = set(mlb.classes_)

        def rows_iter(y_paths):
            if isinstance(y_paths, np.ndarray):
                for row in y_paths:
                    yield list(row)
            else:
                for row in y_paths:
                    yield list(row)

        def clean_and_binarize(y_paths):
            cleaned_sets = []
            for row in rows_iter(y_paths):
                vals = []
                for v in row:
                    if v is None:
                        continue
                    # если вдруг прилетело float('nan')
                    if isinstance(v, float) and np.isnan(v):
                        continue
                    if v in allowed:
                        vals.append(v)
                cleaned_sets.append(set(vals))
            return mlb.transform(cleaned_sets)

        self.y_true_multilabel = clean_and_binarize(self.y_true_raw)
        self.y_pred_multilabel = clean_and_binarize(self.y_pred_raw)

    def _normalize_multiclass(self) -> None:
        self.y_true_multiclass = np.array(self.y_true_raw)
        self.y_pred_multiclass = np.array(self.y_pred_raw)

        mlb = self.context.get("mlb")
        if mlb is None:
            raise ValueError("`mlb` must be provided in context for multiclass metrics")

        leaf_to_id = self.context.get("leaf_to_id", {})
        id_to_leaf = {v: k for k, v in leaf_to_id.items()}

        def id_to_leaf_set(ids):
            return [{id_to_leaf.get(i)} - {None} for i in ids]

        self.y_true_multilabel = mlb.transform(id_to_leaf_set(self.y_true_raw))
        self.y_pred_multilabel = mlb.transform(id_to_leaf_set(self.y_pred_raw))

    def _normalize_multilabel(self) -> None:
        self.y_true_multilabel = np.array(self.y_true_raw)
        self.y_pred_multilabel = np.array(self.y_pred_raw)

    def _guess_is_hiclass_array(self, arr: np.ndarray) -> bool:
        if arr.ndim != 2:
            return False
        if arr.dtype == object:
            return True
        return np.issubdtype(arr.dtype, np.str_)

    def _normalize_inputs(self) -> None:
        x = self.y_pred_raw

        if self._is_nonempty_ndarray(x):
            if self._guess_is_hiclass_array(x):
                self.model_type = "hiclass"
                self._normalize_hiclass()
                return
            if x.ndim == 1 and (np.issubdtype(x.dtype, np.integer) or isinstance(x[0], (int, np.integer))):
                self.model_type = "multiclass"
                self._normalize_multiclass()
                return
            if x.ndim == 2 and (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
                self.model_type = "multilabel"
                self._normalize_multilabel()
                return

        if self._is_nonempty_list(x):
            first = x[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                inner = None
                if isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0:
                    inner = first[0]
                if isinstance(inner, str):
                    self.model_type = "hiclass"
                    self._normalize_hiclass()
                    return
                self.model_type = "multilabel"
                self._normalize_multilabel()
                return
            if isinstance(first, (int, np.integer)):
                self.model_type = "multiclass"
                self._normalize_multiclass()
                return

        self.model_type = "empty"


    def calculate_accuracy(self) -> float:
        if self.model_type in ("unknown", "empty"):
            self._normalize_inputs()

        if self.model_type == "empty":
            return 0.0
        if self.model_type == "multiclass":
            if self.y_true_multiclass.size == 0 or self.y_pred_multiclass.size == 0:
                return 0.0
            try:
                return float(accuracy_score(self.y_true_multiclass, self.y_pred_multiclass))
            except Exception:
                return 0.0

        if self.y_true_multilabel.ndim != 2 or self.y_pred_multilabel.ndim != 2:
            return 0.0
        if self.y_true_multilabel.shape != self.y_pred_multilabel.shape:
            return 0.0
        if self.y_true_multilabel.shape[0] == 0:
            return 0.0

        correct_rows = np.all(self.y_true_multilabel == self.y_pred_multilabel, axis=1)
        return float(np.mean(correct_rows)) if len(correct_rows) > 0 else 0.0

    def _calculate_score(self, score_func, average: str) -> float:
        if self.model_type in ("unknown", "empty"):
            self._normalize_inputs()
        if self.model_type == "empty":
            return 0.0
        if self.y_true_multilabel.ndim != 2 or self.y_pred_multilabel.ndim != 2:
            return 0.0
        if self.y_true_multilabel.shape[0] == 0:
            return 0.0
        try:
            return float(
                score_func(
                    self.y_true_multilabel,
                    self.y_pred_multilabel,
                    average=average,
                    zero_division=0,
                )
            )
        except ValueError:
            return 0.0

    def calculate_f1_micro(self) -> float:
        return self._calculate_score(f1_score, "micro")

    def calculate_f1_macro(self) -> float:
        return self._calculate_score(f1_score, "macro")

    def calculate_precision_micro(self) -> float:
        return self._calculate_score(precision_score, "micro")

    def calculate_precision_macro(self) -> float:
        return self._calculate_score(precision_score, "macro")

    def calculate_recall_micro(self) -> float:
        return self._calculate_score(recall_score, "micro")

    def calculate_recall_macro(self) -> float:
        return self._calculate_score(recall_score, "macro")

    def _calculate_roc_auc_score(self, average: str) -> float:
        if self.model_type in ("unknown", "empty"):
            self._normalize_inputs()

        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None:
            return 0.0

        y_pred_scores = np.asarray(y_pred_scores)
        if y_pred_scores.ndim != 2 or self.y_true_multilabel.ndim != 2:
            return 0.0
        if self.y_true_multilabel.shape[0] == 0:
            return 0.0

        try:
            return float(roc_auc_score(self.y_true_multilabel, y_pred_scores, average=average))
        except ValueError:
            return 0.0

    def calculate_roc_auc_micro(self) -> float:
        return self._calculate_roc_auc_score("micro")

    def calculate_roc_auc_macro(self) -> float:
        return self._calculate_roc_auc_score("macro")

    def calculate_accuracy_at_k(self, k: int = 3) -> float:
        if self.model_type in ("unknown", "empty"):
            self._normalize_inputs()
        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None:
            return 0.0
        y_pred_scores = np.asarray(y_pred_scores)
        if y_pred_scores.ndim != 2 or self.y_true_multilabel.ndim != 2:
            return 0.0
        if y_pred_scores.shape[0] == 0:
            return 0.0

        n_classes = y_pred_scores.shape[1]
        k = max(1, min(int(k), n_classes))

        top_k_indices = np.argsort(y_pred_scores, axis=1)[:, -k:]

        correct = 0
        for i, true_labels in enumerate(self.y_true_multilabel):
            true_indices = np.where(true_labels == 1)[0]
            if any(t_idx in top_k_indices[i] for t_idx in true_indices):
                correct += 1
        return float(correct) / float(len(self.y_true_multilabel)) if len(self.y_true_multilabel) > 0 else 0.0

    def calculate_mrr(self) -> float:
        if self.model_type in ("unknown", "empty"):
            self._normalize_inputs()
        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None:
            return 0.0
        y_pred_scores = np.asarray(y_pred_scores)
        if y_pred_scores.ndim != 2 or self.y_true_multilabel.ndim != 2:
            return 0.0
        if y_pred_scores.shape[0] == 0:
            return 0.0

        sorted_indices = np.argsort(y_pred_scores, axis=1)[:, ::-1]

        reciprocal_ranks = []
        for i, true_labels in enumerate(self.y_true_multilabel):
            true_indices = np.where(true_labels == 1)[0]
            rr = 0.0
            for rank, pred_idx in enumerate(sorted_indices[i], 1):
                if pred_idx in true_indices:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


    def calculate_all_metrics_small_space(self) -> Dict[str, float]:
        self._normalize_inputs()
        return {
            "f1_micro": self.calculate_f1_micro(),
            "f1_macro": self.calculate_f1_macro(),
            "precision_micro": self.calculate_precision_micro(),
            "precision_macro": self.calculate_precision_macro(),
            "recall_micro": self.calculate_recall_micro(),
            "recall_macro": self.calculate_recall_macro(),
            "roc_auc_micro": self.calculate_roc_auc_micro(),
            "roc_auc_macro": self.calculate_roc_auc_macro(),
        }

    def calculate_all_metrics_large_space(self, k: int) -> Dict[str, float]:
        self._normalize_inputs()
        return {
            "accuracy_at_k": self.calculate_accuracy_at_k(k=k),
            "mrr": self.calculate_mrr(),
        }
