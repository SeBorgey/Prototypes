from typing import Any, Dict, List, Tuple

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


    def _normalize_hiclass(self) -> None:
        mlb = self.context.get("mlb")
        if mlb is None:
            raise ValueError("`mlb` must be provided in context for hiclass metrics")

        def clean_and_binarize(y_paths):
            cleaned_sets = [set(filter(None, path)) for path in y_paths]
            return mlb.transform(cleaned_sets)

        self.y_true_multilabel = clean_and_binarize(self.y_true_raw)
        self.y_pred_multilabel = clean_and_binarize(self.y_pred_raw)

    def _normalize_multiclass(self) -> None:
        self.y_true_multiclass = np.array(self.y_true_raw)
        self.y_pred_multiclass = np.array(self.y_pred_raw)
        
        mlb = self.context.get("mlb")
        if mlb is None:
            raise ValueError("`mlb` must be provided in context for multiclass metrics")

        leaf_map = {v: k for k, v in self.context.get("leaf_to_id", {}).items()}
        self.y_true_multilabel = mlb.transform([ {leaf_map.get(i)} for i in self.y_true_raw ])
        self.y_pred_multilabel = mlb.transform([ {leaf_map.get(i)} for i in self.y_pred_raw ])


    def _normalize_multilabel(self) -> None:
        self.y_true_multilabel = np.array(self.y_true_raw)
        self.y_pred_multilabel = np.array(self.y_pred_raw)

    def _normalize_inputs(self) -> None:
        if isinstance(self.y_pred_raw, np.ndarray) and self.y_pred_raw.dtype == object:
            self.model_type = "hiclass"
            self._normalize_hiclass()
        elif self.y_pred_raw and isinstance(self.y_pred_raw[0], int):
            self.model_type = "multiclass"
            self._normalize_multiclass()
        elif self.y_pred_raw and isinstance(self.y_pred_raw[0], list):
            self.model_type = "multilabel"
            self._normalize_multilabel()
        else:
            self.model_type = "empty"

    def calculate_accuracy(self) -> float:
        if self.model_type == "empty": return 0.0
        if self.model_type == "multiclass":
            return accuracy_score(self.y_true_multiclass, self.y_pred_multiclass)
        correct_rows = np.all(self.y_true_multilabel == self.y_pred_multilabel, axis=1)
        return np.mean(correct_rows) if len(correct_rows) > 0 else 0.0

    def _calculate_score(self, score_func, average: str) -> float:
        if self.model_type == "empty" or self.y_true_multilabel.shape[0] == 0:
            return 0.0
        return score_func(
            self.y_true_multilabel, self.y_pred_multilabel, average=average, zero_division=0
        )
    
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
        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None or self.y_true_multilabel.shape[0] == 0:
            return 0.0
        
        try:
            return roc_auc_score(self.y_true_multilabel, y_pred_scores, average=average)
        except ValueError:
            return 0.0

    def calculate_roc_auc_micro(self) -> float:
        return self._calculate_roc_auc_score("micro")
    
    def calculate_roc_auc_macro(self) -> float:
        return self._calculate_roc_auc_score("macro")

    def calculate_accuracy_at_k(self, k: int = 3) -> float:
        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None: return 0.0
        y_pred_scores = np.asarray(y_pred_scores)
        if y_pred_scores.shape[0] == 0: return 0.0
        
        top_k_indices = np.argsort(y_pred_scores, axis=1)[:, -k:]
        
        correct = 0
        for i, true_labels in enumerate(self.y_true_multilabel):
            true_indices = np.where(true_labels == 1)[0]
            if any(t_idx in top_k_indices[i] for t_idx in true_indices):
                correct += 1
        return correct / len(self.y_true_multilabel) if len(self.y_true_multilabel) > 0 else 0.0

    def calculate_mrr(self) -> float:
        y_pred_scores = self.context.get("y_pred_scores")
        if y_pred_scores is None: return 0.0
        y_pred_scores = np.asarray(y_pred_scores)
        if y_pred_scores.shape[0] == 0: return 0.0
            
        sorted_indices = np.argsort(y_pred_scores, axis=1)[:, ::-1]
        
        reciprocal_ranks = []
        for i, true_labels in enumerate(self.y_true_multilabel):
            true_indices = np.where(true_labels == 1)[0]
            found = False
            for rank, pred_idx in enumerate(sorted_indices[i], 1):
                if pred_idx in true_indices:
                    reciprocal_ranks.append(1 / rank)
                    found = True
                    break
            if not found:
                reciprocal_ranks.append(0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_all_metrics(self) -> Dict[str, float]:
        self._normalize_inputs()
        
        results = {
            "accuracy": self.calculate_accuracy(),
            "f1_micro": self.calculate_f1_micro(),
            "f1_macro": self.calculate_f1_macro(),
            "precision_micro": self.calculate_precision_micro(),
            "precision_macro": self.calculate_precision_macro(),
            "recall_micro": self.calculate_recall_micro(),
            "recall_macro": self.calculate_recall_macro(),
            "roc_auc_micro": self.calculate_roc_auc_micro(),
            "roc_auc_macro": self.calculate_roc_auc_macro(),
            "accuracy_at_3": self.calculate_accuracy_at_k(k=3),
            "mrr": self.calculate_mrr(),
        }
        return results