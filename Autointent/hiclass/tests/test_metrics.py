import numpy as np
import pytest
from sklearn.preprocessing import MultiLabelBinarizer

from metrics import MetricsCalculator

@pytest.fixture
def shared_context():
    all_labels = ['science', 'biology', 'arts', 'literature']
    mlb = MultiLabelBinarizer(classes=all_labels).fit([all_labels])
    leaf_to_id = {'biology': 0, 'literature': 1}
    return {"mlb": mlb, "leaf_to_id": leaf_to_id}

def test_metrics_for_hiclass(shared_context):
    y_true_raw = np.array([
        ['science', 'biology', ''],
        ['arts', 'literature', '']
    ], dtype=object)
    
    y_pred = {
        "predictions": np.array([
            ['science', 'biology', ''],
            ['arts', 'biology', '']
        ], dtype=object),
        "scores": np.array([
            [0.8, 0.7, 0.1, 0.1],
            [0.1, 0.6, 0.7, 0.2]
        ])
    }

    calculator = MetricsCalculator(y_true_raw, y_pred, **shared_context)
    results = calculator.calculate_all_metrics()

    assert calculator.model_type == "hiclass"
    assert results['accuracy'] == 0.5
    assert results['precision_micro'] == pytest.approx(0.75)
    assert results['roc_auc_macro'] > 0.6
    assert results['mrr'] == 1.0

def test_metrics_for_multiclass(shared_context):
    y_true_raw = [0, 1]

    y_pred = {
        "predictions": [0, 0],
        "scores": np.array([
            [0.1, 0.9, 0.2, 0.3],
            [0.3, 0.6, 0.4, 0.5],
        ])
    }
    
    calculator = MetricsCalculator(y_true_raw, y_pred, **shared_context)
    results = calculator.calculate_all_metrics()

    assert calculator.model_type == "multiclass"
    assert results['accuracy'] == 0.5
    assert results['recall_macro'] == 0.25
    assert results['roc_auc_micro'] == pytest.approx(0.91666, 0.001) 
    assert results['mrr'] == pytest.approx(0.75)

def test_metrics_for_multilabel():
    y_true_raw = [[1, 1, 0, 0], [0, 0, 1, 1]]
    
    y_pred = {
        "predictions": [[1, 1, 0, 0], [1, 0, 1, 0]],
        "scores": np.array([
            [0.9, 0.8, 0.1, 0.2],
            [0.7, 0.3, 0.6, 0.1]
        ])
    }

    calculator = MetricsCalculator(y_true_raw, y_pred)
    results = calculator.calculate_all_metrics()

    assert calculator.model_type == "multilabel"
    assert results['accuracy'] == 0.5
    assert results['f1_micro'] == pytest.approx(0.75)
    assert results['roc_auc_macro'] > 0.65
    assert results['mrr'] == pytest.approx(0.75)

def test_metrics_without_scores():
    y_true_raw = [[1, 0], [0, 1]]
    y_pred = {
        "predictions": [[1, 0], [1, 0]],
    }
    
    calculator = MetricsCalculator(y_true_raw, y_pred)
    results = calculator.calculate_all_metrics()

    assert results['accuracy'] == 0.5
    assert results['f1_micro'] == pytest.approx(0.5)
    
    assert results['roc_auc_micro'] == 0.0
    assert results['roc_auc_macro'] == 0.0
    assert results['mrr'] == 0.0
    assert results['accuracy_at_3'] == 0.0

def test_empty_predictions():
    y_pred = {"predictions": [], "scores": []}
    
    calculator = MetricsCalculator([], y_pred)
    results = calculator.calculate_all_metrics()
    
    assert calculator.model_type == "empty"
    assert all(value == 0.0 for value in results.values())