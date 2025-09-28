import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call, mock_open

from model_training import (
    AutoIntentMulticlassTrainer,
    AutoIntentMultilabelTrainer,
    HiClassTrainer,
)
from orchestrator import ExperimentOrchestrator


@pytest.fixture
def orchestrator():
    with patch("autointent.Embedder"):
        orch = ExperimentOrchestrator(
            dataset_dirs=["dummy/path/one", "dummy/path/two"],
            embedder_model="dummy-model",
        )
    return orch


def test_load_raw_data():
    train_content = json.dumps([{"text": "train", "labels": ["a"]}])
    test_content = json.dumps([{"text": "test", "labels": ["b"]}])
    
    mock_opener = mock_open()
    mock_opener.side_effect = [
        mock_open(read_data=train_content).return_value,
        mock_open(read_data=test_content).return_value,
    ]

    with patch("builtins.open", mock_opener):
        train_data, test_data = ExperimentOrchestrator._load_raw_data("dummy/path")
        
    assert train_data == [{"text": "train", "labels": ["a"]}]
    assert test_data == [{"text": "test", "labels": ["b"]}]
    
    opened_files = [args[0].name for args, kwargs in mock_opener.call_args_list]
    assert "train.json" in opened_files
    assert "test.json" in opened_files


def test_get_models_to_run(orchestrator):
    models = orchestrator._get_models_to_run()
    
    expected_keys = [
        "hiclass_LCPN_siblings",
        "hiclass_LCPPN",
        "hiclass_LCPL",
        "autointent_multiclass",
        "autointent_multilabel"
    ]
    for key in expected_keys:
        assert key in models
    
    assert isinstance(models["hiclass_LCPPN"], HiClassTrainer)
    assert isinstance(models["autointent_multiclass"], AutoIntentMulticlassTrainer)
    assert isinstance(models["autointent_multilabel"], AutoIntentMultilabelTrainer)


def test_run_experiments(orchestrator, monkeypatch):
    mock_run_single = MagicMock()
    monkeypatch.setattr(orchestrator, "_run_single_dataset", mock_run_single)
    
    orchestrator.run_experiments()
    
    assert mock_run_single.call_count == 2
    mock_run_single.assert_has_calls([
        call("dummy/path/one"),
        call("dummy/path/two"),
    ])


def test_report_results(orchestrator, monkeypatch):
    mock_df_constructor = MagicMock()
    monkeypatch.setattr(pd, "DataFrame", mock_df_constructor)
    
    orchestrator.results = [{"dataset": "d1", "model": "m1", "accuracy": 0.9}]
    
    orchestrator.report_results()
    
    mock_df_constructor.assert_called_once_with(orchestrator.results)
    
    instance = mock_df_constructor.return_value
    instance.to_string.assert_called_once()
    instance.to_csv.assert_called_once_with("final_results.csv", index=False)


@patch("orchestrator.MetricsCalculator")
@patch("orchestrator.ExperimentOrchestrator._get_models_to_run")
@patch("sklearn.preprocessing.MultiLabelBinarizer")
@patch("orchestrator.DatasetPreprocessor")
@patch("orchestrator.ExperimentOrchestrator._load_raw_data")
def test_run_single_dataset_workflow(
    mock_load_data,
    mock_preprocessor_cls,
    mock_mlb_cls,
    mock_get_models,
    mock_metrics_cls,
    orchestrator
):
    dummy_train = [{"text": "train", "labels": ["A", "B"]}]
    dummy_test = [{"text": "test", "labels": ["A", "C"]}]
    
    mock_load_data.return_value = (dummy_train, dummy_test)
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.get_processed_data.return_value = (dummy_train, dummy_test)
    mock_preprocessor.common_leaves = ["B", "C"] # 2 класса -> small scale
    mock_preprocessor.final_labels = ["A", "B", "C"]
    mock_preprocessor_cls.return_value = mock_preprocessor
    
    mock_trainer = MagicMock()
    mock_trainer.prepare.return_value = {"prepared": "data"}
    mock_trainer.run.return_value = {"predictions": [1, 0], "scores": [0.9, 0.1]}
    mock_get_models.return_value = {"mock_model": mock_trainer}
    
    mock_metrics = MagicMock()
    mock_metrics.calculate_small_scale_metrics.return_value = {"accuracy": 0.99}
    mock_metrics_cls.return_value = mock_metrics

    orchestrator._run_single_dataset("dummy/dir")

    mock_load_data.assert_called_once_with("dummy/dir")
    mock_preprocessor_cls.assert_called_once_with(dummy_train, dummy_test, min_train_freq=2)
    mock_preprocessor.run_preprocessing.assert_called_once()
    
    mock_get_models.assert_called_once()
    mock_trainer.prepare.assert_called_once()
    mock_trainer.run.assert_called_once_with({"prepared": "data"})
    
    mock_metrics_cls.assert_called_once()
    mock_metrics.calculate_small_scale_metrics.assert_called_once()
    
    assert len(orchestrator.results) == 1
    assert orchestrator.results[0]["dataset"] == "dummy/dir"
    assert orchestrator.results[0]["accuracy"] == 0.99


@patch("orchestrator.DatasetPreprocessor")
@patch("orchestrator.ExperimentOrchestrator._load_raw_data")
def test_run_single_dataset_skips_on_empty_data(
    mock_load_data, mock_preprocessor_cls, orchestrator, capsys
):
    mock_load_data.return_value = ([{"text":"data"}], [{"text":"data"}])
    mock_preprocessor = MagicMock()
    mock_preprocessor.get_processed_data.return_value = ([], []) # Главное условие
    mock_preprocessor_cls.return_value = mock_preprocessor
    
    orchestrator._run_single_dataset("dummy/dir")
    
    captured = capsys.readouterr()
    assert "Empty splits after filtering" in captured.out
    assert len(orchestrator.results) == 0