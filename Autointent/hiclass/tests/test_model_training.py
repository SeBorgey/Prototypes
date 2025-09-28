import pytest
from autointent import Embedder
from autointent.configs import EmbedderConfig
from hiclass import LocalClassifierPerNode
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from model_training import (
    AutoIntentMulticlassTrainer,
    AutoIntentMultilabelTrainer,
    HiClassTrainer,
)
from preprocessor import DatasetPreprocessor

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_TRAIN_DATA_RAW = [{"text": "What is biology?", "labels": ["science", "biology"]},{"text": "Study of cells", "labels": ["science", "biology"]},{"text": "Tell me about Shakespeare", "labels": ["arts", "literature"]},{"text": "What is a sonnet?", "labels": ["arts", "literature"]},{"text": "What is physics?", "labels": ["science", "physics"]},]
SAMPLE_TEST_DATA_RAW = [{"text": "Plant and animal cells", "labels": ["science", "biology"]},{"text": "Romeo and Juliet", "labels": ["arts", "literature"]},]


@pytest.fixture(scope="module")
def preprocessed_data():
    preprocessor = DatasetPreprocessor(SAMPLE_TRAIN_DATA_RAW, SAMPLE_TEST_DATA_RAW, min_train_freq=2)
    preprocessor.run_preprocessing()
    return preprocessor

def test_hiclass_trainer(preprocessed_data):
    train_f, test_f = preprocessed_data.get_processed_data()
    embedder = Embedder(EmbedderConfig(model_name=EMBEDDER_MODEL))
    trainer = HiClassTrainer(model_class=LocalClassifierPerNode, embedder=embedder, local_classifier=LogisticRegression())
    
    mlb = MultiLabelBinarizer(classes=preprocessed_data.final_labels).fit([set(s["labels"]) for s in train_f])
    prepared_data = trainer.prepare(train_f, test_f, mlb=mlb)
    
    results = trainer.run(prepared_data)
    
    assert isinstance(results, dict)
    assert "predictions" in results
    assert "scores" in results
    
    predictions = results["predictions"]
    scores = results["scores"]

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_f)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(test_f), len(mlb.classes_))


def test_autointent_multiclass_trainer(preprocessed_data):
    train_f, test_f = preprocessed_data.get_processed_data()
    embedder_config = EmbedderConfig(model_name=EMBEDDER_MODEL)
    trainer = AutoIntentMulticlassTrainer(embedder_config, val_size=0.5, random_state=42)
    
    leaf_to_id = {leaf: i for i, leaf in enumerate(preprocessed_data.common_leaves)}
    
    prepared_data = trainer.prepare(train_f, test_f, leaf_to_id=leaf_to_id, final_labels=preprocessed_data.final_labels)
    
    results = trainer.run(prepared_data)

    assert isinstance(results, dict)
    assert "predictions" in results
    assert "scores" in results
    
    predictions = results["predictions"]
    scores = results["scores"]
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(test_f)
    assert isinstance(scores, list) or isinstance(scores, np.ndarray)
    assert len(scores) == len(test_f)
    assert len(scores[0]) == len(leaf_to_id)


def test_autointent_multilabel_trainer(preprocessed_data):
    train_f, test_f = preprocessed_data.get_processed_data()
    embedder_config = EmbedderConfig(model_name=EMBEDDER_MODEL)
    trainer = AutoIntentMultilabelTrainer(embedder_config, val_size=0.5, random_state=42)

    mlb = MultiLabelBinarizer(classes=preprocessed_data.final_labels)
    mlb.fit([set(s["labels"]) for s in train_f])

    prepared_data = trainer.prepare(train_f, test_f, mlb=mlb, final_labels=preprocessed_data.final_labels)
    
    results = trainer.run(prepared_data)

    assert isinstance(results, dict)
    assert "predictions" in results
    assert "scores" in results
    
    predictions = results["predictions"]
    scores = results["scores"]
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(test_f)
    assert isinstance(scores, list) or isinstance(scores, np.ndarray)
    assert len(scores) == len(test_f)
    assert len(scores[0]) == len(mlb.classes_)