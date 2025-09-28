import pytest
from autointent import Embedder
from autointent.configs import EmbedderConfig
from hiclass import LocalClassifierPerNode
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from Autointent.hiclass.src.model_training import (
    AutoIntentMulticlassTrainer,
    AutoIntentMultilabelTrainer,
    HiClassTrainer,
)
from Autointent.hiclass.src.preprocessor import DatasetPreprocessor

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ... [ SAMPLE_TRAIN_DATA_RAW и SAMPLE_TEST_DATA_RAW без изменений ] ...
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
    
    # HiClass prepare работает на всем отфильтрованном трейне, без validation
    prepared_data = trainer.prepare(train_f, test_f)
    predictions = trainer.run(prepared_data)

    assert isinstance(predictions, list) or isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_f)

def test_autointent_multiclass_trainer(preprocessed_data):
    train_f, test_f = preprocessed_data.get_processed_data()
    embedder_config = EmbedderConfig(model_name=EMBEDDER_MODEL)
    trainer = AutoIntentMulticlassTrainer(embedder_config, val_size=0.5, random_state=42)
    
    leaf_to_id = {leaf: i for i, leaf in enumerate(preprocessed_data.common_leaves)}
    
    # autointent prepare сам разделит train_f на train/val
    prepared_data = trainer.prepare(train_f, test_f, leaf_to_id=leaf_to_id, final_labels=preprocessed_data.final_labels)
    predictions = trainer.run(prepared_data)
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(test_f)

def test_autointent_multilabel_trainer(preprocessed_data):
    train_f, test_f = preprocessed_data.get_processed_data()
    embedder_config = EmbedderConfig(model_name=EMBEDDER_MODEL)
    trainer = AutoIntentMultilabelTrainer(embedder_config, val_size=0.5, random_state=42)

    mlb = MultiLabelBinarizer(classes=preprocessed_data.final_labels)
    mlb.fit([set(s["labels"]) for s in train_f])

    prepared_data = trainer.prepare(train_f, test_f, mlb=mlb, final_labels=preprocessed_data.final_labels)
    predictions = trainer.run(prepared_data)
    
    assert isinstance(predictions, list)
    assert len(predictions) == len(test_f)