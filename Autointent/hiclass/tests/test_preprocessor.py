import pytest
from Autointent.hiclass.src.preprocessor import DatasetPreprocessor

SAMPLE_TRAIN_DATA = [
    {"text": "a", "labels": ["cat", "animal", "mammal"]},
    {"text": "b", "labels": ["cat", "animal", "mammal"]},
    {"text": "c", "labels": ["dog", "animal", "mammal"]},
    {"text": "d", "labels": ["car", "vehicle"]},
    {"text": "e", "labels": ["car", "vehicle"]},
    {"text": "f", "labels": ["boat", "vehicle"]},
]

SAMPLE_TEST_DATA = [
    {"text": "g", "labels": ["cat", "animal", "mammal"]},
    {"text": "h", "labels": ["dog", "animal"]},
    {"text": "i", "labels": ["car", "vehicle"]},
]


def test_initialization():
    preprocessor = DatasetPreprocessor(SAMPLE_TRAIN_DATA, SAMPLE_TEST_DATA)
    assert len(preprocessor.train_data) == 6
    assert len(preprocessor.test_data) == 3
    assert preprocessor.min_train_freq == 2
    preprocessor.train_data[0]["text"] = "modified"
    assert SAMPLE_TRAIN_DATA[0]["text"] == "a"


def test_filter_by_common_leaves():
    preprocessor = DatasetPreprocessor(
        SAMPLE_TRAIN_DATA, SAMPLE_TEST_DATA, min_train_freq=2
    )
    preprocessor._filter_by_common_leaves()

    assert preprocessor.common_leaves == ["mammal", "vehicle"]
    
    assert len(preprocessor.train_data) == 6
    train_texts = {s["text"] for s in preprocessor.train_data}
    assert train_texts == {"a", "b", "c", "d", "e", "f"}

    assert len(preprocessor.test_data) == 2
    test_texts = {s["text"] for s in preprocessor.test_data}
    assert test_texts == {"g", "i"}


def test_determine_final_labels():
    preprocessor = DatasetPreprocessor(
        SAMPLE_TRAIN_DATA, SAMPLE_TEST_DATA, min_train_freq=2
    )
    preprocessor._filter_by_common_leaves()
    preprocessor._determine_final_labels()

    expected_labels = ["animal", "car", "cat", "mammal", "vehicle"]
    assert preprocessor.final_labels == expected_labels



def test_run_preprocessing_integration():
    preprocessor = DatasetPreprocessor(
        SAMPLE_TRAIN_DATA, SAMPLE_TEST_DATA, min_train_freq=2
    )
    preprocessor.run_preprocessing()

    train_data, test_data = preprocessor.get_processed_data()

    assert len(train_data) == 6
    assert len(test_data) == 2
    
    expected_labels = ["animal", "car", "cat", "mammal", "vehicle"]
    assert preprocessor.final_labels == expected_labels