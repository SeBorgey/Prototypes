import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from train import prepare_data

CONFIG = {
    "dataset_name": "DeepPavlov/events",
    "bert_model_name": "bert-base-uncased",
    "cache_dir": "./.cache",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def run_baseline_trial(config):
    train_dataset, test_dataset, _, _, _ = prepare_data(config, verbose=True)

    X_train = train_dataset.tensors[0].cpu().numpy()
    y_train = train_dataset.tensors[1].cpu().numpy()
    
    X_test = test_dataset.tensors[0].cpu().numpy()
    y_test = test_dataset.tensors[1].cpu().numpy()

    logreg_model = LogisticRegression(solver='liblinear', random_state=42)
    multilabel_baseline = OneVsRestClassifier(logreg_model, n_jobs=-1)
    
    multilabel_baseline.fit(X_train, y_train)

    scores = multilabel_baseline.predict_proba(X_test)
    
    predictions = (scores > 0.5).astype(int)

    mAP = average_precision_score(y_test, scores, average="macro")
    of1 = f1_score(y_test, predictions, average="micro")
    
    print("\n--- Baseline Results ---")
    print(f"Overall F1 (OF1 / micro-F1): {of1:.4f}")
    print(f"mean Average Precision (mAP):  {mAP:.4f}")
    return mAP, of1

if __name__ == "__main__":
    run_baseline_trial(CONFIG)