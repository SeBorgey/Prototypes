import numpy as np
from autointent import Dataset
from autointent.modules.scoring import LinearScorer
from autointent.metrics.scoring import scoring_map, scoring_f1
import data_loader
def prepare_autointent_dataset(data_splits, all_labels):
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    num_labels = len(all_labels)
    
    autointent_data = {}
    for split_name, samples in data_splits.items():
        processed_samples = []
        for sample in samples:
            if not sample["labels"]:
                continue
            one_hot_labels = [0] * num_labels
            for label in sample["labels"]:
                if label in label_to_id:
                    one_hot_labels[label_to_id[label]] = 1
            
            processed_samples.append({
                "utterance": sample["text"],
                "label": one_hot_labels
            })
        autointent_data[split_name] = processed_samples

    autointent_data["intents"] = [
        {"id": i, "name": name} for name, i in label_to_id.items()
    ]
    
    if 'dev' in autointent_data:
        autointent_data['validation'] = autointent_data.pop('dev')

    return Dataset.from_dict(autointent_data)

def run_evaluation(all_loaded_data):
    results = {}

    for dataset_name, (data_splits, all_labels) in all_loaded_data.items():
        print(f"--- Processing dataset: {dataset_name} ---")
        
        print("1. Preparing data...")
        dataset = prepare_autointent_dataset(data_splits, all_labels)
        
        train_utterances = dataset["train"]["utterance"]
        train_labels = dataset["train"]["label"]
        
        test_utterances = dataset["test"]["utterance"]
        test_labels = dataset["test"]["label"]

        print("2. Initializing and training LinearScorer...")
        scorer = LinearScorer(
            embedder_config="sentence-transformers/all-MiniLM-L6-v2"
        )
        scorer.fit(train_utterances, train_labels)

        print("3. Making predictions on the test set...")
        scores = scorer.predict(test_utterances)

        print("4. Calculating metrics...")
        map_score = scoring_map(test_labels, scores)
        f1 = scoring_f1(test_labels, scores)

        results[dataset_name] = {
            "Mean Average Precision": map_score,
            "Overall F1-score": f1
        }
        print(f"--- Finished with {dataset_name} ---\n")

    return results

if __name__ == '__main__':
    all_loaded_data = data_loader.load_datasets()

    final_results = run_evaluation(all_loaded_data)

    print("===== Final Evaluation Results =====")
    for dataset_name, metrics in final_results.items():
        print(f"\nDataset: {dataset_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")