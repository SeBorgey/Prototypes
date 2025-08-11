import os
import shutil
import subprocess
import tempfile
import json
import glob
import random
import ast

DATASETS_DIR = "datasets"


def download_multi3nlu():
    name = "nlupp_english"
    target_path = os.path.join("datasets", name)
    repo_url = "https://github.com/PolyAI-LDN/task-specific-datasets.git"
    folder_in_repo = "nlupp/data"

    if os.path.exists(target_path):
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True,
        )
        source_folder = os.path.join(temp_dir, folder_in_repo)
        shutil.move(source_folder, target_path)


def download_blendx_datasets():
    repo_url = "https://github.com/HYU-NLP/BlendX"
    datasets = {"banking77": "Banking77", "clinc150": "CLINC150"}

    banking_path = os.path.join("datasets", "banking77")
    clinc_path = os.path.join("datasets", "clinc150")

    if os.path.exists(banking_path) and os.path.exists(clinc_path):
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True,
        )
        for name, original_name in datasets.items():
            target_dir = os.path.join("datasets", name)
            os.makedirs(target_dir, exist_ok=True)
            source_blend_file = os.path.join(
                temp_dir, "v1.0", "BlendX", f"Blend{original_name}.json"
            )
            dest_blend_file = os.path.join(target_dir, "blend.json")
            shutil.move(source_blend_file, dest_blend_file)


def load_nlupp_english(base_path):
    all_samples = []
    for domain in ["banking", "hotels"]:
        domain_path = os.path.join(base_path, domain)
        for file_path in glob.glob(os.path.join(domain_path, "fold*.json")):
            with open(file_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
                for sample in samples:
                    if "text" in sample:
                        labels = sample.get("intents", [])
                        all_samples.append(
                            {"text": sample["text"], "labels": labels}
                        )
    random.seed(42)
    random.shuffle(all_samples)

    train_size = int(0.9 * len(all_samples))
    dev_size = int(0.05 * len(all_samples))

    with open(os.path.join(base_path, "ontology.json"), "r") as f:
        ontology = json.load(f)
    all_labels = sorted(list(ontology["intents"].keys()))

    splits = {
        "train": all_samples[:train_size],
        "dev": all_samples[train_size : train_size + dev_size],
        "test": all_samples[train_size + dev_size :],
    }
    return splits, all_labels


def _load_blendx_style_dataset(file_path):
    splits = {"train": [], "dev": [], "test": []}
    unique_labels = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            
            split_name = sample.get("split")
            text = sample.get("utterance")
            labels_str = sample.get("intent")

            if not (split_name in splits and text and labels_str):
                continue

            labels = ast.literal_eval(labels_str)
            unique_labels.update(labels)
            splits[split_name].append({"text": text, "labels": labels})

    return splits, sorted(list(unique_labels))


def load_banking77(file_path):
    return _load_blendx_style_dataset(file_path)


def load_clinc150(file_path):
    return _load_blendx_style_dataset(file_path)


def load_datasets():
    all_loaded_data = {}

    nlupp_path = os.path.join(DATASETS_DIR, "nlupp_english")
    all_loaded_data["nlupp_english"] = load_nlupp_english(nlupp_path)

    banking_path = os.path.join(DATASETS_DIR, "banking77", "blend.json")
    all_loaded_data["banking77"] = load_banking77(banking_path)

    clinc_path = os.path.join(DATASETS_DIR, "clinc150", "blend.json")
    all_loaded_data["clinc150"] = load_clinc150(clinc_path)

    return all_loaded_data

if __name__ == "__main__":
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # download_multi3nlu()
    # download_blendx_datasets()

    all_loaded_data = {}

    nlupp_path = os.path.join(DATASETS_DIR, "nlupp_english")
    all_loaded_data["nlupp_english"] = load_nlupp_english(nlupp_path)

    banking_path = os.path.join(DATASETS_DIR, "banking77", "blend.json")
    all_loaded_data["banking77"] = load_banking77(banking_path)

    clinc_path = os.path.join(DATASETS_DIR, "clinc150", "blend.json")
    all_loaded_data["clinc150"] = load_clinc150(clinc_path)
    
    print("\n--- Data Loading Summary ---")
    for name, (splits, labels) in all_loaded_data.items():
        print(f"\nDataset: {name}")
        print(f"  Total unique labels: {len(labels)}")
        print(f"  Train samples: {len(splits['train'])}")
        print(f"  Dev samples:   {len(splits['dev'])}")
        print(f"  Test samples:  {len(splits['test'])}")

    