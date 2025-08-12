import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, f1_score
import os
import copy
import functools

from mlgcn_model import TextMLGCN
from data_loader import load_datasets

BASE_CONFIG = {
    "cache_dir": "./.cache",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_epochs": 50,
    "patience": 10,
}

EMBEDDING_MODELS = {
    "NovaSearch/stella_en_400M_v5": {
        "max_dim": 1024,
        "label_dim": 512
    },
}

GCN_ARCHITECTURES = [
    (1024,),
    (512, 1024),
    (1024, 2048),
    (512, 1024, 2048),
    (512, 512, 1024, 2048)
]
GCN_ARCHITECTURES_MAP = {
    f"{len(arch)}_layer{'s' if len(arch) > 1 else ''}_{'_'.join(map(str, arch))}": arch
    for arch in GCN_ARCHITECTURES
}

def get_cache_path(config, name, dim):
    safe_ds = config["dataset_name"].replace("/", "_")
    safe_model = config["embedding_model_name"].replace("/", "_")
    return os.path.join(config["cache_dir"], f"emb_{name}_{safe_ds}_{safe_model}_{dim}.pt")

def ensure_embeddings_cached(config, data_splits, all_labels):
    model_name = config["embedding_model_name"]
    model_info = EMBEDDING_MODELS[model_name]
    
    config["sentence_embedding_dim"] = model_info["max_dim"]
    config["label_embedding_dim"] = model_info["label_dim"]
    
    tasks = {
        "train": (config["sentence_embedding_dim"], [item['text'] for item in data_splits['train'] if item['labels']]),
        "dev": (config["sentence_embedding_dim"], [item['text'] for item in data_splits['dev'] if item['labels']]),
        "test": (config["sentence_embedding_dim"], [item['text'] for item in data_splits['test'] if item['labels']]),
        "labels": (config["label_embedding_dim"], all_labels)
    }

    paths_to_check = [get_cache_path(config, name, dim) for name, (dim, _) in tasks.items()]
    
    if all(os.path.exists(p) for p in paths_to_check):
        return

    print(f"Some embeddings for {model_name} on {config['dataset_name']} are missing. Generating...")
    model = SentenceTransformer(
        model_name, cache_folder=config["cache_dir"], device=config["device"], trust_remote_code=True
    )
    
    for name, (dim, texts) in tasks.items():
        cache_path = get_cache_path(config, name, dim)
        if not os.path.exists(cache_path) and texts:
            print(f"Generating for '{name}' (dim={dim})...")
            full_embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=2)
            embeddings = full_embeddings[:, :dim]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(embeddings.cpu(), cache_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prepare_data_for_trial(config, data_splits, all_labels):
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    num_classes = len(all_labels)
    
    def process_split(split_name):
        samples = [item for item in data_splits[split_name] if item['labels']]
        if not samples:
            return TensorDataset(torch.empty(0, config["sentence_embedding_dim"]), torch.empty(0, num_classes)), torch.empty(0, num_classes)

        path = get_cache_path(config, split_name, config["sentence_embedding_dim"])
        embeddings = torch.load(path)
        
        multi_hot_labels_list = []
        for item in samples:
            label_row = torch.zeros(num_classes)
            for label in item['labels']:
                label_row[label_to_id[label]] = 1
            multi_hot_labels_list.append(label_row)
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ:
        # torch.stack правильно собирает список 1D-тензоров в одну 2D-матрицу
        multi_hot_labels = torch.stack(multi_hot_labels_list)
        return TensorDataset(embeddings, multi_hot_labels), multi_hot_labels
    
    # Теперь мы сразу получаем правильно сформированный тензор
    train_dataset, full_train_labels = process_split('train')
    val_dataset, _ = process_split('dev')
    test_dataset, _ = process_split('test')

    labels_path = get_cache_path(config, "labels", config["label_embedding_dim"])
    label_embeddings = torch.load(labels_path)
    
    return train_dataset, val_dataset, test_dataset, full_train_labels, num_classes, label_embeddings

def run_trial(config, data_splits, all_labels):
    device = torch.device(config["device"])
    train_ds, val_ds, test_ds, full_train_labels, num_classes, label_embeds = prepare_data_for_trial(
        config, data_splits, all_labels
    )
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])
    
    model = TextMLGCN(
        num_classes, config["sentence_embedding_dim"], config["label_embedding_dim"],
        config["gcn_hidden_dims"], config["p_reweight"], config["tau_threshold"]
    ).to(device)
    model.set_correlation_matrix(full_train_labels.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for _ in range(config["max_epochs"]):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features, label_embeds.to(device))
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        current_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features, label_embeds.to(device))
                current_val_loss += loss_function(logits, labels).item()
        
        avg_val_loss = current_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break
    
    if best_model_state is None:
        return 0.0, 0.0

    model.load_state_dict(best_model_state)
    model.eval()
    all_scores, all_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features, label_embeds.to(device))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(labels.cpu().numpy())
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    mAP = average_precision_score(all_true, all_scores, average="macro")
    of1 = f1_score(all_true, all_scores > 0.5, average="micro")
    
    return mAP, of1

def objective(trial, data_splits, all_labels):
    config = BASE_CONFIG.copy()
    config['dataset_name'] = trial.study.study_name
    
    model_name = trial.suggest_categorical('embedding_model_name', list(EMBEDDING_MODELS.keys()))
    config['embedding_model_name'] = model_name
    
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
    config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    config['p_reweight'] = trial.suggest_float('p_reweight', 0.0, 0.9, step=0.1)
    config['tau_threshold'] = trial.suggest_float('tau_threshold', 0.1, 0.9, step=0.1)
    arch_key = trial.suggest_categorical('gcn_architecture', list(GCN_ARCHITECTURES_MAP.keys()))
    config['gcn_hidden_dims'] = list(GCN_ARCHITECTURES_MAP[arch_key])
    
    model_info = EMBEDDING_MODELS[model_name]
    config["sentence_embedding_dim"] = model_info["max_dim"]
    config["label_embedding_dim"] = model_info["label_dim"]

    try:
        mAP, of1 = run_trial(config, data_splits, all_labels)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return mAP, of1

def run_tuning_for_dataset(dataset_name, dataset_tuple):
    print(f"\n{'='*20} STARTING TUNING FOR: {dataset_name.upper()} {'='*20}")
    
    data_splits, all_labels = dataset_tuple
    
    for model_name in EMBEDDING_MODELS.keys():
        config = BASE_CONFIG.copy()
        config['dataset_name'] = dataset_name
        config['embedding_model_name'] = model_name
        ensure_embeddings_cached(config, data_splits, all_labels)

    study = optuna.create_study(
        study_name=dataset_name,
        directions=['maximize', 'maximize']
    )
    
    objective_with_data = functools.partial(objective, data_splits=data_splits, all_labels=all_labels)
    
    try:
        study.optimize(objective_with_data, n_trials=50, timeout=10800)
    except KeyboardInterrupt:
        pass

    print(f"\n--- Optuna Tuning Report for: {dataset_name.upper()} ---")
    print(f"Number of finished trials: {len(study.trials)}")
    if not study.best_trials:
        print("No successful trials completed.")
        return

    print("\nBest trials (Pareto front):")
    best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    
    for i, trial in enumerate(best_trials):
        print(f"\n--- Pareto Trial #{i+1} ---")
        print(f"  Trial Number: {trial.number}")
        print(f"  Metrics: mAP={trial.values[0]:.4f}, OF1={trial.values[1]:.4f}")
        print(f"  Hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    fig1 = optuna.visualization.plot_param_importances(study, target_name="mAP", target=lambda t: t.values[0])
    fig1.update_layout(title=f"{dataset_name}: Hyperparameter Importances for mAP")
    fig1.show()
    
    fig2 = optuna.visualization.plot_param_importances(study, target_name="OF1", target=lambda t: t.values[1])
    fig2.update_layout(title=f"{dataset_name}: Hyperparameter Importances for OF1")
    fig2.show()

    fig3 = optuna.visualization.plot_pareto_front(study, target_names=["mAP", "OF1"])
    fig3.update_layout(title=f"{dataset_name}: Pareto Front")
    fig3.show()

def main():
    all_loaded_data = load_datasets()
    for dataset_name, dataset_tuple in all_loaded_data.items():
        run_tuning_for_dataset(dataset_name, dataset_tuple)

if __name__ == "__main__":
    main()