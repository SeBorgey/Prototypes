import optuna
import torch
from train import run_trial, get_cache_path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os

BASE_CONFIG = {
    "dataset_name": "DeepPavlov/events",
    "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
    "sentence_embedding_dim": 1024,
    "label_embedding_dim": 512,
    "cache_dir": "./.cache",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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
def ensure_embeddings_cached(config):
    print("--- Checking for cached embeddings ---")
    dataset = load_dataset(config["dataset_name"], cache_dir=config["cache_dir"])
    
    tasks = {
        "train": (config["sentence_embedding_dim"], [item['utterance'] for item in dataset['train']]),
        "test": (config["sentence_embedding_dim"], [item['utterance'] for item in dataset['test']]),
        "labels": (config["label_embedding_dim"], [f"event type {i}" for i in range(len(dataset['train'][0]['label']))])
    }
    
    paths_to_check = [get_cache_path(config, name, dim) for name, (dim, _) in tasks.items()]
    
    if all(os.path.exists(p) for p in paths_to_check):
        print("All embeddings found in cache. Skipping generation.")
        return

    print("Some embeddings are missing. Loading SentenceTransformer to generate them...")
    model = SentenceTransformer(
        config["embedding_model_name"], 
        cache_folder=config["cache_dir"],
        device=config["device"]
    )

    for name, (dim, texts) in tasks.items():
        cache_path = get_cache_path(config, name, dim)
        if not os.path.exists(cache_path):
            print(f"Generating embeddings for '{name}' with dim={dim}...")
            full_embeddings = model.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=2
            )
            embeddings = full_embeddings[:, :dim]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(embeddings.cpu(), cache_path)
            print(f"Saved to {cache_path}")

    print("Embedding generation complete. Releasing model from memory.")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def objective(trial):
    config = BASE_CONFIG.copy()

    config['epochs'] = trial.suggest_int('epochs', 10, 100, log=True)
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    config['p_reweight'] = trial.suggest_float('p_reweight', 0.0, 0.9, step=0.1)
    config['tau_threshold'] = trial.suggest_float('tau_threshold', 0.1, 0.9, step=0.1)
    
    arch_key = trial.suggest_categorical('gcn_architecture', list(GCN_ARCHITECTURES_MAP.keys()))
    config['gcn_hidden_dims'] = list(GCN_ARCHITECTURES_MAP[arch_key])

    try:
        mAP, of1 = run_trial(config)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    return mAP, of1

def main():
    ensure_embeddings_cached(BASE_CONFIG)
    
    study = optuna.create_study(
        study_name="mlgcn_qwen_embedding_v2",
        directions=['maximize', 'maximize']
    )
    
    try:
        study.optimize(objective, n_trials=200, timeout=20800)
    except KeyboardInterrupt:
        pass

    print("\n--- Optuna Tuning Report ---")
    print(f"Number of finished trials: {len(study.trials)}")
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
    fig1.update_layout(title="Hyperparameter Importances for mAP")
    fig1.show()
    
    fig2 = optuna.visualization.plot_param_importances(study, target_name="OF1", target=lambda t: t.values[1])
    fig2.update_layout(title="Hyperparameter Importances for OF1")
    fig2.show()

    fig3 = optuna.visualization.plot_pareto_front(study, target_names=["mAP", "OF1"])
    fig3.show()

if __name__ == "__main__":
    main()