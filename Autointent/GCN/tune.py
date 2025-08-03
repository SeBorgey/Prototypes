import optuna
import torch
from train import run_trial

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
    study = optuna.create_study(
        study_name="mlgcn_qwen_embedding_v1",
        directions=['maximize', 'maximize']
    )
    
    try:
        study.optimize(objective, n_trials=5, timeout=20800)
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