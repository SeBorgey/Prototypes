import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset as hf_load_dataset

from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from autointent import Embedder
from autointent.configs import EmbedderConfig
from autointent.modules.scoring import LinearScorer

# --- 1. Настройка моделей и констант ---

# Единая модель для получения эмбеддингов
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_CONFIG = EmbedderConfig(model_name=EMBEDDER_MODEL, device="cpu")

HICLASS_MODELS = {
    "LCPN (siblings)": LocalClassifierPerNode(binary_policy="siblings", n_jobs=-1),
    "LCPPN": LocalClassifierPerParentNode(n_jobs=-1),
    "LCPL": LocalClassifierPerLevel(n_jobs=-1),
}

# --- 2. Функции для загрузки данных ---

def load_mock_data():
    """Создает небольшой датасет-заглушку для проверки работоспособности пайплайна."""
    X = [
        "A text about a striker scoring a goal.",
        "A movie review about a great new film.",
        "The central bank announced new interest rates.",
        "A player was traded to another team.",
        "The director's new movie is a masterpiece.",
        "Analysis of the current economic situation.",
        "This team is going to the finals!",
        "Blockbuster film hits the cinemas.",
        "Government bonds yield decreased.",
    ]
    y = [
        ["Sports", "Football"],
        ["Arts", "Movies"],
        ["Business", "Economics"],
        ["Sports", "Football"],
        ["Arts", "Movies"],
        ["Business", "Economics"],
        ["Sports", "Football"],
        ["Arts", "Movies"],
        ["Business", "Economics"],
    ]
    return np.array(X), np.array(y, dtype=object)

# !!! РАСКОММЕНТИРУЙТЕ НУЖНУЮ ФУНКЦИЮ В main, ЧТОБЫ ЗАПУСТИТЬ НА РЕАЛЬНЫХ ДАННЫХ !!!
#
# def load_wos_data():
#     """Загружает датасет Web of Science из Hugging Face."""
#     dataset = hf_load_dataset("marcelsun/wos_hierarchical_multi_label_text_classification", "wos_46985")
#     data = pd.DataFrame(dataset['train'])
#     X = data['text'].values
#     y = np.array([path.split('~') for path in data['y_true']], dtype=object)
#     return X, y
#
# def load_dbpedia_data():
#     """Загружает датасет DBPedia. Требует наличия файла DBPedia.csv."""
#     df = pd.read_csv("DBPedia.csv", encoding='utf-8')
#     X = df['text'].astype(str).values
#     y_df = df[['l1', 'l2', 'l3']].copy()
#     y_df.fillna('', inplace=True)
#     y_raw = y_df.values.tolist()
#     y = [[item for item in sublist if item] for sublist in y_raw]
#     return X, np.array(y, dtype=object)
#
# def load_wiki_subjects_data():
#     """Загружает датасет Wiki Academic Subjects из Hugging Face."""
#     dataset = hf_load_dataset("meliascosta/wiki_academic_subjects")
#     data = pd.DataFrame(dataset['train'])
#     X = data['text'].values
#     y = np.array([path.split(' / ') for path in data['label']], dtype=object)
#     return X, y
#

# --- 3. Функции для метрик и эмбеддингов ---

def get_embeddings(texts: np.ndarray, config: EmbedderConfig) -> np.ndarray:
    """Получает эмбеддинги для списка текстов с помощью AutoIntent."""
    embedder = Embedder(config)
    embeddings = embedder.embed(texts.tolist())
    embedder.clear_ram()
    return embeddings

def calculate_hierarchical_metrics(y_true, y_pred):
    """Рассчитывает метрики на основе полного совпадения иерархического пути."""
    y_true_tuples = [tuple(path) for path in y_true]
    y_pred_tuples = [tuple(path) for path in y_pred]

    true_positives = sum(1 for true_path, pred_path in zip(y_true_tuples, y_pred_tuples) if true_path == pred_path)
    
    num_predicted = len(y_pred)
    num_true = len(y_true)
    
    precision_val = true_positives / num_predicted if num_predicted > 0 else 0
    recall_val = true_positives / num_true if num_true > 0 else 0
    
    if precision_val + recall_val == 0:
        f1_val = 0.0
    else:
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        
    return {
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
    }

# --- 4. Функции для оценки моделей ---

def evaluate_hiclass_models(X_train_emb, y_train, X_test_emb, y_test):
    """Оценивает все модели из библиотеки hiclass."""
    results = []
    for name, model in tqdm(HICLASS_MODELS.items(), desc="Evaluating hiclass models"):
        model.fit(X_train_emb, y_train)
        y_pred = model.predict(X_test_emb)
        metrics = calculate_hierarchical_metrics(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)
    return results

def evaluate_autointent_multiclass(X_train, y_train, X_test, y_test, embedder_config):
    """Оценивает AutoIntent в режиме мультиклассовой классификации по последнему уровню."""
    y_train_list = [list(path) for path in y_train]
    leaf_to_path = {path[-1]: path for path in y_train_list if path}
    y_train_flat = [path[-1] for path in y_train_list if path]
    
    X_train_filtered = [x for x, path in zip(X_train, y_train_list) if path]

    scorer = LinearScorer(embedder_config=embedder_config, cv=2)
    scorer.fit(X_train_filtered, y_train_flat)
    
    scores = scorer.predict(X_test)
    
    le = scorer._clf.classes_
    y_pred_flat_indices = np.argmax(scores, axis=1)
    y_pred_flat = le[y_pred_flat_indices]
    
    y_pred_paths = [leaf_to_path.get(leaf, ["<unknown>"]) for leaf in y_pred_flat]
    
    metrics = calculate_hierarchical_metrics(y_test, y_pred_paths)
    metrics["model"] = "AutoIntent (LogReg Multiclass)"
    return metrics

def evaluate_autointent_multilabel(X_train, y_train, X_test, y_test, embedder_config):
    """Оценивает AutoIntent в режиме мультилейбл классификации по всему пути."""
    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(y_train)
    
    scorer = LinearScorer(embedder_config=embedder_config)
    scorer.fit(X_train, y_train_mlb.tolist())
    
    scores = scorer.predict(X_test)
    y_pred_mlb = (scores > 0.5).astype(int)
    
    y_pred_paths = mlb.inverse_transform(y_pred_mlb)
    
    y_pred_paths_list = [list(path) for path in y_pred_paths]
    
    metrics = calculate_hierarchical_metrics(y_test, y_pred_paths_list)
    metrics["model"] = "AutoIntent (LogReg Multilabel)"
    return metrics

# --- 5. Функция для визуализации результатов ---

def plot_results(df):
    """Строит графики для сравнения результатов."""
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name]
        
        melted_df = subset.melt(id_vars=['dataset', 'model'], 
                                value_vars=['precision', 'recall', 'f1'],
                                var_name='metric', value_name='score')

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=melted_df, x='model', y='score', hue='metric')
        
        ax.set_title(f'Сравнение моделей на датасете: {dataset_name}', fontsize=16)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('Результат', fontsize=12)
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, 1)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
            
        plt.legend(title='Метрика')
        plt.tight_layout()
        plt.show()


# --- 6. Основной пайплайн ---

def main():
    """Основной пайплайн для запуска эксперимента."""
    
    datasets_to_run = {
        "Mock Dataset": load_mock_data,
        # "Web of Science": load_wos_data,
        # "DBPedia": load_dbpedia_data,
        # "Wiki Subjects": load_wiki_subjects_data,
    }
    
    all_results = []
    
    for name, loader in datasets_to_run.items():
        print(f"--- Processing dataset: {name} ---")
        X, y = loader()
        
        # ИСПРАВЛЕНИЕ: Преобразуем y в список для безопасной работы
        y_list = [list(path) for path in y]
        y_first_level = [path[0] for path in y_list if len(path) > 0]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y_first_level
        )
        
        print("Получение эмбеддингов для обучающей выборки...")
        X_train_emb = get_embeddings(X_train, EMBEDDER_CONFIG)
        print("Получение эмбеддингов для тестовой выборки...")
        X_test_emb = get_embeddings(X_test, EMBEDDER_CONFIG)

        # Оценка hiclass
        hiclass_results = evaluate_hiclass_models(X_train_emb, y_train, X_test_emb, y_test)
        for result in hiclass_results:
            result['dataset'] = name
            all_results.append(result)

        # Оценка autointent (Multiclass)
        print("Оценка AutoIntent (Multiclass)...")
        autointent_mc_results = evaluate_autointent_multiclass(X_train.tolist(), y_train, X_test.tolist(), y_test, EMBEDDER_CONFIG)
        autointent_mc_results['dataset'] = name
        all_results.append(autointent_mc_results)

        # Оценка autointent (Multilabel)
        print("Оценка AutoIntent (Multilabel)...")
        autointent_ml_results = evaluate_autointent_multilabel(X_train.tolist(), y_train, X_test.tolist(), y_test, EMBEDDER_CONFIG)
        autointent_ml_results['dataset'] = name
        all_results.append(autointent_ml_results)
    
    results_df = pd.DataFrame(all_results)
    print("\n--- Итоговые результаты ---")
    print(results_df.to_string())

    plot_results(results_df)

if __name__ == '__main__':
    main()