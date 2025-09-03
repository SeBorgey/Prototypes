import json

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autointent import Embedder
from autointent.configs import EmbedderConfig
from autointent.modules.scoring import LinearScorer
from datasets import load_dataset as hf_load_dataset
from hiclass import (
    LocalClassifierPerLevel,
    LocalClassifierPerNode,
    LocalClassifierPerParentNode,
)
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

# --- 1. Настройка моделей и констант ---

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_CONFIG = EmbedderConfig(model_name=EMBEDDER_MODEL, device="cpu")

HICLASS_MODELS = {
    "hiclass LCPN (siblings)": LocalClassifierPerNode(
        binary_policy="siblings", n_jobs=-1
    ),
    "hiclass LCPPN": LocalClassifierPerParentNode(n_jobs=-1),
    "hiclass LCPL": LocalClassifierPerLevel(n_jobs=-1),
}

# --- 2. Функции для загрузки и подготовки данных ---


def load_mock_data():
    """Создает небольшой датасет-заглушку для проверки работоспособности пайплайна."""
    X = [
        "A text about a striker scoring a goal in the finals.",
        "A movie review about a great new fantasy film.",
        "The central bank announced new interest rates today.",
        "A key player was traded to another team before the deadline.",
        "The director's new historical movie is a masterpiece.",
        "An in-depth analysis of the current economic situation in Europe.",
    ]
    y = [
        ["Sports", "Football"],
        ["Arts", "Movies"],
        ["Business", "Economics"],
        ["Sports", "Football"],
        ["Arts", "Movies"],
        ["Business", "Economics"],
    ]
    return np.array(X), np.array(y, dtype=object)


# !!! РАСКОММЕНТИРУЙТЕ НУЖНУЮ ФУНКЦИЮ В `main`, ЧТОБЫ ЗАПУСТИТЬ НА РЕАЛЬНЫХ ДАННЫХ !!!
#
def load_dbpedia_data():
    """Загружает и объединяет DBPedia из Kaggle Hub, создавая иерархию из колонок l1, l2, l3."""
    print("Загрузка DBPedia... Это может занять некоторое время.")
    dataset_handle = "danofer/dbpedia-classes"
    try:
        df_train = kagglehub.dataset_load(
            kagglehub.KaggleDatasetAdapter.PANDAS, dataset_handle, "DBPEDIA_train.csv"
        )
        df_test = kagglehub.dataset_load(
            kagglehub.KaggleDatasetAdapter.PANDAS, dataset_handle, "DBPEDIA_test.csv"
        )
        df = pd.concat([df_train, df_test], ignore_index=True)
    except Exception as e:
        print(f"Ошибка загрузки DBPedia: {e}")
        print(
            "Убедитесь, что у вас настроен kagglehub (pip install kagglehub; kagglehub auth login)."
        )
        return np.array([]), np.array([])

    X = df["text"].astype(str).values
    y_raw = df[["l1", "l2", "l3"]].values.tolist()
    # Удаляем пустые строки в конце путей, если они есть
    y = [[item for item in sublist if pd.notna(item) and item] for sublist in y_raw]
    return X, np.array(y, dtype=object)


def load_wos_data():
    """Загружает датасет Web of Science из Hugging Face и преобразует метки ID в строки."""
    print("Загрузка Web of Science...")
    dataset_id = "marcelsun/wos_hierarchical_multi_label_text_classification"
    config_name = "wos_46985"
    dataset_dict = hf_load_dataset(dataset_id, config_name)

    # Загрузка маппинга меток
    mapping_path = hf_hub_download(
        repo_id=dataset_id, filename="label.json", repo_type="dataset"
    )
    with open(mapping_path, "r") as f:
        id_to_label = json.load(f)

    full_df = pd.concat(
        [
            dataset_dict["train"].to_pandas(),
            dataset_dict["validation"].to_pandas(),
            dataset_dict["test"].to_pandas(),
        ]
    )

    X = np.array([" ".join(tokens) for tokens in full_df["token"]])
    y_indices = full_df["label"].tolist()
    y = [[id_to_label[str(idx)] for idx in path] for path in y_indices]

    return X, np.array(y, dtype=object)


def load_wiki_subjects_data():
    """Загружает датасет Wiki Academic Subjects из Hugging Face."""
    print("Загрузка Wiki Academic Subjects...")
    dataset_id = "meliascosta/wiki_academic_subjects"
    dataset_dict = hf_load_dataset(dataset_id)

    full_df = pd.concat(
        [
            dataset_dict["train"].to_pandas(),
            dataset_dict["validation"].to_pandas(),
            dataset_dict["test"].to_pandas(),
        ]
    )

    X = np.array([" ".join(tokens) for tokens in full_df["token"]])
    y = full_df["label"].values

    return X, y


# --- 3. Функции для метрик и эмбеддингов ---


def get_embeddings(texts: np.ndarray, config: EmbedderConfig) -> np.ndarray:
    """Получает эмбеддинги для списка текстов с помощью AutoIntent."""
    embedder = Embedder(config)
    # Преобразуем ndarray в list, так как embedder ожидает list[str]
    embeddings = embedder.embed(texts.tolist())
    embedder.clear_ram()
    return embeddings


def calculate_hierarchical_metrics(y_true, y_pred):
    """Рассчитывает метрики на основе полного совпадения иерархического пути."""
    y_true_tuples = [tuple(path) for path in y_true]
    y_pred_tuples = [tuple(path) for path in y_pred]

    true_positives = sum(
        1
        for true_path, pred_path in zip(y_true_tuples, y_pred_tuples)
        if true_path == pred_path
    )

    num_predicted = len(y_pred)
    num_true = len(y_true)

    precision_val = true_positives / num_predicted if num_predicted > 0 else 0
    recall_val = true_positives / num_true if num_true > 0 else 0

    f1_val = 0.0
    if precision_val + recall_val > 0:
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

    return {"precision": precision_val, "recall": recall_val, "f1": f1_val}


# --- 4. Функции для оценки моделей ---


def evaluate_hiclass_models(X_train_emb, y_train, X_test_emb, y_test):
    results = []
    for name, model in tqdm(HICLASS_MODELS.items(), desc="Evaluating hiclass models"):
        print(f"Training {name}...")
        model.fit(X_train_emb, y_train)
        y_pred = model.predict(X_test_emb)
        metrics = calculate_hierarchical_metrics(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)
    return results


def evaluate_autointent_multiclass(
    X_train_list, y_train, X_test_list, y_test, embedder_config
):
    # Создаем словарь для маппинга листа на полный путь.
    # Если один и тот же лист принадлежит разным путям, будет взята последняя запись.
    leaf_to_path = {path[-1]: path for path in y_train if path}
    y_train_flat = [path[-1] for path in y_train if path]

    # Отфильтровываем обучающие примеры, где нет пути
    X_train_filtered = [X_train_list[i] for i, path in enumerate(y_train) if path]

    scorer = LinearScorer(embedder_config=embedder_config)
    scorer.fit(X_train_filtered, y_train_flat)

    scores = scorer.predict(X_test_list)

    # Классы в обученном классификаторе могут отличаться
    le_classes = scorer._clf.classes_
    y_pred_flat_indices = np.argmax(scores, axis=1)
    y_pred_flat = le_classes[y_pred_flat_indices]

    y_pred_paths = [
        leaf_to_path.get(leaf, [f"unknown_leaf_{leaf}"]) for leaf in y_pred_flat
    ]

    metrics = calculate_hierarchical_metrics(y_test, y_pred_paths)
    metrics["model"] = "AutoIntent (LogReg Multiclass)"
    return metrics


def evaluate_autointent_multilabel(
    X_train_list, y_train, X_test_list, y_test, embedder_config
):
    mlb = MultiLabelBinarizer()
    y_train_mlb = mlb.fit_transform(y_train)

    scorer = LinearScorer(embedder_config=embedder_config)
    scorer.fit(X_train_list, y_train_mlb.tolist())

    scores = scorer.predict(X_test_list)
    y_pred_mlb = (scores > 0.5).astype(int)

    # inverse_transform возвращает кортежи, преобразуем в списки
    y_pred_paths = [list(path) for path in mlb.inverse_transform(y_pred_mlb)]

    metrics = calculate_hierarchical_metrics(y_test, y_pred_paths)
    metrics["model"] = "AutoIntent (LogReg Multilabel)"
    return metrics


# --- 5. Функция для визуализации результатов ---


def plot_results(df):
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]

        melted_df = subset.melt(
            id_vars=["dataset", "model"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=melted_df, x="model", y="score", hue="metric", palette="viridis"
        )

        ax.set_title(
            f"Сравнение моделей на датасете: {dataset_name}", fontsize=18, pad=20
        )
        ax.set_xlabel("Модель", fontsize=14)
        ax.set_ylabel("Результат", fontsize=14)
        plt.xticks(rotation=10, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 1.05)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=10, padding=3)

        plt.legend(title="Метрика", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


# --- 6. Основной пайплайн ---


def main():
    datasets_to_run = {
        # "Mock Dataset": load_mock_data,
        "DBPedia": load_dbpedia_data,
        "Web of Science": load_wos_data,
        "Wiki Subjects": load_wiki_subjects_data,
    }

    all_results = []

    for name, loader in datasets_to_run.items():
        print(f"\n{'=' * 20} Processing dataset: {name} {'=' * 20}")
        X, y = loader()

        if X.size == 0:
            print(f"Пропущен датасет {name} из-за ошибки загрузки.")
            continue

        # Стратификация по первому уровню иерархии
        y_stratify = [labels[0] if len(labels) > 0 else "" for labels in y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y_stratify
        )

        X_train_list, X_test_list = X_train.tolist(), X_test.tolist()

        print("Получение эмбеддингов для обучающей выборки...")
        X_train_emb = get_embeddings(X_train, EMBEDDER_CONFIG)
        print("Получение эмбеддингов для тестовой выборки...")
        X_test_emb = get_embeddings(X_test, EMBEDDER_CONFIG)

        # Оценка hiclass
        hiclass_results = evaluate_hiclass_models(
            X_train_emb, y_train, X_test_emb, y_test
        )
        for result in hiclass_results:
            result["dataset"] = name
            all_results.append(result)

        # Оценка autointent (Multiclass)
        print("\nОценка AutoIntent (Multiclass)...")
        autointent_mc_results = evaluate_autointent_multiclass(
            X_train_list, y_train, X_test_list, y_test, EMBEDDER_CONFIG
        )
        autointent_mc_results["dataset"] = name
        all_results.append(autointent_mc_results)

        # Оценка autointent (Multilabel)
        print("Оценка AutoIntent (Multilabel)...")
        autointent_ml_results = evaluate_autointent_multilabel(
            X_train_list, y_train, X_test_list, y_test, EMBEDDER_CONFIG
        )
        autointent_ml_results["dataset"] = name
        all_results.append(autointent_ml_results)

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 20 + " Итоговые результаты " + "=" * 20)
    print(results_df.to_string())

    plot_results(results_df)


if __name__ == "__main__":
    main()
