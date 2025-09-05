import json
from tqdm import tqdm
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter


from datasets import load_dataset
from huggingface_hub import hf_hub_download


BASE_OUTPUT_DIR = Path("unified_datasets")

# --- Унифицированный формат ---
# Каждый элемент в итоговом JSON файле будет выглядеть так:
# {
#   "text": "...",
#   "labels": [["level1", "level2", ...], ["other_path_level1", ...]]
# }

def save_to_json(data: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[+] Данные успешно сохранены в: {output_path}")

def process_dbpedia():
    dataset_name = "danofer/dbpedia-classes"
    output_dir = BASE_OUTPUT_DIR / "dbpedia_classes"
    print("\n" + "="*80)
    print(f"[*] Обработка датасета: DBPedia Classes ({dataset_name})")
    print("="*80)

    file_paths = {"train": "DBPEDIA_train.csv", "test": "DBPEDIA_test.csv"}

    for split_name, file_path in file_paths.items():
        print(f"\n[*] Загрузка сплита '{split_name}'...")
        try:
            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                dataset_name,
                file_path,
            )
            print("[+] Загрузка завершена.")

            unified_data = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  -> Преобразование '{split_name}'"):
                # Для DBPedia у нас одна иерархическая метка на запись
                unified_entry = {
                    "text": str(row['text']),
                    "labels": [[str(row['l1']), str(row['l2']), str(row['l3'])]]
                }
                unified_data.append(unified_entry)
            
            save_to_json(unified_data, output_dir / f"{split_name}.json")

        except Exception as e:
            print(f"[!] Ошибка при обработке {file_path}: {e}")
            print("    Убедитесь, что вы аутентифицированы в Kaggle.")
            break

def process_wos():
    dataset_name = "marcelsun/wos_hierarchical_multi_label_text_classification"
    output_dir = BASE_OUTPUT_DIR / "wos_hierarchical"
    print("\n" + "="*80)
    print(f"[*] Обработка датасета: Web of Science ({dataset_name})")
    print("="*80)

    try:
        # --- Шаг 1: Скачать и распарсить таксономию для преобразования ID в имена ---
        print("[*] Загрузка файла таксономии 'label.taxonomy'...")
        id_to_name_map = {}
        try:
            taxonomy_path = hf_hub_download(
                repo_id=dataset_name,
                filename="label.taxonomy",
                repo_type="dataset"
            )
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Ожидаемый формат: "ID<tab>LabelName"
                    parts = line.split('\t')
                    if len(parts) == 2:
                        label_id, label_name = parts
                        id_to_name_map[int(label_id)] = label_name
            
            print(f"[+] Таксономия успешно загружена. Найдено {len(id_to_name_map)} меток.")

        except Exception as e:
            print(f"[!] Не удалось загрузить или распарсить 'label.taxonomy': {e}")
            print("    Обработка продолжится с числовыми ID в качестве меток.")

        # --- Шаг 2: Загрузить основной датасет ---
        dataset_dict = load_dataset(dataset_name)
        
        # --- Шаг 3: Обработать каждый сплит ---
        for split_name, split_data in dataset_dict.items():
            print(f"\n[*] Обработка сплита '{split_name}'...")
            unified_data = []
            for item in tqdm(split_data, desc=f"  -> Преобразование '{split_name}'"):
                # Получаем числовой путь, например [6, 275]
                numeric_path = item['label']
                
                # Преобразуем числовой путь в текстовый с помощью карты
                # Если карта не загрузилась, используем ID как строки
                if id_to_name_map:
                    string_path = [id_to_name_map.get(label_id, f"UNKNOWN_ID_{label_id}") for label_id in numeric_path]
                else:
                    string_path = [str(label_id) for label_id in numeric_path]

                # Текст представлен как одна строка с именем 'token'
                # ВАЖНО: Весь массив меток - это ОДИН путь. Оборачиваем его в список.
                unified_entry = {
                    "text": str(item['token']),
                    "labels": [string_path] # -> [["Computer Science", "Artificial Intelligence"]]
                }
                unified_data.append(unified_entry)
            
            save_to_json(unified_data, output_dir / f"{split_name}.json")
    
    except Exception as e:
        print(f"[!] Ошибка при обработке {dataset_name}: {e}")

def process_wiki_academic():
    dataset_name = "meliascosta/wiki_academic_subjects"
    output_dir = BASE_OUTPUT_DIR / "wiki_academic_subjects"
    print("\n" + "="*80)
    print(f"[*] Обработка датасета: Wiki Academic Subjects ({dataset_name})")
    print("="*80)
    
    try:
        dataset_dict = load_dataset(dataset_name)
        
        for split_name, split_data in dataset_dict.items():
            print(f"\n[*] Обработка сплита '{split_name}'...")
            unified_data = []
            for item in tqdm(split_data, desc=f"  -> Преобразование '{split_name}'"):
                # Текст представлен списком токенов, объединяем их в строку.
                # Метки уже представляют собой иерархический путь.
                unified_entry = {
                    "text": " ".join(item['token']),
                    "labels": [item['label']] # Оборачиваем в еще один список
                }
                unified_data.append(unified_entry)
            
            save_to_json(unified_data, output_dir / f"{split_name}.json")

        # Скачиваем файл с таксономией для дополнительной информации
        print("\n[*] Скачивание файла с иерархией `label.taxonomy`...")
        taxonomy_path = hf_hub_download(
            repo_id=dataset_name,
            filename="label.taxonomy",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False # Копируем файл, а не создаем симлинк
        )
        print(f"[+] Файл таксономии сохранен в: {taxonomy_path}")
        
    except Exception as e:
        print(f"[!] Ошибка при обработке {dataset_name}: {e}")


def main():
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # process_dbpedia()
    process_wos()
    # process_wiki_academic()

if __name__ == "__main__":
    main()