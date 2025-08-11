import datasets

def inspect_dataset_v2(dataset_name, sample_text_length=150):
    print(f"Анализ датасета: {dataset_name}\n")
    
    try:
        builder = datasets.load_dataset_builder(dataset_name)
        builder.download_and_prepare()
        dataset = builder.as_dataset()
    except Exception as e:
        print(f"Не удалось загрузить датасет. Ошибка: {e}")
        return

    print("--- 1. Общая структура (Splits) ---")
    print(dataset)

    if 'train' not in dataset:
        print("В датасете отсутствует обязательный сплит 'train'.")
        return
        
    train_split = dataset['train']
    
    print("--- 2. Детальная схема данных 'train' (Features) ---")
    features = train_split.features
    for column_name, feature_type in features.items():
        print(f"  - Колонка: '{column_name}' | Тип: {feature_type}")

    print("--- 3. Пример первой записи из 'train' (с обрезкой) ---")
    first_record = train_split[0]
    for key, value in first_record.items():
        display_value = value
        if isinstance(value, str) and len(value) > sample_text_length:
            display_value = f"'{value[:sample_text_length]}...'"
        
        print(f"  - Поле: '{key}' | Тип значения: {type(value).__name__} | Значение: {display_value}")

if __name__ == '__main__':
    DATASET_ID = "DeepPavlov/events"
    inspect_dataset_v2(DATASET_ID)