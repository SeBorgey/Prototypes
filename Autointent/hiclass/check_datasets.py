import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, DefaultDict

DATASET_DIRS = [
    "unified_datasets/dbpedia_classes",
    "unified_datasets/wiki_academic_subjects",
]


def load_raw_data(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
    train_path = Path(dataset_path) / "train.json"
    test_path = Path(dataset_path) / "test.json"
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    return train_data, test_data


def analyze_path_depths(data: List[Dict]) -> Counter:
    depth_counts = Counter()
    for item in data:
        path = item.get("labels", [[]])[0]
        depth = len(path)
        depth_counts[depth] += 1
    return depth_counts


def find_non_unique_children(data: List[Dict]) -> Dict[str, Set[str]]:
    child_to_parents: DefaultDict[str, Set[str]] = defaultdict(set)

    for item in data:
        path = item.get("labels", [[]])[0]
        if len(path) < 2:
            continue
        
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i+1]
            child_to_parents[child].add(parent)
            
    non_unique_children = {
        child: parents 
        for child, parents in child_to_parents.items() 
        if len(parents) > 1
    }
    
    return non_unique_children


def main():
    for dataset_dir in DATASET_DIRS:
        print("=" * 80)
        print(f"Анализ датасета: {dataset_dir}")
        print("=" * 80)

        train_raw, test_raw = load_raw_data(dataset_dir)
        combined_raw = train_raw + test_raw
        print("\n 1. Анализ глубины иерархических путей ")
        
        train_depths = analyze_path_depths(train_raw)
        test_depths = analyze_path_depths(test_raw)

        print("\nРаспределение глубины в обучающей выборке (train):")
        if not train_depths:
            print("  Нет данных для анализа.")
        else:
            for depth, count in sorted(train_depths.items()):
                print(f"  Глубина {depth}: {count} примеров")

        print("\nРаспределение глубины в тестовой выборке (test):")
        if not test_depths:
            print("  Нет данных для анализа.")
        else:
            for depth, count in sorted(test_depths.items()):
                print(f"  Глубина {depth}: {count} примеров")

        print("\n 2. Анализ дочерних меток с несколькими родителями ")
        
        non_unique = find_non_unique_children(combined_raw)

        if not non_unique:
            print("\nВсе дочерние метки имеют уникального родителя. Проблем не обнаружено.")
        else:
            total_non_unique_count = len(non_unique)
            print(f"\nОбнаружено {total_non_unique_count} дочерних меток, принадлежащих разным родителям.")
            
            parent_count_stats = Counter(len(parents) for parents in non_unique.values())
            print("\nСтатистика по количеству родителей у таких меток:")
            for num_parents, count in sorted(parent_count_stats.items()):
                print(f"  - {count} меток имеют {num_parents} родителя(ей)")
            
            num_examples_to_show = 5
            print(f"\nПримеры первых {num_examples_to_show} таких меток (если есть):")
            
            sorted_non_unique_items = sorted(non_unique.items())
            
            for i, (child, parents) in enumerate(sorted_non_unique_items):
                if i >= num_examples_to_show:
                    break
                parent_list = ", ".join(sorted(list(parents)))
                print(f"  - Метка '{child}' имеет родителей: [{parent_list}]")
            
            if total_non_unique_count > num_examples_to_show:
                print(f"  ... и еще {total_non_unique_count - num_examples_to_show} меток.")
        
        print("\n" * 2)

if __name__ == "__main__":
    main()