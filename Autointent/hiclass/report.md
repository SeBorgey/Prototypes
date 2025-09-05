Произведено сравнение качества работы `hiclass` и `Autointent` на иерархических датасетах при базовых настройках.

Обнаруженный датасет `Web of Science (WoS) Hierarchical Multi-Label Text Classification` не использовался, так как он оказался всего лишь двухуровневым. И каждая метка на каждом уровне отнесена в нём к примеру независимо от иерархии.

Датасеты сначала переведены в `json` формат чтобы унифицировать работу с ними.

LLM сгенерирован `custom_intents` маленький датасет чтобы отлаживать на нём работу пайплайна.

Чтобы `autointent` мог прожевать большие датасеты, они были подвергнуты выравниванию после разделения.

В `hiclass` выбор того, что считать `Positive`  или `Negative`  примерами, выполняет параметр `binary_policy`. Он есть только у `LocalClassifierPerNode`. Для данной задачи использовалась политика `siblings`.

Каждое обучение выполнялось на 500 эпохах.

Метрика `accuracy`, которой производилось сравнение считалась так:
Для `hiclass` засчитывалось правильным только полное совпадение выданного пути.
Для `autointent_multiclass_logreg` засчитывалось правильным совпадение предсказанной метки с концом пути.
Для `autointent_multilabel_logreg` засчитывалось только полное совпадение предсказанного и истинного наборов меток.

Для обучения `autointent_multiclass_logreg` в `train` оставлялись только метки конца пути.

Ниже представлен вывод файла `run_comparison.py` в самом конце результаты `accuracy` для каждой модели и каждого датасета.

```
--- Processing dataset: unified_datasets/custom_intents ---
Preparing data for hiclass...
Running hiclass: LCPN...
Results for LCPN: {'accuracy': 0.5}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.5}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.5}
Running autointent: Multiclass LogReg...
Filter: 100%|████████████████████████████████████████████████████| 98/98 [00:00<00:00, 87773.18 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
[I 2025-09-05 16:12:57,503] A new study created in memory with name: NodeType.scoring
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multiclass LogReg: {'accuracy': 0.75}
Running autointent: Multilabel LogReg...
Classes: total=14, train_present=14, test_present=14
Map: 100%|███████████████████████████████████████████████████████| 49/49 [00:00<00:00, 21068.26 examples/s]
Map: 100%|███████████████████████████████████████████████████████| 49/49 [00:00<00:00, 22542.60 examples/s]
Map: 100%|██████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 5665.11 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.0)}
--- Processing dataset: unified_datasets/dbpedia_classes ---
Preparing data for hiclass...
Running hiclass: LCPN...
Results for LCPN: {'accuracy': 0.8652992071586012}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.8822910155607461}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.880136197651084}
Running autointent: Multiclass LogReg...
Filter: 100%|████████████████████████████████████████████████████████████████████| 240942/240942 [00:00<00:00, 658518.78 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
Results for Autointent Multiclass LogReg: {'accuracy': 0.9081159324933381}
Running autointent: Multilabel LogReg...
Classes: total=298, train_present=298, test_present=298
Map: 100%|████████████████████████████████████████████████████████████████████████| 120470/120470 [00:11<00:00, 10707.49 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████| 120472/120472 [00:10<00:00, 11253.07 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████| 60794/60794 [00:05<00:00, 10789.79 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.5190314833700694)}
--- Processing dataset: unified_datasets/wiki_academic_subjects ---
Preparing data for hiclass...
Running hiclass: LCPN...
Results for LCPN: {'accuracy': 0.41124398198478024}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.4258425221307656}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.40984624941761144}
Running autointent: Multiclass LogReg...
Filter: 100%|████████████████████████████████████████████████████████████████████████████| 49134/49134 [00:00<00:00, 807522.35 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multiclass LogReg: {'accuracy': 0.4877859032207873}
Running autointent: Multilabel LogReg...
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/preprocessing/_label.py:909: UserWarning: unknown class(es) ['Accompanying', 'Acoustical engineering', 'Aerial', 'Aerobiology', 'Alternative education', 'Andrology', 'Anthroponics', 'Anthrozoology', 'Applied physics', 'Arachnology', 'Architectural analytics', 'Architectural engineering', 'Architectural sociology', 'Arithmetic combinatorics', 'Arms race', 'Arthropodology', 'Asian American Literature', 'Asian studies', 'Astronautics', 'Astrophysical plasma', 'Athletic director', 'Attrition', 'Bacteriology', 'Batoning', 'Batrachology', 'Bengali Literature', 'Bibliometrics', 'Biochemical systems theory', 'Biocultural anthropology', 'Biomechanical engineering', 'Biophysics', 'Bryozoology', 'Bulgarian Literature', 'Business English', 'Canadian studies', 'Carcinology', 'Cardiac electrophysiology', 'Cetology', 'Charge', 'Choreography', 'Citation analysis', 'Civil procedure', 'Classical Archaeology', 'Clinical biochemistry', 'Cnidariology', 'Cold war (general term)', 'Communication design', 'Comparative anatomy', 'Comparative law', 'Comparative politics', 'Computational economics', 'Computational finance', 'Computational mathematics', 'Computational number theory', 'Computational systems biology', 'Computer engineering', 'Computer-aided engineering', 'Conceptual systems', 'Conchology', 'Conflict theory', 'Consumer economics', 'Consumer education', 'Convex geometry', 'Cosmochemistry', 'Counselor education', 'Counter-offensive', 'Creative Nonfiction', 'Criminal procedure', 'Cross-cultural studies', 'Cultural geography', 'Culturology', 'Cyber', 'Cynology', 'Dental surgery', 'Dermatopathology', 'Differential psychology', 'Dogmatic theology', 'Domestic policy', 'Early music', 'Econometrics', 'Education economics', 'Electronic', 'Electronic media', 'Elementary particle physics', 'Empirical sociology', 'Energy economics', 'Engineering cybernetics', 'Enterprise systems engineering', 'Environmental science', 'Ethnochoreology', 'Ethnolinguistics', 'Ethnology', 'European studies', 'Evolutionary anthropology', 'Federal law', 'Felinology', 'Feminine psychology', 'Feminist philosophy', 'Femtochemistry', 'Field theory', 'Figurational sociology', 'Film theory', 'Financial econometrics', 'Five laws of library science', 'Fogponics', 'Foxhole', 'Gaelic Literature', 'Geometric number theory', 'Geometric topology', 'Geostatistics', 'Gerontology', 'Grammatology', 'Hand Surgery', 'Helminthology', 'Hematology', 'Hematopathology', 'Hepatology', 'High-energy astrophysics', 'Histopathology', 'Historical geography', 'Housing', 'Human anatomy', 'Human biology', 'Humanistic informatics', 'Hydrology', 'Immunochemistry', 'Indexer', 'Industrial', 'Industrial organization', 'Industrial policy', 'Industrial sociology', 'Infiltration', 'Information architecture', 'Integrated geography', 'Integrated library system', 'International organizations', 'Investment policy', 'Land management', 'Landscape design', 'Landscape planning', 'Law 2', 'Law enforcement', 'Legal psychology', 'Library binding', 'Library technical services', 'Limacology', 'Limited war', 'Linnaean taxonomy', 'Live action', 'Logical Reasoning', 'Malacology', 'Mammalogy', 'Marine chemistry', 'Marxist sociology', 'Mass transfer', 'Materiel', 'Mechanochemistry', 'Medical cybernetics', 'Medical social work', 'Medical toxicology', 'Mesosociology', 'Microsociology', 'Military exercises', 'Military policy', 'Military sports', 'Mixed media', 'Mock combat', 'Modern', 'Modern Language', 'Molecular pathology', 'Molecular physics', 'Morale', 'Multilinear algebra', 'Myriapodology', 'Nanoengineering', 'Nationalism studies', 'Native American Studies', 'Nematology', 'Neuro-ophthalmology', 'Neuroethology', 'Nomology', 'Non-fiction', 'Normative ethics', 'Nuclear', 'Oenology', 'Offensive', 'Optical engineering', 'Optimal maintenance', 'Organizational studies', 'Orthodontics', 'Orthoptics', 'P-adic analysis', 'Palaeogeography', 'Paleobiology', 'Parallel algorithms', 'Paramilitary', 'Pastoral counseling', 'Pastoral theology', 'Pedology', 'Pedology (children study)', 'Pharmaceutics', 'Pharmacocybernetics', 'Phenomenological sociology', 'Phenomenology', 'Philosophy of Action', 'Philosophy of chemistry', 'Philosophy of engineering', 'Physical education', 'Phytochemistry', 'Planetary cartography', 'Planktology', 'Plant science', 'Playwrighting', 'Police science', 'Policy sociology', 'Political behavior', 'Polymer science', 'Pomology', 'Population geography', 'Port management', 'Prospect research', 'Protistology', 'Psephology', 'Purification', 'Quantitative psychology', 'Quantum chemistry', 'Rajasthani Literature', 'Recording', 'Regional Geography', 'Registrar', 'Regression', 'Regulation', 'Religion geography', 'Scandinavian studies', 'Scoutcraft', 'Security policy', 'Sindhology', 'Slow fire', 'Social change', 'Social development', 'Social dynamics', 'Social geography', 'Social policy', 'Sociocybernetics', 'Sociology of conflict', 'Sociology of immigration', 'Sociology of markets', 'Sociology of peace, war, and social conflict', 'Sociology of punishment', 'Solid-state chemistry', 'Special operations', 'Strategic studies', 'Structural mechanics', 'Supranational law', 'Surface chemistry', 'Surgical pathology', 'Surgical strike', 'Synthetic chemistry', 'Systems analysis', 'Tactical objective', 'Tax law', 'Television studies', 'Teratology', 'Teuthology', 'Thai studies', 'Theological Anthropology', 'Thermal physics', 'Tourism geography', 'Traffic psychology', 'Traumatology', 'Travel', 'Usage', 'Vehicle Dynamics', 'Weapons Systems', 'Welsh Literature', 'Woodcraft', 'Word usage', 'Zoosemiotics'] will be ignored
  warnings.warn(
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/preprocessing/_label.py:909: UserWarning: unknown class(es) ['Evolutionary anthropology', 'Infiltration', 'Justification', 'Maneuver', 'Optimal maintenance', 'Pharmaceutics', 'Purification'] will be ignored
  warnings.warn(
Classes: total=1446, train_present=1446, test_present=1446
Map: 100%|█████████████████████████████████████████████████████████████████████████████████| 25752/25752 [00:09<00:00, 2687.72 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████| 25761/25761 [00:09<00:00, 2812.80 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████| 6439/6439 [00:02<00:00, 2750.97 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.02562509706476161)}

--- Final Comparison Results ---
                                    dataset                         model  accuracy
                           dataset                         model  accuracy
0           unified_datasets/custom_intents                  hiclass_LCPN      0.50
1           unified_datasets/custom_intents                 hiclass_LCPPN      0.50
2           unified_datasets/custom_intents                  hiclass_LCPL      0.50
3           unified_datasets/custom_intents  autointent_multiclass_logreg      0.75
4           unified_datasets/custom_intents  autointent_multilabel_logreg      0.00
5          unified_datasets/dbpedia_classes                  hiclass_LCPN  0.865299
6          unified_datasets/dbpedia_classes                 hiclass_LCPPN  0.882291
7          unified_datasets/dbpedia_classes                  hiclass_LCPL  0.880136
8          unified_datasets/dbpedia_classes  autointent_multiclass_logreg  0.908116
9          unified_datasets/dbpedia_classes  autointent_multilabel_logreg  0.519031
10  unified_datasets/wiki_academic_subjects                  hiclass_LCPN  0.411244
11  unified_datasets/wiki_academic_subjects                 hiclass_LCPPN  0.425843
12  unified_datasets/wiki_academic_subjects                  hiclass_LCPL  0.409846
13  unified_datasets/wiki_academic_subjects  autointent_multiclass_logreg  0.487786
14  unified_datasets/wiki_academic_subjects  autointent_multilabel_logreg  0.025625
```