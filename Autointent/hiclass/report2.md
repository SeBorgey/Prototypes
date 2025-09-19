check_datasets.py
================================================================================
Анализ датасета: unified_datasets/dbpedia_classes
================================================================================

 1. Анализ глубины иерархических путей 

Распределение глубины в обучающей выборке (train):
  Глубина 3: 240942 примеров

Распределение глубины в тестовой выборке (test):
  Глубина 3: 60794 примеров

 2. Анализ дочерних меток с несколькими родителями 

Все дочерние метки имеют уникального родителя. Проблем не обнаружено.



================================================================================
Анализ датасета: unified_datasets/wiki_academic_subjects
================================================================================

 1. Анализ глубины иерархических путей 

Распределение глубины в обучающей выборке (train):
  Глубина 1: 175 примеров
  Глубина 2: 1650 примеров
  Глубина 3: 26054 примеров
  Глубина 4: 21446 примеров
  Глубина 5: 2188 примеров

Распределение глубины в тестовой выборке (test):
  Глубина 1: 14 примеров
  Глубина 2: 179 примеров
  Глубина 3: 3287 примеров
  Глубина 4: 2686 примеров
  Глубина 5: 273 примеров

 2. Анализ дочерних меток с несколькими родителями 

Все дочерние метки имеют уникального родителя. Проблем не обнаружено.






run_comparison.py
--- Processing dataset: unified_datasets/custom_intents ---
After common leaf filtering: train=98, test=8, leaves=8
External split: train=49, val=49, test=8
Preparing data for hiclass...
Running hiclass: LCPN exclusive...
Results for LCPN exclusive: {'accuracy': 0.5}
Running hiclass: LCPN less exclusive...
Results for LCPN less exclusive: {'accuracy': 0.5}
Running hiclass: LCPN less inclusive...
Results for LCPN less inclusive: {'accuracy': 0.5}
Running hiclass: LCPN inclusive...
Results for LCPN inclusive: {'accuracy': 0.5}
Running hiclass: LCPN siblings...
Results for LCPN siblings: {'accuracy': 0.5}
Running hiclass: LCPN exclusive siblings...
Results for LCPN exclusive siblings: {'accuracy': 0.5}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.5}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.625}
Running autointent: Multiclass LogReg...
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
[I 2025-09-19 03:23:26,204] A new study created in memory with name: NodeType.scoring
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multiclass LogReg: {'accuracy': 0.625}
Running autointent: Multilabel LogReg...
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 20924.55 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 22940.16 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 5903.31 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.0)}
--- Processing dataset: unified_datasets/dbpedia_classes ---
After common leaf filtering: train=240942, test=60794, leaves=219
External split: train=120470, val=120472, test=60794
Preparing data for hiclass...
Running hiclass: LCPN exclusive...
Results for LCPN exclusive: {'accuracy': 0.8304108958120867}
Running hiclass: LCPN less exclusive...
Results for LCPN less exclusive: {'accuracy': 0.8304108958120867}
Running hiclass: LCPN less inclusive...
Results for LCPN less inclusive: {'accuracy': 0.8304108958120867}
Running hiclass: LCPN inclusive...
Results for LCPN inclusive: {'accuracy': 0.7465045892686778}
Running hiclass: LCPN siblings...
Results for LCPN siblings: {'accuracy': 0.856038424844557}
Running hiclass: LCPN exclusive siblings...
Results for LCPN exclusive siblings: {'accuracy': 0.856038424844557}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.8718623548376484}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.8692798631443892}
Running autointent: Multiclass LogReg...
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
Results for Autointent Multiclass LogReg: {'accuracy': 0.9082475244267526}
Running autointent: Multilabel LogReg...
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████| 120470/120470 [00:11<00:00, 10918.54 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████| 120472/120472 [00:10<00:00, 10953.36 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████| 60794/60794 [00:05<00:00, 10442.31 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.5190314833700694)}
--- Processing dataset: unified_datasets/wiki_academic_subjects ---
After common leaf filtering: train=49134, test=6427, leaves=1402
External split: train=24567, val=24567, test=6427
Preparing data for hiclass...
Running hiclass: LCPN exclusive...
Results for LCPN exclusive: {'accuracy': 0.2599968881282091}
Running hiclass: LCPN less exclusive...
Results for LCPN less exclusive: {'accuracy': 0.2599968881282091}
Running hiclass: LCPN less inclusive...
Results for LCPN less inclusive: {'accuracy': 0.2599968881282091}
Running hiclass: LCPN inclusive...
Results for LCPN inclusive: {'accuracy': 0.27944608682122296}
Running hiclass: LCPN siblings...
Results for LCPN siblings: {'accuracy': 0.35848763030963127}
Running hiclass: LCPN exclusive siblings...
Results for LCPN exclusive siblings: {'accuracy': 0.35848763030963127}
Running hiclass: LCPPN...
Results for LCPPN: {'accuracy': 0.3771588610549245}
Running hiclass: LCPL...
Results for LCPL: {'accuracy': 0.35428660339194024}
Running autointent: Multiclass LogReg...
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multiclass LogReg: {'accuracy': 0.4894974327057725}
Running autointent: Multilabel LogReg...
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████| 24567/24567 [00:09<00:00, 2471.71 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████| 24567/24567 [00:09<00:00, 2548.68 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 6427/6427 [00:02<00:00, 2543.34 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'accuracy': np.float64(0.022872257662984286)}

--- Final Comparison Results ---
                                    dataset                            model  accuracy
0           unified_datasets/custom_intents           hiclass_LCPN exclusive  0.500000
1           unified_datasets/custom_intents      hiclass_LCPN less exclusive  0.500000
2           unified_datasets/custom_intents      hiclass_LCPN less inclusive  0.500000
3           unified_datasets/custom_intents           hiclass_LCPN inclusive  0.500000
4           unified_datasets/custom_intents            hiclass_LCPN siblings  0.500000
5           unified_datasets/custom_intents  hiclass_LCPN exclusive siblings  0.500000
6           unified_datasets/custom_intents                    hiclass_LCPPN  0.500000
7           unified_datasets/custom_intents                     hiclass_LCPL  0.625000
8           unified_datasets/custom_intents     autointent_multiclass_logreg  0.625000
9           unified_datasets/custom_intents     autointent_multilabel_logreg  0.000000
10         unified_datasets/dbpedia_classes           hiclass_LCPN exclusive  0.830411
11         unified_datasets/dbpedia_classes      hiclass_LCPN less exclusive  0.830411
12         unified_datasets/dbpedia_classes      hiclass_LCPN less inclusive  0.830411
13         unified_datasets/dbpedia_classes           hiclass_LCPN inclusive  0.746505
14         unified_datasets/dbpedia_classes            hiclass_LCPN siblings  0.856038
15         unified_datasets/dbpedia_classes  hiclass_LCPN exclusive siblings  0.856038
16         unified_datasets/dbpedia_classes                    hiclass_LCPPN  0.871862
17         unified_datasets/dbpedia_classes                     hiclass_LCPL  0.869280
18         unified_datasets/dbpedia_classes     autointent_multiclass_logreg  0.908248
19         unified_datasets/dbpedia_classes     autointent_multilabel_logreg  0.519031
20  unified_datasets/wiki_academic_subjects           hiclass_LCPN exclusive  0.259997
21  unified_datasets/wiki_academic_subjects      hiclass_LCPN less exclusive  0.259997
22  unified_datasets/wiki_academic_subjects      hiclass_LCPN less inclusive  0.259997
23  unified_datasets/wiki_academic_subjects           hiclass_LCPN inclusive  0.279446
24  unified_datasets/wiki_academic_subjects            hiclass_LCPN siblings  0.358488
25  unified_datasets/wiki_academic_subjects  hiclass_LCPN exclusive siblings  0.358488
26  unified_datasets/wiki_academic_subjects                    hiclass_LCPPN  0.377159
27  unified_datasets/wiki_academic_subjects                     hiclass_LCPL  0.354287
28  unified_datasets/wiki_academic_subjects     autointent_multiclass_logreg  0.489497
29  unified_datasets/wiki_academic_subjects     autointent_multilabel_logreg  0.022872