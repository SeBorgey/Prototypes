--- Processing dataset: unified_datasets/dbpedia_classes ---
Starting preprocessing...
  - Initial state                       -> Train: 240942, Test: 60794, CommonLeaves: 0, FinalLabels: 0
  - After filtering by common leaves    -> Train: 240942, Test: 60794, CommonLeaves: 219, FinalLabels: 0
  - After determining final labels      -> Train: 240942, Test: 60794, CommonLeaves: 219, FinalLabels: 298
Preprocessing finished.

After preprocessing: train=240942, test=60794, leaf_classes=219, all_final_labels=298
Running hiclass: LCPN exclusive...
Results for LCPN exclusive: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.8715169260124355, 'mrr': 0.8416770221309701}
Running hiclass: LCPN less exclusive...
Results for LCPN less exclusive: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.8715169260124355, 'mrr': 0.8416770221309701}
Running hiclass: LCPN less inclusive...
Results for LCPN less inclusive: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.8715169260124355, 'mrr': 0.8416770221309701}
Running hiclass: LCPN inclusive...
Results for LCPN inclusive: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.799618383393098, 'mrr': 0.7504287401436711}
Running hiclass: LCPN siblings...
Results for LCPN siblings: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.8926703293088134, 'mrr': 0.868182789870395}
Running hiclass: LCPN exclusive siblings...
Results for LCPN exclusive siblings: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.8926703293088134, 'mrr': 0.868182789870395}
Running hiclass: LCPPN...
Results for LCPPN: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.9057143797085239, 'mrr': 0.8848559160789623}
Running hiclass: LCPL...
Results for LCPL: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.9040036845741356, 'mrr': 0.8827479132167063}
Running autointent: Multiclass LogReg...
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
[I 2025-09-28 23:50:57,740] A new study created in memory with name: NodeType.scoring
Storage directory must be provided for study persistence.
Results for Autointent Multiclass LogReg: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.9996052241997565, 'mrr': 0.9456068392125498}
Running autointent: Multilabel LogReg...
Map: 100%|█████████████████████████████████████████████████████████████████████████████| 120470/120470 [00:11<00:00, 10863.76 examples/s]
Map: 100%|█████████████████████████████████████████████████████████████████████████████| 120472/120472 [00:11<00:00, 10177.31 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████| 60794/60794 [00:05<00:00, 10199.86 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'n_classes': 219, 'k': 44, 'accuracy_at_k': 0.9999506530249695, 'mrr': 0.9786895721994955}
--- Processing dataset: unified_datasets/wiki_academic_subjects ---
Starting preprocessing...
  - Initial state                       -> Train: 51513, Test: 6439, CommonLeaves: 0, FinalLabels: 0
  - After filtering by common leaves    -> Train: 49134, Test: 6427, CommonLeaves: 1402, FinalLabels: 0
  - After determining final labels      -> Train: 49134, Test: 6427, CommonLeaves: 1402, FinalLabels: 1446
Preprocessing finished.

After preprocessing: train=49134, test=6427, leaf_classes=1402, all_final_labels=1446
Running hiclass: LCPN exclusive...
Results for LCPN exclusive: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.45371090711062706, 'mrr': 0.31617822109970145}
Running hiclass: LCPN less exclusive...
Results for LCPN less exclusive: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.45371090711062706, 'mrr': 0.31617822109970145}
Running hiclass: LCPN less inclusive...
Results for LCPN less inclusive: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.45371090711062706, 'mrr': 0.31617822109970145}
Running hiclass: LCPN inclusive...
Results for LCPN inclusive: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.48000622374358176, 'mrr': 0.34697083064046175}
Running hiclass: LCPN siblings...
Results for LCPN siblings: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.5707172864477984, 'mrr': 0.4589942838972765}
Running hiclass: LCPN exclusive siblings...
Results for LCPN exclusive siblings: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.5707172864477984, 'mrr': 0.4589942838972765}
Running hiclass: LCPPN...
Results for LCPPN: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.5826979928426949, 'mrr': 0.4748613547370621}
Running hiclass: LCPL...
Results for LCPL: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.5644935428660339, 'mrr': 0.45257347105182494}
Running autointent: Multiclass LogReg...
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multiclass LogReg: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 0.9712151859343395, 'mrr': 0.5981343137069058}
Running autointent: Multilabel LogReg...
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 24567/24567 [00:09<00:00, 2684.51 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 24567/24567 [00:09<00:00, 2712.33 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████| 6427/6427 [00:02<00:00, 2458.95 examples/s]
Memory storage is not compatible with resuming optimization. Modules from previous runs won't be available. Set dump_modules=True in LoggingConfig to enable proper resuming.
Storage directory must be provided for study persistence.
Storage directory must be provided for study persistence.
/home/kaneki/.cache/pypoetry/virtualenvs/prototypes-0Fl_Nofl-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
Results for Autointent Multilabel LogReg: {'n_classes': 1402, 'k': 281, 'accuracy_at_k': 1.0, 'mrr': 0.8582169646942486}

--- Final Comparison Results ---
                                dataset                           model  n_classes   k  accuracy_at_k      mrr
       unified_datasets/dbpedia_classes          hiclass_LCPN exclusive        219  44       0.871517 0.841677
       unified_datasets/dbpedia_classes     hiclass_LCPN less exclusive        219  44       0.871517 0.841677
       unified_datasets/dbpedia_classes     hiclass_LCPN less inclusive        219  44       0.871517 0.841677
       unified_datasets/dbpedia_classes          hiclass_LCPN inclusive        219  44       0.799618 0.750429
       unified_datasets/dbpedia_classes           hiclass_LCPN siblings        219  44       0.892670 0.868183
       unified_datasets/dbpedia_classes hiclass_LCPN exclusive siblings        219  44       0.892670 0.868183
       unified_datasets/dbpedia_classes                   hiclass_LCPPN        219  44       0.905714 0.884856
       unified_datasets/dbpedia_classes                    hiclass_LCPL        219  44       0.904004 0.882748
       unified_datasets/dbpedia_classes    autointent_multiclass_logreg        219  44       0.999605 0.945607
       unified_datasets/dbpedia_classes    autointent_multilabel_logreg        219  44       0.999951 0.978690
unified_datasets/wiki_academic_subjects          hiclass_LCPN exclusive       1402 281       0.453710 0.316178
unified_datasets/wiki_academic_subjects     hiclass_LCPN less exclusive       1402 281       0.453710 0.316178
unified_datasets/wiki_academic_subjects     hiclass_LCPN less inclusive       1402 281       0.453710 0.316178
unified_datasets/wiki_academic_subjects          hiclass_LCPN inclusive       1402 281       0.480006 0.346970
unified_datasets/wiki_academic_subjects           hiclass_LCPN siblings       1402 281       0.570717 0.458994
unified_datasets/wiki_academic_subjects hiclass_LCPN exclusive siblings       1402 281       0.570717 0.458994
unified_datasets/wiki_academic_subjects                   hiclass_LCPPN       1402 281       0.582697 0.474861
unified_datasets/wiki_academic_subjects                    hiclass_LCPL       1402 281       0.564493 0.452573
unified_datasets/wiki_academic_subjects    autointent_multiclass_logreg       1402 281       0.971215 0.598134
unified_datasets/wiki_academic_subjects    autointent_multilabel_logreg       1402 281       1.000000 0.858217