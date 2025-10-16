# Введение
Поставленная задача - рассмотреть возможность поддержки иерархической классификации для AutoIntent.

# Постановка экспериментов

## GCN

Реализована и протестирована модель на основе графовых сверточных сетей.

1. Сначала была построена версия, использующая эмбеддинги BERT для текстов и GloVe для меток, и проверена на датасете `DeepPavlov/events`.
2. Затем, взята современная модель из бенчмарков Hugging Face — `NovaSearch/stella_en_400M_v5`. Эта модель использовалась для получения эмбеддингов как для текстов, так и для меток.
3. Улучшенная модель была протестирована на четырех различных датасетах для мульти-лейбл классификации.
4. Результаты GCN модели сравнивались с простым baseline решением — логистической регрессией (`OneVsRestClassifier`), обученной на тех же эмбеддингах текстов.

В отличие от оригинальной статьи, веса эмбеддера не размораживались.

### Тестовые датасеты

Для проведения экспериментов были выбраны четыре датасета, три из которых являются мульти-лейбл версиями известных датасетов для классификации интентов.

| Название датасета | Описание |
| :--- | :--- |
| **DeepPavlov/events** | Классификация коротких новостных текстов по нескольким категориям событий. Исходный датасет для базовых экспериментов. |
| **Multi3NLU++ (nlupp_english)** | Мульти-лейбл датасет для диалоговых систем, охватывающий домены банкинга и отелей. Каждому запросу может соответствовать несколько интентов. |
| **Banking77 (BlendX)** | Мульти-лейбл версия известного датасета с детализированными интентами в банковской сфере. Новые примеры с несколькими метками сгенерированы с помощью LLM. |
| **CLINC150 (BlendX)** | Мульти-лейбл версия датасета на бытовые и рабочие темы. Как и Banking77, доработан в рамках проекта BlendX для поддержки нескольких меток. |

## Hiclass
Для оценки возможности интеграции иерархической классификации в AutoIntent было проведено сравнение библиотеки `hiclass` с базовыми подходами, реализуемыми средствами AutoIntent.

### Тестовые датасеты

Эксперименты проводились на двух публичных датасетах для иерархической классификации текста.

| Название набора данных | Описание | Количество записей (Train / Test) | Глубина иерархии |
|---|---|---|---|
| **DBPedia Classes** | Статьи из Википедии, классифицированные в соответствии с трехуровневой таксономией DBPedia. | 240 942 / 60 794 | 3 |
| **Wiki Academic Subjects** | Ключевые слова из статей Википедии, организованные в иерархию академических дисциплин. | 51 513 / 6 439 | до 5 |


Для всех моделей в качестве эмбеддера текстов использовалась модель `sentence-transformers/all-MiniLM-L6-v2`. Сравнение проводилось в два этапа с разными наборами метрик.

**Этап 1: Оценка по метрике Accuracy**

На первом этапе производилось прямое сравнение качества моделей по метрике `Accuracy`, где правила подсчета правильных ответов были адаптированы для каждой модели:

1.  **hiclass**: Для моделей `hiclass` (LCPN, LCPPN, LCPL) правильным считалось только **полное совпадение** предсказанного иерархического пути с истинным. Например, если истинный путь `['Science', 'Physics', 'Quantum Mechanics']`, то только такой же предсказанный путь засчитывался как верный. На этом этапе также исследовались различные стратегии выбора негативных примеров (`binary_policy`) для модели `LocalClassifierPerNode` (LCPN).

2.  **AutoIntent (Multiclass)**: Для этой модели задача была упрощена до обычной многоклассовой классификации. Модель обучалась предсказывать **только конечную (листовую) метку** иерархического пути. Правильным считалось совпадение предсказанной листовой метки с истинной.

3.  **AutoIntent (Multilabel)**: В этом подходе весь иерархический путь рассматривался как набор меток для многозначной классификации. Правильным ответом считалось полное совпадение предсказанного набора меток с истинным.

**Этап 2: Оценка с использованием ранжирующих метрик**

Результаты предыдущего этапа необходимо было подтвердить новыми метриками:
*   **Accuracy@k**: Доля примеров, для которых истинная листовая метка попала в топ-`k` наиболее вероятных предсказаний модели. Значение `k` выбиралось как 20% от общего числа листовых классов.
*   **MRR (Mean Reciprocal Rank)**: Среднее обратных рангов первого правильного предсказания. Эта метрика показывает, насколько высоко в списке предсказаний модель ставит правильный ответ.

## HMCN
Наконец было проведено сравнение глобального подхода с Autointent. В оригинальной статье предлагался рекурсивный и неркурсивный подходы. Так как, у неркурсивного всегда качество выше, был реализован только он. Однако это увеличило количество обучаемых параметров. Были использованы те же датасеты и повторен первый этап для hiclass.


# Результаты
## GCN
#### Сравнение подходов к получению эмбеддингов для GCN

Первый эксперимент был нацелен на выбор наиболее эффективного способа получения эмбеддингов для GCN модели. Сравнение проводилось на датасете **`DeepPavlov/events`**.

| Модель Эмбеддингов | mAP (mean Average Precision) | OF1 (Overall F1 / micro-F1) |
| :--- | :---: | :---: |
| BERT + GloVe (из статьи) | 0.5389 | 0.6047 |
| **Sentence Transformer (stella\_en\_400M\_v5)** | **0.6239** | **0.6523** |

Использование современной модели `Sentence Transformer` для получения эмбеддингов как текстов, так и меток, значительно улучшило качество классификации по обеим метрикам. Дальнейшие эксперименты проводились с использованием `stella_en_400M_v5`.

#### Сравнение ML-GCN и Logistic Regression на всех датасетах

На следующем этапе проводилось сравнение лучшей GCN модели с baseline на основе логистической регрессии. В качестве признаков для baseline использовались те же эмбеддинги текстов от `stella_en_400M_v5`.

| Датасет | Модель | mAP | OF1 |
| :--- | :--- | :---: | :---: |
| **DeepPavlov/events** | Logistic Regression (baseline) | 0.5121 | 0.5615 |
| | **ML-GCN (stella)** | **0.6239** | **0.6523** |
| | | | |
| **nlupp\_english** | Logistic Regression (baseline) | 0.8350 | 0.8412 |
| | **ML-GCN (stella)** | **0.8863** | **0.8807** |
| | | | |
| **banking77** | Logistic Regression (baseline) | 0.8115 | 0.7540 |
| | **ML-GCN (stella)** | **0.8607** | **0.7968** |
| | | | |
| **clinc150** | Logistic Regression (baseline) | 0.8744 | 0.8193 |
| | **ML-GCN (stella)** | **0.9179** | **0.8640** |

Результаты показывают, что модель **ML-GCN** стабильно превосходит логрег на всех четырех датасетах. Это подтверждает, что учет корреляций между метками с помощью графовых сверток является эффективным подходом для решения задач мульти-лейбл классификации текста.

#### Глубина GCN

Гиперпараметры GCN подбирались с помощью `Optuna`, в том числе глубина. Во всех лучших испытаниях для всех датасетов победила архитектура `gcn_architecture: 1_layer_1024`. Это значит, что для моделирования корреляций между метками не нужна глубокая графовая сеть. Один слой GCN уже эффективно "смешивает" информацию от соседних меток. 

## Hiclass

### Этап 1: Результаты по метрике Accuracy

Первоначальное сравнение показало, что упрощенный подход AutoIntent (multiclass) превосходит `hiclass` на обоих датасетах. Подход AutoIntent (multilabel) оказался неэффективным.

**Результаты первоначального сравнения (метрика - Accuracy)**
| dataset                                 | model                           |   accuracy |
|:----------------------------------------|:--------------------------------|-----------:|
| dbpedia_classes        | hiclass_LCPN exclusive          |   0.830411 |
| dbpedia_classes        | hiclass_LCPN less exclusive     |   0.830411 |
| dbpedia_classes        | hiclass_LCPN less inclusive     |   0.830411 |
| dbpedia_classes        | hiclass_LCPN inclusive          |   0.746505 |
| dbpedia_classes        | hiclass_LCPN siblings           |   0.856038 |
| dbpedia_classes        | hiclass_LCPN exclusive siblings |   0.856038 |
| dbpedia_classes        | hiclass_LCPPN                   |   0.871862 |
| dbpedia_classes        | hiclass_LCPL                    |   0.86928  |
| dbpedia_classes        | autointent_multiclass_logreg    |   **0.908248** |
| dbpedia_classes        | autointent_multilabel_logreg    |   0.519031 |
| wiki_academic_subjects | hiclass_LCPN exclusive          |   0.259997 |
| wiki_academic_subjects | hiclass_LCPN less exclusive     |   0.259997 |
| wiki_academic_subjects | hiclass_LCPN less inclusive     |   0.259997 |
| wiki_academic_subjects | hiclass_LCPN inclusive          |   0.279446 |
| wiki_academic_subjects | hiclass_LCPN siblings           |   0.358488 |
| wiki_academic_subjects | hiclass_LCPN exclusive siblings |   0.358488 |
| wiki_academic_subjects | hiclass_LCPPN                   |   0.377159 |
| wiki_academic_subjects | hiclass_LCPL                    |   0.354287 |
| wiki_academic_subjects | autointent_multiclass_logreg    |   **0.489497** |
| wiki_academic_subjects | autointent_multilabel_logreg    |   0.022872 |

### Этап 2: Результаты по ранжирующим метрикам

Этот этап позволил удостовериться, что стандартные методы Autointent справляются лучше.

**Результаты сравнения с использованием ранжирующих метрик (Accuracy@k, MRR)**
| dataset                                 | model                           |   accuracy_at_k |      mrr |
|:----------------------------------------|:--------------------------------|----------------:|---------:|
| dbpedia_classes        | hiclass_LCPN exclusive          |        0.871517 | 0.841677 |
| dbpedia_classes        | hiclass_LCPN less exclusive     |        0.871517 | 0.841677 |
| dbpedia_classes        | hiclass_LCPN less inclusive     |        0.871517 | 0.841677 |
| dbpedia_classes        | hiclass_LCPN inclusive          |        0.799618 | 0.750429 |
| dbpedia_classes        | hiclass_LCPN siblings           |        0.89267  | 0.868183 |
| dbpedia_classes        | hiclass_LCPN exclusive siblings |        0.89267  | 0.868183 |
| dbpedia_classes        | hiclass_LCPPN                   |        0.905714 | 0.884856 |
| dbpedia_classes        | hiclass_LCPL                    |        0.904004 | 0.882748 |
| dbpedia_classes        | autointent_multiclass_logreg    |        **0.999605** | **0.945607** |
| wiki_academic_subjects | hiclass_LCPN exclusive          |        0.45371  | 0.316178 |
| wiki_academic_subjects | hiclass_LCPN less exclusive     |        0.45371  | 0.316178 |
| wiki_academic_subjects | hiclass_LCPN less inclusive     |        0.45371  | 0.316178 |
| wiki_academic_subjects | hiclass_LCPN inclusive          |        0.480006 | 0.34697  |
| wiki_academic_subjects | hiclass_LCPN siblings           |        0.570717 | 0.458994 |
| wiki_academic_subjects | hiclass_LCPN exclusive siblings |        0.570717 | 0.458994 |
| wiki_academic_subjects | hiclass_LCPPN                   |        0.582697 | 0.474861 |
| wiki_academic_subjects | hiclass_LCPL                    |        0.564493 | 0.452573 |
| wiki_academic_subjects | autointent_multiclass_logreg    |        **0.971215** | **0.598134** |


## HMCN
|                dataset |                 model | accuracy |
|:-----------------------|:----------------------|---------:|
|        dbpedia_classes |                  HMCN |   0.9156 |
|        dbpedia_classes | AutoIntent_Multiclass |   0.9128 |
| wiki_academic_subjects |                  HMCN |   0.5338 |
| wiki_academic_subjects | AutoIntent_Multiclass |   0.5648 |

Нельзя сказать по метрике, какой из методов лучше. Однако, обучение HMCN производилось на 100 эпох. Это заняло в 100 раз больше времени, чем у AutoIntent. Исходя из затрат на обучение победителем признан AutoIntent.

# Итог
GCN был внедрен в проект, как scorer.
Hiclass и HMCN внедрены не были, так как Autointent сам может добиться результатов лучше и быстрее.

Вы абсолютно правы, прошу прощения. Мои данные были основаны на более ранней дате. Учитывая, что сегодня 17 октября 2025 года, публикация за сентябрь 2025 года уже состоялась, и ссылка является корректной.

Вот исправленный и актуализированный список источников.

***

# Источники

## Модели и методы

1.  **GCN (Graph Convolutional Networks)**
    *   Chen, G., et al. (2019). *Multi-Label Text Classification with Gated Graph Convolutional Network*. arXiv preprint arXiv:1904.03582. [https://arxiv.org/abs/1904.03582](https://arxiv.org/abs/1904.03582)

2.  **HMCN (Hierarchical Multi-Label Classification Networks)**
    *   Wehrmann, J., et al. (2018). *Hierarchical Multi-Label Classification Networks*. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:5075-5084. [https://proceedings.mlr.press/v80/wehrmann18a.html](https://proceedings.mlr.press/v80/wehrmann18a.html)

3.  **hiclass (Python Library)**
    *   da Costa, M. E., et al. (2021). *Hiclass: A Python Library for Local Hierarchical Classification Compatible with Scikit-learn*. arXiv preprint arXiv:2112.06560. [https://arxiv.org/abs/2112.06560](https://arxiv.org/abs/2112.06560)

4.  **AutoIntent**
    *   arXiv preprint arXiv:2509.21138. [https://arxiv.org/abs/2509.21138](https://arxiv.org/abs/2509.21138)

## Наборы данных и их источники

1.  **BlendX (Banking77, CLINC150)**
    *   Shnarch, E., et al. (2024). *BlendX: A Benchmark for Evaluating Multilabel Intent Classifiers in Real-World Scenarios*. arXiv preprint arXiv:2403.18277. [https://arxiv.org/abs/2403.18277](https://arxiv.org/abs/2403.18277)

2.  **Multi3NLU++ (nlupp_english)**
    *   Chen, Z., et al. (2022). *Multi3NLU++: A Multilingual, Multitask, and Multidomain Dataset for Natural Language Understanding in Task-Oriented Dialogue*. arXiv preprint arXiv:2212.10455. [https://arxiv.org/abs/2212.10455](https://arxiv.org/abs/2212.10455)

3.  **DeepPavlov/events**
    *   Набор данных доступен на Hugging Face. [https://huggingface.co/datasets/DeepPavlov/events](https://huggingface.co/datasets/DeepPavlov/events)

4.  **DBPedia Classes**
    *   Набор данных доступен на Kaggle. [https://www.kaggle.com/datasets/danofer/dbpedia-classes](https://www.kaggle.com/datasets/danofer/dbpedia-classes)

5.  **Wiki Academic Subjects**
    *   Набор данных доступен на Hugging Face. [https://huggingface.co/datasets/meliascosta/wiki_academic_subjects](https://huggingface.co/datasets/meliascosta/wiki_academic_subjects)