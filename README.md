# Решение задания Avito Bootcamp: поиск объявлений

Я подготовил итоговый `answer.csv` для задания «Кандидатогенерация для категории услуг». Мой лучший подтверждённый результат на Stepik — **Recall@50 = 0,835023**. В файле 2 452 запроса и по 50 уникальных объявлений на каждый запрос. После проверки последних экспериментов я сохранил этот ответ: устойчивого улучшения на двух отложенных выборках не получил.

В итоговый архив я включил готовый ответ, код с комментариями, пять локальных моделей, отчёты и инструкцию запуска. Исходные данные задания нужно получить отдельно. После распаковки структура проекта выглядит так:

```text
avito_bootcamp_solution/
  answer.csv
  README.md
  PROGRESS.md
  requirements.txt
  models/                 # четыре CatBoost-модели и LightGBM
  reports/                # сохранённые метрики и проверка воспроизведения
  retrieval.py
  solution.py
  improved_predict.py
  prototype_predict.py
  prototype_experiment.py
  ranker_experiment.py
  package_solution.py
```

## Как повторить предсказание

Я проверил зависимости и загрузку моделей в изолированной среде **Python 3.11.16**. Все установленные версии совпали с `requirements.txt`; `pip check` ошибок не нашёл. Из корня распакованного проекта я создаю среду:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Для Windows среда активируется командой `.venv\Scripts\activate`. На macOS LightGBM использует системную библиотеку `libomp`; при ошибке её загрузки я устанавливаю её через `brew install libomp`.

Я помещаю три исходных файла в папку `dataset/`:

```text
dataset/train.parquet
dataset/benchmark_queries.parquet
dataset/benchmark_items.parquet
```

Затем я запускаю две команды. Сначала вычисляю оценки исходного пула кандидатов, затем добавляю кандидатов по заголовкам ранее выбранных объявлений:

```bash
python improved_predict.py --data-dir dataset --output answer_ranker.csv --candidate-cap 600 --ranker-weight 0.4 --view-boost 0
python prototype_predict.py --data-dir dataset --output answer_reproduced.csv --limit 500 --ranker-weight 0.4 --view-boost 0
```

Я записываю повторный результат в `answer_reproduced.csv`, чтобы сохранить приложенный лучший ответ. При первом запуске код сам создаёт `artifacts_final/`, строит текстовый индекс и сохраняет оценки кандидатов. Индекс занимает около 1,2 ГБ; для вычислений нужны несколько ГБ оперативной памяти. Данные и большие кэши в ZIP не включены. Готовые модели уже лежат в `models/`, переобучение для этих команд не требуется. После установки зависимостей все вычисления идут локально, без внешних API и скачивания моделей.

Для повторного запуска с теми же данными и моделями я использую сохранённые оценки первой команды:

```bash
python improved_predict.py --data-dir dataset --reuse-cache --output answer_ranker.csv --ranker-weight 0.4 --view-boost 0
python prototype_predict.py --data-dir dataset --output answer_reproduced.csv --limit 500 --ranker-weight 0.4 --view-boost 0
```

Кэши относятся к конкретным данным и весам. После замены датасета или моделей я удаляю `artifacts_final/` и запускаю первые две команды заново.

## Как проверить ответ

Я проверяю названия колонок, состав запросов, число кандидатов, отсутствие повторов и точное совпадение каждого `item_id` с корпусом:

```bash
python -c "import pandas as pd; from solution import validate_answer; q=pd.read_parquet('dataset/benchmark_queries.parquet'); i=pd.read_parquet('dataset/benchmark_items.parquet', columns=['item_id']); print(validate_answer('answer.csv', q, i))"
```

Для итогового файла проверка возвращает `rows=2452`, `candidates_per_query={50: 2452}` и SHA-256:

```text
fb293d0ca73313fffcf0f4b5710c86bddf48021703e9dbeec7379a55a6df8c7f
```

Я сравниваю повторный результат с приложенным ответом так:

```bash
python -c "from pathlib import Path; import hashlib; a=Path('answer.csv').read_bytes(); b=Path('answer_reproduced.csv').read_bytes(); assert a == b, 'Файлы различаются'; print(hashlib.sha256(b).hexdigest())"
```

## Как устроено решение

Я построил словный TF–IDF-индекс заголовков, описаний и параметров объявлений с русским стеммингом, а также символьный индекс заголовков. Дополнительные каналы используют историю выборов по похожим запросам и предсказанную микрокатегорию. География влияет на оценку плавно: некоторые услуги ищут по области или получают дистанционно.

Для ранжирования я использую четыре CatBoost-модели и LightGBM LambdaRank. Сначала усредняю вероятности CatBoost и перевожу их в log-odds. Затем стандартизирую оценки обоих методов внутри каждого запроса и смешиваю их: 60 % CatBoost, 40 % LightGBM. Из полученных кандидатов выбираю 50 объявлений.

В итоговой версии я также ищу по усреднённым заголовкам объявлений, которые выбирали для похожих обучающих запросов. Беру до 500 кандидатов из этого канала и до 500 после географического взвешивания, убираю повторы и оцениваю новые объявления теми же моделями. Этот канал повысил подтверждённый результат Stepik с 0,834286 до 0,835023.

| Файл | Назначение |
|---|---|
| `retrieval.py` | Нормализация текста, индексы, каналы кандидатов и признаки |
| `solution.py` | Разбиение обучающих данных, обучение CatBoost и проверка CSV |
| `ranker_experiment.py` | Обучение LightGBM на кандидатах обучающей части |
| `improved_predict.py` | Оценка базовых кандидатов четырьмя CatBoost-моделями и LightGBM |
| `prototype_experiment.py` | Поиск по заголовкам ранее выбранных объявлений |
| `prototype_predict.py` | Добавление этих кандидатов и запись итогового ответа |
| `package_solution.py` | Проверка CSV, сборка ZIP и проверка контрольных сумм |

Я использовал только предоставленные признаки и обучающие взаимодействия. `query_id` нужен для записи строк CSV и проверки соответствия кэша; в признаки модели он не входит. Скрытую разметку и ручной подбор ответов я не использовал. В решении применены открытые библиотеки NumPy, pandas, PyArrow, SciPy, scikit-learn, NLTK Snowball, CatBoost, LightGBM и joblib; версии перечислены в `requirements.txt`.

## Проверка качества и последние эксперименты

Я проверял базовую смесь на двух отложенных разбиениях с корпусами размера бенчмарка. Исключал запросы, чьи тексты встречались в обучении сравниваемых моделей. На 942 и 931 оставшемся запросе Recall@50 составил **0,87606 / 0,88641**, потолок пула кандидатов — **0,94851 / 0,95560**. Эти локальные результаты относятся к базовой смеси; подтверждённый балл итогового CSV получен отдельно на Stepik.

| Гипотеза | Recall@50, v10 / v9 | Мой вывод |
|---|---|---|
| Редкие слова в заголовках | 0,87606 / 0,88641 | Прироста нет |
| Редкие слова в описаниях и параметрах | 0,87818 / 0,88641 | Прирост только на одном разбиении |
| Популярные объявления предсказанной микрокатегории | 0,87606 / 0,88748 | Прирост только на одном разбиении |
| Отдельный символьный поиск по заголовкам | 0,87606 / 0,88748 | Прирост только на одном разбиении |
| RRF вместо смеси оценок | 0,86438 / 0,88122 | Качество снизилось |

Я также проверял семантическую модель на меньшей выборке, квоты на лексические кандидаты и другие веса смеси. Устойчивого выигрыша не получил. Эти варианты в итоговое предсказание не включил; сохранённые численные отчёты лежат в `reports/`. Для новых каналов доверительные интервалы прироста касались нуля. В бенчмарке редкие слова встречаются чаще, поэтому локальная оценка не гарантирует такой же балл на Stepik.

## Как повторить обучение

Для предсказания я использую приложенные веса. При необходимости повторить обучение с исходных Parquet-файлов я запускаю:

```bash
python solution.py --data-dir dataset --output-dir artifacts_v5 --corpus-style location --split-style benchmark_profile --skip-predict
python solution.py --data-dir dataset --output-dir artifacts_v6 --corpus-style location --split-style benchmark_profile --seen-style frequency --skip-predict
python solution.py --data-dir dataset --output-dir artifacts_v9 --corpus-style location --split-style benchmark_profile --seen-style frequency --microcat-channel --skip-predict
python solution.py --data-dir dataset --output-dir artifacts_v10 --corpus-style location --split-style benchmark_profile --microcat-channel --skip-predict
python ranker_experiment.py --source . --data dataset --output artifacts_ranker
```

Затем я копирую `artifacts_v5/ranker.cbm`, `artifacts_v6/ranker.cbm`, `artifacts_v9/ranker.cbm`, `artifacts_v10/ranker.cbm` в `models/` под именами `catboost_v5.cbm`, `catboost_v6.cbm`, `catboost_v9.cbm`, `catboost_v10.cbm`, а `artifacts_ranker/lambdarank.txt` — в `models/lambdarank.txt`. После замены весов я пересоздаю кэши предсказания. Полное обучение занимает существенно больше времени, чем запуск готовых моделей.

## Как собрать архив для GitHub

Я собираю отдельную версию командой:

```bash
python package_solution.py --data-dir dataset --tag final_recall_0_835023
```

Она сохраняет `dist/avito_bootcamp_solution_final_recall_0_835023.zip`. Перед упаковкой код проверяет CSV, а после — целостность ZIP и SHA-256 каждого включённого файла. Внутри есть `manifest.json` с размерами и контрольными суммами. Для GitHub я распаковываю ZIP и добавляю его содержимое в репозиторий; модели и ответ включены, локальные данные и кэши исключены через `.gitignore`.
