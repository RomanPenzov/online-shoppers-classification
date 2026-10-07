# Online Shoppers Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![uv](https://img.shields.io/badge/dependencies-uv-purple)
![Ruff](https://img.shields.io/badge/style-Ruff-red)

Самостоятельный ML-проект на основе частей I-III. Все вычисления вынесены в
`src/online_shoppers/`. Ноутбук `notebooks/10_project.ipynb` - тонкий интерфейс;
`notebooks/archive/` хранит три нутбука начальных исследований.
Архивные ноутбуки исключены из Ruff: это исторические материалы, а не модули приложения.
Активный ноутбук и весь Python-код проходят Ruff.

## Установка на Windows / VS Code

Распакуйте проект, например, в `E:\Ian\online-shoppers-classification` и откройте
эту папку в VS Code. Для проекта нужно отдельное окружение. В терминале PowerShell:

```powershell
cd E:\Ian\online-shoppers-classification
py -3.12 -m uv sync --locked
py -3.12 -m uv run ruff check .
py -3.12 -m uv run ruff format --check .
py -3.12 -m uv run ty check
py -3.12 -m uv run pytest
```

Если `uv` ещё не установлен: `py -3.12 -m pip install --user uv`.
Если команда `uv` доступна напрямую, префикс `py -3.12 -m uv` можно заменить на `uv`.
`uv sync` создаёт `.venv` и устанавливает зависимости проекта и dev-инструменты.
Активация окружения для `uv run` не требуется. В VS Code выберите интерпретатор
`.venv\Scripts\python.exe`. Предупреждение uv о hardlink решается добавлением
`--link-mode=copy` к `uv sync`, если кэш и проект находятся на разных дисках.

На Linux/macOS используются те же команды с `uv` вместо `py -3.12 -m uv`.
Зафиксирован Python 3.12; конкретный patch-релиз и платформа записываются в отчёт.
`uv.lock` фиксирует транзитивные зависимости. Основные ML-библиотеки в pyproject
совпадают с версиями из переданного выполненного ноутбука III.

## Запуск

Полный эксперимент с официальным скачиванием при отсутствии CSV:

```powershell
py -3.12 -m uv run online-shoppers train --download --output artifacts/run_1
```

Или с уже скачанным файлом (сеть не используется):

```powershell
py -3.12 -m uv run online-shoppers train --data "E:\Ian\classic_ml\data\online_shoppers_intention\online_shoppers_intention.csv" --output artifacts/run_1
```

Путь по умолчанию: `data/online_shoppers_intention.csv`. `--download` не обновляет
существующий файл. Ожидается именно официальный CSV: другой SHA-256 останавливает
эксперимент, чтобы незаметно не заменить протокол I–III. Если CSV пересохранён с
другими окончаниями строк, скачайте исходные байты заново в новый путь.

Каталог результатов должен быть новым или пустым. Повторный запуск:

```powershell
py -3.12 -m uv run online-shoppers train --output artifacts/run_2
py -3.12 -m uv run online-shoppers verify --first artifacts/run_1 --second artifacts/run_2
```

`verify` сравнивает хеши, выбор, параметры, порог, CV-таблицу, OOF и итоговые
test-прогнозы/метрики; время исполнения исключается, допуск: rtol=1e-10, atol=1e-12.
Повторный запуск - проверка воспроизводимости уже утверждённого протокола,
а не основание менять параметры по результатам test.

Инференс:

```powershell
py -3.12 -m uv run online-shoppers predict --model artifacts/run_1/model.joblib --data data/new_sessions.csv --output artifacts/new_predictions.csv
```

CSV для инференса должен содержать 16 разрешённых признаков; `Revenue` и
`PageValues` не нужны. При их наличии pipeline игнорирует их, как и неизвестные
дополнительные поля. Выход сохраняет порядок строк: `score`, `prediction`.
Для SVM score - `decision_function`, не вероятность; порог загружается вместе с моделью.
Файлы joblib загружайте только из доверенного источника: их загрузка исполняет Python.

Для Jupyter:

```powershell
py -3.12 -m uv sync --locked --group notebooks
py -3.12 -m uv run --group notebooks python -m ipykernel install --user --name online-shoppers --display-name "Python (online-shoppers)"
```

Откройте `notebooks/10_project.ipynb` и выберите это ядро.

## Данные и лицензия

[Online Shoppers Purchasing Intention, UCI, id=468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
Авторы: C. Sakar и Y. Kastro, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
Лицензия датасета: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Атрибуция сохранена здесь; лицензия данных не назначает автоматически лицензию коду.
Исходный датасет: 12 330 сессий, 17 признаков, 1 908 положительных Revenue.
CSV скачивается только с официального UCI URL, указанного в `config.py`.
Исходный CSV не входит в Git или поставляемый архив.

## Feature policy и временная утечка

**PageValues запрещён для real-time предсказания**: доступность на момент решения
не подтверждена, признак связан с ценностью страниц и транзакциями и создаёт риск
target leakage. FeaturePolicy - первый шаг каждого pipeline на fit и inference;
allowlist не допускает PageValues, Revenue и произвольные новые колонки к preprocessing.

| Разрешённые признаки | Условие использования в real-time |
|---|---|
| Administrative, Informational, ProductRelated и три Duration | Только накопленное до момента скоринга |
| BounceRates, ExitRates | Только доступные до скоринга исторические агрегаты |
| SpecialDay, Month, Weekend | Известные календарные сведения |
| OperatingSystems, Browser, Region, TrafficType, VisitorType | Сведения, установленные к моменту запроса |

**Исключение PageValues не доказывает полную временную безопасность CSV.**
UCI содержит итоговые сессии, а не снимки состояния на момент решения.
Для production нужны событийные данные, point-in-time сборка признаков и свежий
временной test. В этом проекте доказаны программные инварианты feature policy,
изоляция preprocessing внутри CV и порядок выбора до test текущего запуска.

Схема CSV строгая: 18 уникальных колонок, бинарный target без пропусков, числовые
признаки неотрицательны, доли лежат в [0,1], категориальные коды целые.
Пропуски в признаках допустимы и заполняются только внутри fit-фолда.

## Протокол эксперимента

1. Проверить схему и SHA-256 исходного CSV.
2. Удалить полные дубликаты с сохранением первого: остаётся 12 205 строк.
   Если разрешённые X всё ещё повторяются, остановиться: нужен отдельный group-split.
3. Stratified train/test = 80/20, seed=42: train=9 764, test=2 441.
4. Создать общие `StratifiedKFold(5, shuffle=True, random_state=42)` на train.
5. Сравнить все кандидаты и их фиксированные сетки из `models.py`. Imputer,
   scaling, категории и границы интервалов обучаются отдельно внутри каждого fold.
6. Выбрать модель по среднему CV Average Precision; при равенстве - по имени.
   Dummy служит контрольной линией и не выбирается финальной моделью.
7. Получить train OOF scores выбранного pipeline, выбрать максимальный F1;
   при равных F1 брать наибольший порог. Повторить OOF и refit и сравнить scores.
8. Сохранить `selection_before_test.json`, затем оценить **только победителя** на test.

PR-AUC здесь означает `average_precision_score` (AP), а не трапецеидальную площадь
под PR-кривой. Также считаются ROC AUC, precision, recall, F1, confusion matrix.
В CV precision/recall/F1 измеряются при штатном пороге модели; итоговая test-оценка
использует выбранный train OOF порог. Не сравнивайте эти F1 как одинаковые режимы.

CV используется и для настройки, и для выбора: это development-оценка, не nested CV.
OOF после подбора параметров на всём train также не является независимой оценкой
качества выбора. Test уже просматривался в частях I-II: это прежний holdout,
а не новый независимый тест. Разница средних CV не доказывает значимость превосходства;
std по folds не является доверительным интервалом.

## Модели

LogisticRegression, DecisionTree, RandomForest, GradientBoosting, XGBoost,
LightGBM, CatBoost, SVC(linear), SVC(rbf), BernoulliNB и контрольный Dummy.
Компактные сетки перенесены из части III; это не повтор 16 случайных конфигураций
части II. Все бустинги, включая CatBoost, используют общий one-hot preprocessing.
Это сравнение конкретных pipeline при заданном бюджете, а не предельного качества библиотек.

Числа для LR/SVM стандартизируются. Для деревьев scaling не нужен.
Для BernoulliNB числа разбиваются на квантильные интервалы и кодируются one-hot;
категории также one-hot. Получаются бинарные индикаторы, подходящие Bernoulli-модели.
Взаимоисключающие категории и связь количества страниц с длительностью нарушают
условную независимость признаков: NB - приближённый быстрый baseline;
дискретизация теряет информацию, вероятности могут быть плохо откалиброваны.
Интервалы обучаются только на fit-фолде. Предупреждения об объединении узких
интервалов на признаках с множеством нулей ожидаемы. SVM не калибруется,
так как AP/ROC AUC достаточно ранжирующего score.

## Результаты

Результаты исходного выполненного ноутбука III, **не заявление о новом запуске**:

| Модель | Mean CV AP |
|---|---:|
| RandomForest | 0.383186 |
| LightGBM | 0.378976 |
| XGBoost | 0.376382 |
| CatBoost | 0.373541 |
| GradientBoosting | 0.371543 |
| LogisticRegression | 0.327799 |
| DecisionTree | 0.325228 |
| BernoulliNB | 0.317447 |
| SVM RBF | 0.300687 |
| SVM linear | 0.259047 |
| Dummy | 0.156288 |

В ноутбуке выбран RandomForest: holdout AP=0.380490, ROC AUC=0.785015,
F1=0.442638, precision=0.337931, recall=0.641361, порог=0.21663162596825716.
Подлинные текстовые выводы: `docs/notebook_reference_results.txt`.
Самостоятельный проект также полностью выполнен: выбран тот же RandomForest,
показанные метрики и порог совпали. Статус проверок и точные значения:
`docs/VALIDATION.md` и `docs/verified_results/`.

Модель склонности к покупке не оценивает эффект скидки. Для решения о вмешательстве
нужны стоимость ошибок, A/B-тест или uplift-постановка; порог F1 не максимизирует прибыль.

## Структура

| Путь | Назначение |
|---|---|
| src/online_shoppers/config.py | Константы, schema/allowlist, seed, SHA-256 |
| data.py | Загрузка и валидация |
| features.py | FeaturePolicy на fit и inference |
| models.py | Preprocessing, модели, фиксированные сетки |
| training.py | Общий GridSearchCV, таблица сравнения |
| evaluation.py | Scores, метрики, OOF, порог |
| plots.py | CV-график и финальные PR/ROC/confusion matrix |
| experiment.py | Последовательность эксперимента и журнал воспроизводимости |
| inference.py, reproducibility.py, cli.py | Инференс, сравнение запусков, CLI |
| tests/ | Offline-тесты, включая повтор полного цикла на фиктивных данных |
| notebooks/ | Активный интерфейс и исторические оригиналы |
| docs/ | Доказательства проверки и рекомендации по Git |
| artifacts/ | Игнорируемые результаты запусков |
| data/ | Игнорируемые входные CSV |

Основные артефакты: `cv_comparison.csv`, `search_<model>.csv`, `split_manifest.json`,
`selection_before_test.json`, `selected_oof.csv`, `model.joblib`,
`selected_test_metrics.csv`, `selected_test_predictions.csv` и два PNG.
В журнале фиксируются версии, SHA-256 CSV/split и детерминированная сигнатура
таблицы без времени. Незначительные отличия между ОС/CPU/BLAS возможны;
никакой универсальной битовой идентичности на разных платформах не заявляется.

## Тесты, CI и Docker (в плане)

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Тесты не скачивают данные; socket-подключения в них заблокированы. Проверяются
схема, бинарный target, forbidden columns на fit/inference, статистики imputer,
все семейства моделей, бинарный вход NB, новые категории, выбор порога,
повторяемость split/CV/OOF, сохранение и повторное применение модели.

#### (в плане)
GitHub Actions содержит проверки на Windows и Linux и отдельную сборку Docker.
Workflow начнёт работать после загрузки репозитория на GitHub. CI-бейдж с реальным
URL добавляется после выбора владельца/репозитория; фиктивного зелёного бейджа нет.

Docker включён по шаблону 02 как необязательный способ запуска:

```bash
docker build -t online-shoppers:local .
docker run --rm online-shoppers:local --help
docker compose up --build
```

Compose монтирует data/artifacts и запускает обучение в `artifacts/docker_run`;
для повторного запуска укажите новый каталог. Локальные команды uv Docker не требуют.