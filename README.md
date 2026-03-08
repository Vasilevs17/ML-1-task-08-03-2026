# ML Project Bootstrap (Tabular Data)

Репозиторий подготовлен для ML-задач на табличных данных с внешним хранением raw-данных.

## Что уже настроено

- Автоматический setup среды и данных через `scripts/setup.sh`.
- Загрузка архива данных по секрету `GDRIVE_URL`.
- Распаковка в `data/raw/` с проверкой успешности.
- Базовая инфраструктура проекта:
  - `src/data` — загрузка и валидация,
  - `src/features` — подготовка признаков,
  - `src/models` — обучение baseline и инференс,
  - `configs` — конфиги,
  - `tests` — smoke-тесты.
- Исключения в `.gitignore` для данных/артефактов/моделей/логов.

## Быстрый старт

1. Установите переменную окружения `GDRIVE_URL` (значение секрета не логируйте).
2. Выполните setup:

```bash
bash scripts/setup.sh
```

Скрипт:
- установит зависимости из `requirements.txt`,
- скачает архив,
- распакует его в `data/raw/`,
- проверит, что файлы действительно появились.

## Анализ данных

После загрузки данных выполните:

```bash
python -m src.data.report --raw-dir data/raw --output artifacts/data_report.json
```

Отчёт содержит по каждому CSV/Parquet:
- имя и формат файла,
- число строк и столбцов,
- оценку потребления памяти,
- долю пропусков,
- кандидаты на target-колонку.

## Обучение baseline

1. Укажите `target_col` в `configs/train.yaml`.
2. Запустите:

```bash
python -m src.models.train --config configs/train.yaml
```

Результаты:
- модель: `models/baseline_model.joblib`,
- метрики: `artifacts/metrics.json`.

## Предсказание

```bash
python -m src.models.predict --model models/baseline_model.joblib --input data/raw/<file>.csv --output artifacts/predictions.csv
```

## Тесты

```bash
pytest -q
```

## Важно

- Не коммитьте содержимое `data/raw/`, архивы и артефакты (уже закрыто через `.gitignore`).
- Если формат архива нестандартный, обновите `scripts/setup_data.sh` и зафиксируйте изменения в README.
