# Анализ `data/raw/`

## 1) Краткий обзор файлов

### pretest.parquet
- Строк: **14,202,075**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **155,629,206** (47.64% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: pos_cd: 13,297,710, browser_language: 13,022,905, battery: 13,022,905, timezone: 13,012,072, operating_system_type: 13,012,072
- Предполагаемая роль: Дополнительный тест/валидационный поток без target.

### pretrain_part_1.parquet
- Строк: **30,329,960**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **419,127,028** (60.08% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: browser_language: 30,329,960, session_id: 30,329,960, operating_system_type: 30,329,960, battery: 30,329,960, device_system_version: 30,329,960
- Предполагаемая роль: Дополнительные (возможно предобучающие) события без target, часть 1.

### pretrain_part_2.parquet
- Строк: **30,237,860**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **417,727,735** (60.06% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: browser_language: 30,237,860, session_id: 30,237,860, operating_system_type: 30,237,860, battery: 30,237,860, device_system_version: 30,237,860
- Предполагаемая роль: Дополнительные (возможно предобучающие) события без target, часть 2.

### pretrain_part_3.parquet
- Строк: **30,372,137**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **419,538,307** (60.06% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: browser_language: 30,372,137, session_id: 30,372,137, operating_system_type: 30,372,137, battery: 30,372,137, device_system_version: 30,372,137
- Предполагаемая роль: Дополнительные (возможно предобучающие) события без target, часть 3.

### test.parquet
- Строк: **633,683**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **6,953,212** (47.71% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: pos_cd: 591,478, browser_language: 579,776, battery: 579,776, timezone: 579,129, operating_system_type: 579,129
- Предполагаемая роль: Тестовый набор событий для финального инференса.

### train_labels.parquet
- Строк: **87,514**, столбцов: **3**
- Столбцы: `customer_id, event_id, target`
- Типы признаков: числовые=3, строковые/категориальные=0
- Пропуски: суммарно **0** (0.00% ячеек); столбцов с пропусками: 0
- Наибольшие пропуски: нет
- Предполагаемая роль: Таблица разметки (target) для части событий train.

### train_part_1.parquet
- Строк: **28,618,594**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **333,860,206** (50.72% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: pos_cd: 26,134,680, browser_language: 25,968,210, battery: 25,968,210, operating_system_type: 25,917,436, timezone: 25,867,893
- Предполагаемая роль: Основной обучающий поток событий (часть 1).

### train_part_2.parquet
- Строк: **28,558,397**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **333,366,892** (50.75% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: pos_cd: 26,040,592, browser_language: 25,999,078, battery: 25,999,078, operating_system_type: 25,950,291, timezone: 25,902,669
- Предполагаемая роль: Основной обучающий поток событий (часть 2).

### train_part_3.parquet
- Строк: **28,500,849**, столбцов: **23**
- Столбцы: `customer_id, event_id, event_dttm, event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id, operating_system_type, battery, device_system_version, screen_size, developer_tools, phone_voip_call_state, web_rdp_connection, compromised`
- Типы признаков: числовые=14, строковые/категориальные=9
- Пропуски: суммарно **332,456,935** (50.72% ячеек); столбцов с пропусками: 16
- Наибольшие пропуски: pos_cd: 26,052,102, browser_language: 25,845,914, battery: 25,845,914, operating_system_type: 25,795,167, timezone: 25,745,245
- Предполагаемая роль: Основной обучающий поток событий (часть 3).

## 2) Предполагаемая целевая переменная и тип задачи

- Целевая переменная: **`target`** (из `train_labels.parquet`).
- Распределение target: {1: 51438, 0: 36076}.
- Тип задачи: **бинарная классификация** (0/1).

## 3) Проверка объединения `train_part_*` с `train_labels.parquet`

- Рекомендуемый ключ объединения: **(`customer_id`, `event_id`)**.
- В `train_labels.parquet` строк: 87,514, уникальных пар ключей: 87,514.
- Диагностика по частям train:
  - `train_part_1.parquet`: строк=28,618,594, совпадений с labels=29,466, уникальных совпавших ключей=29,466.
  - `train_part_2.parquet`: строк=28,558,397, совпадений с labels=29,178, уникальных совпавших ключей=29,178.
  - `train_part_3.parquet`: строк=28,500,849, совпадений с labels=28,870, уникальных совпавших ключей=28,870.
- Ключей labels без матчей в `train_part_*`: **0**.
- Ключей labels, встречающихся более чем в одной части train: **0**.
- Вывод: каждая размеченная пара ключей находится ровно в одной из `train_part_*`, поэтому корректно: `concat(train_part_1..3)` -> `inner join` с labels по двум ключам.

## 4) Корректная baseline-стратегия (без тяжёлого обучения)

1. Объединить `train_part_1..3` вертикально и присоединить `train_labels.parquet` по (`customer_id`, `event_id`) через `inner join`.
2. Преобразовать `event_dttm` в datetime, добавить простые временные фичи: час, день недели, признак выходного.
3. Обработать пропуски:
   - числовые: медиана + бинарный флаг `is_missing`;
   - категориальные: отдельная категория `__MISSING__`.
4. Свести event-уровень в объектный уровень (`customer_id`,`event_id`) простыми агрегациями (count/mean/std/min/max/nunique) по числовым и top-frequency/nunique по категориальным.
5. Модель baseline: LightGBM/CatBoost (или LogisticRegression на one-hot при ограничениях), стратифицированный split по target.
6. Метрика baseline: ROC-AUC (дополнительно PR-AUC при дисбалансе).
7. Контроль утечек: не использовать post-event признаки, не смешивать `pretrain_*` с размеченным train до проверки постановки.
