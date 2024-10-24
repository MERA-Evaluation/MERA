# BPS

## Описание задачи

**Balanced Parentheses Sequence (BPS)** / Сбалансированная последовательность — алгоритмическая задача из [BIG-bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/cs_algorithms/valid_parentheses). Основная цель данной задачи — измерить способность языковых моделей понимать алгоритмические концепции CS, такие как стеки, рекурсия или динамическое программирование.
Каждый пример представляет из себя последовательность скобок. Задача модели — правильно ответить, сбалансирована данная последовательность или нет.

Входная строка сбалансирована, если:

1. Открытые скобки закрываются скобками того же типа.
2. Открытые скобки закрываются в правильном порядке.
3. Каждой закрывающей скобке соответствует открывающая скобка того же типа.

**Замечание:** Это диагностическое задание с открытым тестом. Результат на ней не участвует в расчет общего результата (Total score) модели на бенчмарке.

**Ключевые** **слова**: алгоритмы, численный ответ, длина контекста, скобки, бинарный ответ.

**Авторы:** Harsh Mehta, Behnam Neyshabur

### Мотивация

Алгоритмы — это способ экстраполяции примеров и один из способов описывать паттерны в задачах. В этом ключе способность языковых моделей понимать их - можно считать важным показателем “интеллекта”.

## Описание датасета

### Поля датасета

- instruction — строка, содержащая инструкцию для задачи и информацию о требованиях к формату вывода модели;
- inputs — пример скобочной последовательности;
- outputs — строка, содержащая правильный ответ: “1” если последовательность скобок сбалансирована, иначе “0”;
- meta — словарь, содержащий метаинформацию:
    - id — целое число, обозначающее номер задания.

### Пример данных

Ниже приведен пример данных:

```json
{
    "instruction": "Проверьте, сбалансирована ли входная последовательность скобок.\n\"{inputs}\"\nВыведите 1, если да и 0 в противном случае.",
    "inputs": "} } ) [ } ] ) { [ { { ] ( ( ] ) ( ) [ {",
    "outputs": "0",
    "meta": {
        "id": 242
    }
}
```

### Разбиение данных

Обучающий набор включает 250 примеров, тестовый набор включает 1000 примеров.

### Промпты

Для датасета было подготовлено 10 промптов различной сложности.

Пример:

```json
"Проверьте входную последовательность скобок: \"{inputs}\" на сбалансированность. В случае положительного ответа выведите 1, иначе 0.".
```

### Создание датасета

Последовательности скобок c длинами: 2, 4, 8, 12, 20 были сгенерированы со следующим распределением: `{20: 0,336, 12: 0,26, 8: 0,24, 4: 0,14, 2: 0,024}` для обучающего набора и: `{20: 0,301, 12: 0,279, 8: 0,273, 4: 0,121, 2: 0,026}` для тестового набора.

## Оценка

### Метрика

В качестве метрики качества используется Accuracy.

### Человеческая оценка

Человеческая оценка замерялась на подмножестве размера 100 (с аналогичным распределением как в исходном сете). Результат на этой задаче равен 1.0.
