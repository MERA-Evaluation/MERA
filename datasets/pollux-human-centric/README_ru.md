# POLLUXHumanCentric

POLLUXHumanCentric оценивает открытое следование инструкциям на сложной части Human Interaction датасета [`ai-forever/POLLUX-instructions`](https://huggingface.co/datasets/ai-forever/POLLUX-instructions), оформленной в формате MERA.

Каждый пример содержит пользовательский запрос, обернутый в один из шаблонов инструкций MERA, эталонный ответ и список критериев оценки.

## Мотивация

Датасет ориентирован на instruction-tuned генеративные модели и проверяет, насколько хорошо они отвечают на человеко-ориентированные открытые запросы. Он полезен там, где exact match не является осмысленной метрикой, а важнее качество ответа.

## Создание датасета

Датасет создается фильтрацией `train`-раздела `ai-forever/POLLUX-instructions`: сохраняются только записи, где `meta == "Human Interaction"`, а `difficulty.lower() == "hard"`. Пять примеров помещаются в `shots.json`, остальные примеры сохраняются в `test.json`. Исходные `criteria` и `reference_answer` вынесены в верхнеуровневые поля и передаются Pollux Judge при оценке.

## Оценка

Ответы моделей оцениваются через Pollux-4B-Judge, развернутый как OpenAI-compatible vLLM endpoint. Judge вызывается отдельно для каждого критерия из списка `criteria` с отрендеренной инструкцией, ответом модели, `reference_answer`, названием критерия и rubric. Используются только критерии со шкалой `0/1/2`; итоговая метрика считается как `mean(raw criterion scores) / 2`.

## Human Baseline

Human baseline пока не добавлен. Значение-заглушка: `0.0`.

## Авторы

MERA contributors
