# RuRegions


## Task Description

**RussianRegions** is a dataset for evaluating the capabilities of language models in solving tasks in variations of the Russian language specific to certain regions of Russia. The benchmark tests knowledge of Russian regionalisms, as well as the ability to use them in context.

Evaluated skills: Russian language proficiency, Regional culture awareness

Contributors: Alexander Kharitonov


## Motivation

Language models often use texts written in codified (standard) language during training. However, there are less frequent linguistic phenomena, such as slang, dialectisms, neologisms, archaisms, or regionalisms, which are also part of the language and should be considered when evaluating a model's language proficiency. This dataset is dedicated to assessing the level of proficiency in Russian regionalisms, given the vast geography of the Russian Federation.


### Intended Use of Results

The evaluation results are useful for researchers and developers of language models focused on Russian, especially when it is important to test how a model handles regionalisms, rare dialectal expressions, and non-standard vocabulary. The dataset may also be useful for specialists in Russian dialectology, regional linguistics, and model evaluation tasks that require accounting for regional variation in Russian.


### Limitations

The dataset is not suitable for evaluating general-purpose language models or language models that are not focused on Russian culture, because solving the task requires deep knowledge of rarely used regional dialects of Russia.


## Data Description

### Data Fields

Each dataset question contains the following fields:

- `instruction` [str] — a prompt template for the language model. The template is assembled from block fields inside `inputs`.
- `inputs` — input data that forms the task for the model:
    - `introduction` [str] — a short introductory phrase for the prompt;
    - `task_formulation` [str] — the task formulation;
    - `lang_unit` [str] — the language unit to use as a replacement; used in task types 2 and 3;
    - `example_text` [str] — the reference text that defines the required meaning of the language unit; used in task type 2;
    - `region` [str] — the region for which the language unit is characteristic; used in task type 3;
    - `option_a` [str] — the first answer option;
    - `option_b` [str] — the second answer option;
    - `option_c` [str] — the third answer option;
    - `option_d` [str] — the fourth answer option;
    - `format_description` [str] — the answer format description; for task type 1 it specifies that the model must output a single capital Russian letter from (А, Б, В, Г) corresponding to the chosen text; for task types 2 and 3 it also requires the language unit from the chosen text after a semicolon.
- `outputs` [str] — a string containing the correct answer.
- `meta` — metadata related to the test example, not used in the question (hidden from the tested model):
    - `id` [int] — an integer ID of the example;
    - `group_id` [int] — an integer ID of the group of examples. Examples share the same value when the regionalism needed for the correct answer is the same;
    - `task_type` [int] — an integer ID of the task type: 1 — general understanding of text with regional expressions; 2 — synonymy between regionalisms and codified language; 3 — regions of origin of regional words.


### Data Formatting Example

```json
{
    "instruction": "{introduction}\n\nЗадача:\n{task_formulation}\n\nФормат ответа:\n{format_description}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}",
    "inputs": {
        "introduction": "Пожалуйста, прочитайте тексты внимательно.",
        "task_formulation": "Выберите текст, в котором нет смысловых ошибок.",
        "option_a": "Не везет мне с мужчинами почему-то. Все подруги давно замужем, многие не по одному разу. Даже такая толчёнка, как ты, нашла себе любящего мужа.",
        "option_b": "Скубались они всегда громко, наслаждаясь процессом и полностью ему отдаваясь. Сначала меня это удивляло, но скоро я уже не обращал внимания.",
        "option_c": "Виктор сильно окреп, в свои 12 лет он был ростом со старшего брата, весил так же. Он буквально вырос на жидриках да на свежем воздухе.",
        "option_d": "Для тебя одна радость — мжить в свой дебилизатор и только, пока дым из ушей не пойдёт. Лучше бы с ребятами мяч погонял.",
        "format_description": "В ответе напишите только одну строку, начинающуюся с «Ответ:». После двоеточия укажите заглавную букву из множества (А, Б, В, Г), обозначающую выбранный вариант ответа."
    },
    "outputs": "Б",
    "meta": {
        "id": 1,
        "group_id": 0,
        "task_type": 1
    }
}
```


### Prompts

For the task, 15 prompts were prepared: 5 prompt variants for each of the 3 task types. Each prompt contains the fields `task_type`, `instruction`, `introduction`, `task_formulation`, and `format_description`. These blocks are used in the `instruction` field; values in curly braces are filled from `inputs`.

Prompt example:

```json
{
    "task_type": 1,
    "instruction": "{introduction}\n\nЗадача:\n{task_formulation}\n\nФормат ответа:\n{format_description}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}",
    "introduction": "Интересно, сможете ли вы найти корректный текст?",
    "task_formulation": "Определите, какой текст не содержит смысловых искажений.",
    "format_description": "В ответе напишите только одну строку, начинающуюся с «Ответ:». После двоеточия укажите только одну заглавную букву русского алфавита — А, Б, В или Г — которая соответствует выбранному тексту."
}
```


### Dataset Creation

Open materials were used as sources for regional expressions: [1](https://ru.wiktionary.org/wiki/%D0%9F%D1%80%D0%B8%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5:%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D0%B9%D1%81%D0%BA%D0%B8%D0%B5_%D1%80%D0%B5%D0%B3%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%D0%BC%D1%8B), [2](https://yandex.ru/company/researches/2021/local-words), as well as expressions collected by expert linguists. The tasks for the dataset were also compiled by expert linguists. The tasks in the dataset are divided into 3 types: Type 1. General understanding of a text containing regional expressions; Type 2. Understanding the synonymy between regionalisms and the codified language; Type 3. Understanding the regions of origin of regional words. Each task type contains 201 examples, for a total of 603 examples in the dataset.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Exact match`: Exact match is the average of scores for all processed cases, where a given case score is 1 if the predicted string is the exact same as its reference string, and is 0 otherwise.
- `Group Exact match`: Group Exact match is the average over groups of examples with the same `group_id`. The `group_id` value groups tasks that test knowledge of the same regionalism. A group receives a score of 1 only if the model answers all tasks related to that regionalism correctly, and 0 otherwise.
- `Judge Score`: Judge Score uses an LLM-as-a-Judge approach to compare the model prediction with the reference answer for each example. The judge model, API endpoint, and prompt path are configured through environment variables.
- `Group Judge Score`: Group Judge Score aggregates Judge Score over groups of examples with the same `group_id`. A group receives a score of 1 only if the judge marks all tasks related to the same regionalism as correct, and 0 otherwise.
