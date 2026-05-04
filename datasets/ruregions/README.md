# RuRegions


## Task description

**RussianRegions** is a dataset for evaluating the capabilities of language models in solving tasks in the Russian language that are specific to certain regions of Russia. The benchmark tests knowledge of Russian regionalisms, as well as the ability to use them in context.

**Keywords:** russian language, regional culture awareness.

Evaluated skills: Russian language proficiency, Regional culture awareness

Contributors: Alexander Kharitonov


## Motivation

Language models often use texts written in codified (standard) language during training. However, there are less frequent linguistic phenomena, such as slang, dialectisms, neologisms, archaisms, or regionalisms, which are also part of the language and should be considered when evaluating a model's language proficiency. This dataset is dedicated to assessing the level of proficiency in Russian regionalisms, given the vast geography of the Russian Federation.


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — a string containing the prompt formulation for the language model;
- `inputs` — Input data that forms the task for the model. Can include one or multiple modalities - video, audio, image, text.
    - `option_a` [str] — a string containing the first option;
    - `option_b` [str] — a string containing the second option;
    - `option_c` [str] — a string containing the third option;
    - `option_d` [str] — a string containing the fourth option;
    - `condition` [str] — a string describing the input data;
    - `task_formulation` [str] — a string containing the task formulation;
    - `format_description` [str] — a string containing the format description;
- `outputs` [str] — a string containing the correct answer;
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — an integer ID of the example;
    - `group_id` [int] — an integer ID of the group of examples;
    - `task_part` [int] — an integer ID of the task part


### Data formatting example

```json
{
    "instruction": "{condition}\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\n\n{task_formulation}\n\n{format_description}",
    "inputs": {
        "option_a": "Не везет мне с мужчинами почему-то. Все подруги давно замужем, многие не по одному разу. Даже такая толчёнка, как ты, нашла себе любящего мужа.",
        "option_b": "Скубались они всегда громко, наслаждаясь процессом и полностью ему отдаваясь.Сначала меня это удивляло, но скоро я уже не обращал внимания.",
        "option_c": "Виктор сильно окреп, в свои 12 лет он был ростом со старшего брата, весил так же. Он буквально вырос на жидриках да на свежем воздухе. ",
        "option_d": "Для тебя одна радость — мжить в свой дебилизатор и только, пока дым из ушей не пойдёт. Лучше бы с ребятами мяч погонял.",
        "condition": "Дан следующий набор текстов:",
        "task_formulation": "Выберите среди них тот, который не содержит смысловых ошибок.",
        "format_description": "В качестве ответа выпишите букву этого текста."
    },
    "outputs": "Б",
    "meta": {
        "id": 0,
        "group_id": 0,
        "task_part": 1
    }
}
```


### Prompts

For the task, 15 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
"Дан следующий набор текстов:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\n\nВыберите среди них тот, который не содержит смысловых ошибок.\n\nВ качестве ответа выпишите букву этого текста."
```


### Dataset creation

Open materials were used as sources for regional expressions: [1](https://ru.wikipedia.org/wiki/Регионализмы_русского_языка), [2](https://yandex.ru/company/researches/2021/local-words), as well as expressions collected by expert linguists. The tasks for the dataset were also compiled by expert linguists. The tasks in the dataset are divided into 3 types: Type 1. General understanding of a text containing regional expressions; Type 2. Understanding the synonymy between regionalisms and the codified language; Type 3. Understanding the regions of origin of regional words. Each task type contains 200 examples, for a total of 600 examples in the dataset.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Exact match`: Exact match is the average of scores for all processed cases, where a given case score is 1 if the predicted string is the exact same as its reference string, and is 0 otherwise.
- `Group Exact match`: Group Exact match is the average of scores for subsets of processed cases (all cases are split into disjoint subsets and the metric is computed independently for each of them), where a given case score is 1 if the predicted string is the exact same as its reference string, and is 0 otherwise.
