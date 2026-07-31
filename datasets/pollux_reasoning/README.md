# POLLUX Reasoning

## Task Description

**POLLUX Reasoning** is a generative dataset for MERA Text 2.0: **117** test items and **5** few-shot examples (122 total). Examples are selected from [POLLUX](https://arxiv.org/abs/2505.24616) with the filter `Technical Problems` / `Hard` / `Общее`.

The dataset evaluates technical generation and reasoning: writing, editing, and analyzing code (SQL, Python, C#, C++, JavaScript). The model answer is scored by LLM-as-a-Judge (Pollux-4B-Judge) separately for each `criteria` item.

Evaluated skills: General Reasoning, Chain-of-Thought, Decomposition, Code Generation.

Contributors: Nikita Martynov, Anastasia Mordasheva, Dmitry Gorbetsky, Danil Astafurov, Ulyana Isaeva, Elina Basyrova, Sergey Skachkov, Victoria Berestova, Nikolay Ivanov, Valeriia Zanina, Alena Fenogenova

## Motivation

The dataset is intended to evaluate models' ability to solve technical tasks in free form: SQL and code in Python, C#, C++, and JavaScript — writing, editing, and analysis.


### Limitations

The dataset is not intended for evaluating formal literacy, linguistic analysis, or summarization. The scope is limited to technical / code tasks that require multi-step reasoning.

### Who benefits

The results are useful for builders of chatbots and assistants for business and household scenarios, as well as for developers of systems where reasoning and coding skills matter.

### Evaluated abilities

- Adequately understand the technical statement and choose a solution approach
- Keep a coherent logical sequence in the answer and in the code
- Show common sense when handling conditions and edge cases
- Write and analyze code in the listed languages

## Dataset Description

- `test.json` — **117** test examples (`meta.id` 6–122)
- `shots.json` — **5** few-shot examples (`meta.id` 1–5)
- Total: **122**

All examples: `categories.meta = Technical Problems`, `difficulty = Hard`, `domain = Общее`. Task types: write code, edit code, analyze code; subtypes: SQL, Python, C#, C++, JavaScript.


### Data Fields

- `instruction` [str] — one of five wrapper prompts from `dataset_meta.json` with the `{question}` placeholder
- `inputs.question` [str] — problem statement (text and/or code)
- `outputs` [str] — intentionally empty string; not a gold answer; the model answer appears at evaluation time
- `meta.id` [int]
- `meta.prompt_id` [int] — source POLLUX prompt id
- `meta.source_instruction` [str]
- `meta.categories` — `difficulty`, `domain`, `meta`, `task_type`, `task_subtype`, `task_subsubtype`
- `criteria` [list] — judge criteria: `criteria_name`, `criteria_description`, `rubrics`, `rubrics_example`
- `reference_answer` [str] — reference solution or fixed code; passed to the judge at evaluation time

### Data Instance

Example from `shots.json` (`meta.id = 1`):

```json
{
    "instruction": "Пожалуйста, помоги мне с решением вопроса, требующего логического размышления.\nОтвет выдай в формате:\nОтвет: <сгенерированный текст>\nСам вопрос, требующий логического подхода, таков: {question}",
    "inputs": {
        "question": "Предположим, у нас есть таблица shop_visits с полями:\n\nИдентификатор посещения — id\nДата посещения — visit_date\nЧисло покупателей — customer_cnt\n\nДата посещения является первичным ключом данной таблицы.\nПо мере увеличения id увеличивается date.\n\nЗадача: Напиши запрос на SQL, который будет выводить записи, у которых три или более последовательных идентификаторов (id), а количество покупателей для каждой строки этих последовательностей равно или превышает 100."
    },
    "outputs": "",
    "meta": {
        "categories": {
            "difficulty": "Hard",
            "domain": "Общее",
            "meta": "Technical Problems",
            "task_subsubtype": "",
            "task_subtype": "SQL",
            "task_type": "Написать код"
        },
        "id": 1,
        "prompt_id": 70,
        "source_instruction": "Предположим, у нас есть таблица shop_visits с полями:\n\nИдентификатор посещения — id\nДата посещения — visit_date\nЧисло покупателей — customer_cnt\n\nДата посещения является первичным ключом данной таблицы.\nПо мере увеличения id увеличивается date.\n\nЗадача: Напиши запрос на SQL, который будет выводить записи, у которых три или более последовательных идентификаторов (id), а количество покупателей для каждой строки этих последовательностей равно или превышает 100."
    },
    "criteria": [
        {
            "criteria_description": "Соответствие коду принятым стандарту и культуре.",
            "criteria_name": "Чистота и культура кода",
            "rubrics": "0: Код не читаем, стандарт языка не соблюдается, код плохо форматирован.\n\n1: Код читаем, однако содержит огрехи в форматировании, названии переменных, отступах, не везде выполняется стандарт языка.\n\n2: В решении использован общепринятый стандарт, названия переменных и функций отражают их суть, код хорошо форматирован.",
            "rubrics_example": ""
        },
        {
            "criteria_description": "Эффективность решения с точки зрения пространственной и временной сложности, оптимальность использования компонентов решения в разрезе итоговой производительности и потребления ресурсов",
            "criteria_name": "Оптимальный подход",
            "rubrics": "0: Решение не оптимально по вычислительной или пространственной сложности (худший или средний случай), либо решение реализовано не эффективно с точки зрения используемых компонентов, что увеличивает потребление ресурсов и / или временные затраты на отработку программы.\n\n1: Решение оптимально только с точки зрения либо пространственной, либо вычислительной сложности. Используемые компоненты существенно не увеличивают потребление ресурсов программой, такое решение может быть использовано, однако может быть улучшено с точки зрения потребления ресурсов.\n\n2: Решение эффективно с точки зрения используемых компонентов, решение соответствует оптимальным пространственной и вычислительной сложностям.",
            "rubrics_example": ""
        },
        {
            "criteria_description": "Решение содержит компоненты, заявленные в условии, или выполнено с учетом требований, указанных в задаче",
            "criteria_name": "Формальный учет требований из запроса пользователя",
            "rubrics": "0: Решение игнорирует требования условия задачи к решению. Использованы не те компоненты и / или методы, через которые предполагалось решать задачу. Сюда же входит решение, которое не проходит по вычислительной или пространственной сложности, если такие требования были указаны в условии.\n\n1: Решение содержит часть компонентов или методов, которые требовались в условии. Решение частично реализует требования, предъявляемые в условии задачи. Либо решение, кроме нужно функционала или его части, выполняет также лишний функционал, который не требовался изначально в условии.\n\n2: Решение содержит и выполняет ровно то, что требовалось в условии, код решает поставленную задачу, в коде нет ничего лишнего, никакого функционала, который не был заявлен в условии. В коде использованы ровно те компоненты, методы и парадигмы, которые требовались в условии.",
            "rubrics_example": ""
        },
        {
            "criteria_description": "Код решения выдает правильный ответ и / или решает поставленную в условии задачу, то есть этот критерий оценивает исключительно правильность ответа, который выдает написанный код, в отрыве от того, как и с помощью чего этот код реализован.",
            "criteria_name": "Правильность ответа",
            "rubrics": "0: В результате работы программы получается неправильный ответ, ответ отсутствует или программа не выполняет ни один из требуемых в условии функционалов, либо ответ программы меняется в зависимости от входных данных. \n\n1: Программа выдает правильный ответ, который не зависит от входных данных, на часть вопросов, заданных в условии задачи, или обеспечивает корректное выполнение только части функционала, заложенного в условии задачи.\n\n2: Ответ программы полностью правильный, весь функционал реализован, правильный ответ не зависит от входных данных",
            "rubrics_example": ""
        }
    ],
    "reference_answer": "WITH q1 AS (\nSELECT *, \n     COUNT(*) OVER( ORDER BY id RANGE BETWEEN CURRENT ROW AND 2 FOLLOWING ) following_cnt,\n     COUNT(*) OVER( ORDER BY id RANGE BETWEEN 2 PRECEDING AND CURRENT ROW ) preceding_cnt,\n     COUNT(*) OVER( ORDER BY id RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING ) current_cnt\nFROM shop_visits\nWHERE customer_cnt >= 100\n)\nSELECT id, visit_date, customer_cnt\nFROM q1\nWHERE following_cnt = 3 OR preceding_cnt = 3 OR current_cnt = 3\nORDER BY visit_date"
}
```

### Prompts

Five prompt variants were prepared for the task. They are distributed evenly across examples on a one-example–one-prompt basis. Each prompt asks for free-form text generation and requires an answer in the format `Ответ: <сгенерированный текст>`. Placeholders in curly braces (in particular `{question}`) are filled from the fields inside `inputs` for each question. The prompt texts are stored in `dataset_meta.json` and duplicated in each example's `instruction` field.

### Dataset Creation
The dataset is a subsample of the closed part of the POLLUX set that complements the public version: [POLLUX](https://arxiv.org/abs/2505.24616). Examples are selected with the filter `Technical Problems` / `Hard` / `Общее`.

### Metrics

- `llm_as_judge`. For each `criteria` item, a separate Pollux-4B-Judge call is made via an OpenAI-compatible endpoint. Criterion scores are 0/1/2; the example score is the mean of the scores divided by 2.
