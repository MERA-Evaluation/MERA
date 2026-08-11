# T-Math


## Task Description

**T-Math** is a Russian-language dataset of math olympiad problems for evaluating the mathematical reasoning capabilities of large language models. The task is to solve a problem written in Russian and return only the final verifiable answer.

The dataset contains 310 problems from the [All-Russian School Olympiad](https://vos.olimpiada.ru/) and the [Moscow Olympiad](https://mos.olimpiada.ru) for school students, covering the period from 2005 to 2025. The tasks and their ground-truth answers were extracted automatically and then verified by human assessors.

Details are presented in the academic paper [T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground](https://aclanthology.org/2026.eacl-demo.22/)

Evaluated skills: Mathematical reasoning, Multi-step problem solving, Russian language understanding

Contributors: T-Pro Team


## Motivation

Mathematical olympiad problems are a challenging benchmark for language models because they require multi-step reasoning, careful interpretation of the problem statement, and the ability to produce a compact final answer. T-Math focuses on Russian olympiad-level problems and therefore helps evaluate reasoning models in a setting that is underrepresented in many English-centric math benchmarks.

The dataset is designed to be easily verifiable: target answers are numeric or mathematical expressions and can be checked automatically with expression verification tools such as `math_verify`.


### Intended Use of Results

The evaluation results are useful for researchers and developers of language models focused on mathematical reasoning in Russian. The benchmark can be used to compare reasoning-oriented models, assess the effect of instruction tuning or reinforcement learning, and track progress on olympiad-level problem solving.


### Limitations

The dataset is not intended to measure general mathematical knowledge across all areas of mathematics. Problems were filtered to keep tasks with a single verifiable final answer, so theorem-proving tasks, tasks with multiple answers, non-numeric answers, and tasks requiring images are excluded. Because the dataset contains olympiad problems, results may depend strongly on a model's reasoning budget and decoding setup.


## Data Description

### Data Fields

Each dataset question contains the following fields:

- `instruction` [str] — a prompt template for the language model. The template contains the task formulation, the problem statement placeholder, and answer format requirements.
- `inputs` — input data that forms the task for the model:
    - `question` [str] — the mathematical olympiad problem statement.
- `outputs` [str] — a string containing the correct final answer.
- `meta` — metadata related to the test example, not used in the question (hidden from the tested model):
    - `id` [int] — an integer ID of the example;
    - `grade` [str] — the school grade associated with the problem, or `all` if the grade is not specified;
    - `task_complexity` [str] — the difficulty label of the problem (`EASY`, `MEDIUM`, or `HARD`);
    - `olympiad` [str] — the source olympiad: `mos` for the Moscow Olympiad or `vos` for the All-Russian School Olympiad;
    - `solutions` [str] — the reference solution or explanation;
    - `year` [str] — the year or academic year of the problem.


### Data Formatting Example

```json
{
    "instruction": "Привет! Поможешь с задачей?\n\nЗадача:\nНайдите ответ к следующей математической задаче.\n\nУсловие:\n{question}\n\nФормат ответа:\nУкажите только итоговый ответ.",
    "inputs": {
        "question": "Дима спускается на лифте с этажа за 1 минуту, а поднимается пешком после нажатия доступной кнопки. Путь наверх занимает 1 минуту 10 секунд. Лифт и Дима имеют постоянные скорости. На каком этаже живет Дима?"
    },
    "outputs": "7",
    "meta": {
        "id": 329,
        "grade": "all",
        "task_complexity": "MEDIUM",
        "olympiad": "mos",
        "solutions": "Первое решение: Разница между подъемом и спуском составляет 10 секунд. Путь пешком — 1 промежуток между этажами. Второе решение: Решение системы уравнений дает n=7, m=6.",
        "year": "2008"
    }
}
```


### Prompts

For the task, five prompt variants were prepared. Each prompt asks the model to solve a mathematical problem and output only the final answer. The problem statement is substituted into the `{question}` placeholder.

Prompt variants used in the dataset:

```json
[
    {
        "instruction": "Решите математическую задачу.\n\nУсловие:\n{question}\n\nФормат ответа:\nВыведите только окончательный ответ."
    },
    {
        "instruction": "Привет! Поможешь с задачей?\n\nЗадача:\nНайдите ответ к следующей математической задаче.\n\nУсловие:\n{question}\n\nФормат ответа:\nУкажите только итоговый ответ."
    },
    {
        "instruction": "Помогите мне, пожалуйста.\n\nЗадача:\nРешите математическую задачу и найдите правильный ответ.\n\nУсловие:\n{question}\n\nФормат ответа:\nВ качестве ответа укажите только окончательный ответ."
    },
    {
        "instruction": "Задача:\nВыполните решение математической задачи и определите итоговый результат.\n\nУсловие:\n{question}\n\nФормат ответа:\nВерните только окончательный ответ."
    },
    {
        "instruction": "Задача:\nПо данному условию нужно найти правильный математический ответ.\n\nУсловие:\n{question}\n\nФормат ответа:\nВ ответе укажите только окончательный результат."
    }
]
```


### Dataset Creation

The source texts were extracted from PDFs using [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct). Problems, ground-truth answers, and verifiable numeric answers were extracted with LLM calls.

Invalid questions were filtered out using an LLM according to the following criteria:

- tasks requiring multiple answers;
- tasks without a single correct answer;
- theorem-like tasks where the main goal is proving a statement;
- tasks with non-numeric answers;
- tasks that cannot be solved without access to an accompanying image.

Tasks of moderate difficulty where Qwen3-8B achieved a 100% pass@16 rate were removed, because they offered limited value for benchmarking reasoning. Finally, both the questions and the verifiable answers were manually reviewed by assessors to ensure consistency with the original sources.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Exact match`: Exact match is the average of scores for all processed cases, where a given case score is 1 if the predicted string exactly matches the reference answer, and 0 otherwise. In this task, exact match means mathematical equivalence between the prediction and the reference answer, checked with the `math_verify` library.


## Citation

If you find our work useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{stoianov-etal-2026-pro,
    title = "{T}-pro 2.0: An Efficient {R}ussian Hybrid-Reasoning Model and Playground",
    author = "Stoianov, Dmitrii  and
      Taranets, Danil  and
      Tsymboi, Olga  and
      Latypov, Ramil  and
      Dautov, Almaz  and
      Kruglikov, Vladislav  and
      Nikita, Surkov  and
      Abramov, German  and
      Gein, Pavel  and
      Abulkhanov, Dmitry  and
      Gashkov, Mikhail  and
      Zelenkovskiy, Viktor  and
      Batalov, Artem  and
      Medvedev, Aleksandr  and
      Potapov, Anatolii",
    editor = "Croce, Danilo  and
      Leidner, Jochen  and
      Moosavi, Nafise Sadat",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = mar,
    year = "2026",
    address = "Rabat, Marocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-demo.22/",
    doi = "10.18653/v1/2026.eacl-demo.22",
    pages = "297--319",
    ISBN = "979-8-89176-382-1"
   }
```
