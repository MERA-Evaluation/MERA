# Luzitania


## Task description

**Luzitania** is a dataset for evaluating the ability of language models to solve mathematical problems in Russian. The dataset focuses on multi-step textual reasoning and measures the model's ability to arrive at the correct answer given the problem statement.

**Keywords:** math, reasoning.

Evaluated skills: Reasoning

Contributors: Irina Piontkovskaya, Alexey Rukhovich, Kirill Morozov, Maksim Titov, Elena Entina, Laida Kushnareva, Kristian Kuznetsov, Alexander Podolskiy, Semen Molokov, Timofey Gerasin, Artem Khrapov, Yuliya Skripkar, Nikita Okhotnikov, Pavel Efimov.


## Motivation

This dataset tests the ability of models to build sequential reasoning in the mathematical domain and arrive at justified answers. The difficulty level requires models to maintain the reasoning chain and sometimes perform lengthy calculations. The dataset is not intended for evaluating multimodal capabilities or image understanding: all problem statements are textual. The dataset helps to understand how well a model maintains context, does not get "confused" in reasoning, and arrives at a justified numerical result.


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — a string containing the prompt formulation for the language model;;
- `inputs` — Input data that forms the task for the model. Can include one or multiple modalities - video, audio, image, text.
    - `question` [str] — a string containing the actual problem statement;;
- `outputs` [str] — a string containing the correct answer (in most cases, an integer);;
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — ID (integer).
    - `source` [str] — a string indicating the problem source (see the Dataset Creation section).
    - `spec` [str] — a string, possibly empty, providing additional information for locating the problem within its source.


### Data formatting example

```json
{
    "instruction": "Прочитайте задачу: \"{question}\". Найдите ответ. По умолчанию ответ — целое число. В случае нестандартного формата ответа следуйте указаниям в условии.",
    "inputs": {
        "question": "Вы заперты в комнате с единственным выходом — длинным коридором с рядом дверей и минными ловушками. Чтобы выбраться, нужно открыть все двери и обезвредить все мины. В комнате есть панель с 3 кнопками и инструкция. Красная кнопка устанавливает мину, жёлтая кнопка обезвреживает две мины и закрывает одну дверь, а зелёная кнопка открывает две двери. Изначально закрыты 3 двери и установлены 3 мины. В инструкции предупреждается, что попытка обезвредить две мины или открыть две двери, когда активна только одна мина или закрыта только одна дверь, сбрасывает систему в исходное состояние. Какое минимальное количество нажатий кнопок потребуется, чтобы выбраться?"
    },
    "outputs": "9",
    "meta": {
        "id": 999,
        "source": "example",
        "spec": ""
    }
}
```


### Prompts

For the task, 10 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
Решите следующую задачу. Ответ запишите в виде целого числа, если в задаче не указано иное. {question}
```


### Dataset creation

The test set was collected from open sources, with problem statements translated into Russian where necessary. Below are all problem sources (i.e., all possible unique values in the `meta.source` field) and the number of problems from each source in the dataset.

| source | count |
|--------|-------|
| olympiads | 133 |
| olympic_reason | 50 |
| MathArena | 36 |
| turgor | 31 |
| chinese_olympiads_2002_2006 | 8 |

#### olympiads

Problems taken from the open dataset [olympiads](https://huggingface.co/datasets/aslawliet/olympiads).

Filtering was performed as follows: first, answers were extracted from the solutions for all problems, then problems with non-negative integer answers were selected. A medium-sized reasoning model (GPT-oss-120B, reasoning "high", `max_length=65536` tokens) was run on this set of problems for 8 attempts per problem (4 without `tool_call` and 4 with `tool_call`). The current set includes problems with a success rate of `0 < x ≤ 50%` such that the average response length of GPT-oss-120B was at least 12k tokens, and the shortest correct answer was at least 10k tokens.

The `spec` field is empty in this case.

#### olympic_reason

50 olympiad problems from various countries released after April 2025. A subsample of these problems can be used as a separate validation set for earlier models to avoid data leakage into training sets.

The `spec` field contains the abbreviation of the olympiad name and the problem number.

#### MathArena

Selected problems from the open leaderboard [MathArena](https://matharena.ai/).

The `spec` field contains the name of one of the ArxivMath[1] and Apex[2] subsets:

- `ArXivMath_Feb_2026`: 12 problems from [ArXivMath_Feb_2026](https://huggingface.co/datasets/MathArena/arxivmath-0226)
- `ArXivMath_Jan_2026`: 9 problems from [ArXivMath_Jan_2026](https://huggingface.co/datasets/MathArena/arxivmath-0126)
- `ArXivMath_Dec_2025`: 9 problems from [ArXivMath_Dec_2025](https://huggingface.co/datasets/MathArena/arxivmath-1225)
- `apex`: 8 problems, [apex](https://huggingface.co/datasets/MathArena/apex_2025)

#### turgor

Problems from the "Tournament of Towns".

[https://turgor.ru/problems/](https://turgor.ru/problems/)

The `spec` field is empty in this case.

#### chinese_olympiads_2002_2006

Book "Mathematical Olympiad in China: Problems and Solutions", Xiong Bin and Lee Peng Yee
([PDF](https://phuylai.wordpress.com/wp-content/uploads/2009/10/mathematical-olympiad-in-china-problems-and-solutions.pdf))

The `spec` field is empty in this case.

Some problem statements from all sources may be slightly modified so that the answer is an integer.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Accuracy`: Accuracy is the proportion of correct model predictions among the total number of cases processed.
