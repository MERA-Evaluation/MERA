# MMReD

## Task Description

**MMReD** (Dense Context Reasoning Benchmark) is a synthetic benchmark for evaluating long-context dense reasoning in language models. The benchmark generates sequences of room occupancy states where characters move between rooms at each step. Models must aggregate information across the entire sequence to answer questions about character behavior patterns.

The MERA subset focuses on 5 Dense Context (DC) question types at 3 sequence lengths (32, 64, 128 steps), totaling 15 subtasks with 50 questions each (750 total).

Tested model skills: Long-context reasoning, Dense aggregation, Counting, Temporal tracking

Authors: Boris Shirokikh, Maxim Kurkin

## Motivation

Existing reasoning benchmarks either test needle-in-a-haystack retrieval (finding a single fact) or rely on natural language texts where ground truth is ambiguous. MMReD fills the gap by evaluating dense context reasoning — the ability to aggregate information across many observations to derive a single answer.

This is important for evaluating:
- LLMs handling structured long-context data
- Models' ability to count, track, and compare across temporal sequences

The synthetic nature ensures perfect ground truth, no data contamination, and controllable difficulty via sequence length. Longer sequences require tracking more state, making the task progressively harder.

Exact Match is used as the metric because answers are unambiguous single words or numbers. Scores are aggregated at three levels: per subtask, per question type (length-weighted), and overall (harmonic mean across types).

## Dataset Description

The dataset contains **750** examples across **15** subtasks (5 question types × 3 sequence lengths).

### Distribution by Question Type

Each example is assigned a task type in the `meta.categories.task_type` field:

| `task_type` | Question type | Count | Share |
|-------------|---------------|------:|------:|
| DC-SA-C | Who spent the most/least time alone? | 150 | 20.0% |
| DC-SR-I | How many steps did X spend in room Y? | 150 | 20.0% |
| DC-CC-I | For how many steps was there a crowd (3+ people)? | 150 | 20.0% |
| DC-WS-R | In which room did X spend the most/least time? | 150 | 20.0% |
| DC-WHS-C | Who spent the most/least time in room X? | 150 | 20.0% |

### Distribution by Sequence Length

Each example is assigned a sequence length in the `meta.categories.seq_len` field:

| `seq_len` | Count | Share |
|----------:|------:|------:|
| 32 | 250 | 33.3% |
| 64 | 250 | 33.3% |
| 128 | 250 | 33.3% |

The answer format is a single word or number: a character name, a room name, or a count. The model must respond in the format `Ответ: X`, where X is one token.

### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] — instruction prompt template with question element placeholders

- `inputs` — input data that forms the task:
    - `context` [str] — step sequence as JSONL: one `{"step_id", "rooms"}` object per line
    - `question` [str] — text of the question

- `outputs` [str] — the correct answer to the question

- `meta` — metadata:
    - `id` [int] — identification number of the question in the dataset
    - `categories` — categorial features characterizing the test example:
        - `task_type` [str] — task type code (DC-SA-C, DC-SR-I, DC-CC-I, DC-WS-R, DC-WHS-C)
        - `seq_len` [int] — sequence length (number of steps)
        - `atype` [str] — answer type: person, room, or number

### Data Example

```json
{
    "instruction": "Задача:\nПроанализируй перемещения и расположение персонажей по шагам.\n\nКонтекст:\nПоследовательность шагов в формате JSON показывает, какие персонажи находятся в каких комнатах.\n{context}\n\nФормат ответа:\nОтветь одним словом или одним числом в формате: Ответ: X\n\nВопрос:\n{question}\n\nОтвет:",
    "inputs": {
        "context": "{\"step_id\": 1, \"rooms\": {\"Кухня\": [\"Сандра\", \"Мария\"], \"Ванная\": [\"Иван\", \"Михаил\"], \"Сад\": [], \"Офис\": [], \"Спальня\": [\"Даниил\"], \"Коридор\": []}}\n{\"step_id\": 2, \"rooms\": {\"Кухня\": [\"Сандра\"], \"Ванная\": [], \"Сад\": [\"Даниил\"], \"Офис\": [\"Иван\", \"Мария\"], \"Спальня\": [\"Михаил\"], \"Коридор\": []}}",
        "question": "Кто провёл больше всего времени в одиночестве?"
    },
    "outputs": "Даниил",
    "meta": {
        "id": 1,
        "categories": {
            "task_type": "DC-SA-C",
            "seq_len": 2,
            "atype": "person"
        }
    }
}
```

### Prompt Creation

10 prompts were prepared for the task. They were distributed evenly across questions using a one-question–one-prompt scheme. Template placeholders in curly braces in a prompt are filled from the fields inside `inputs` for each question.

### Dataset Creation

The dataset is fully synthetic, generated procedurally:
1. For each step, characters (Sandra, Mary, John, Daniel, Michael) are randomly assigned to rooms (Kitchen, Bathroom, Garden, Office, Bedroom, Hallway).
2. Questions are generated from templates with verified ground-truth answers.
3. Sequences are deduplicated via SHA-256 hashing.
4. Each question type × sequence length combination has 50 samples.

Seed: 0xBADFACE for reproducibility.

Generation code: https://github.com/MERA-Evaluation/MERA (after merge) or https://github.com/Fr0do/mmred

### Metrics

Evaluation uses a three-level hierarchy:

| Level | Group / task | Metric | Aggregation |
|-------|--------------|--------|-------------|
| 1 | Each of 15 subtasks (`mmred_dc_*_{32,64,128}`) | **Subtask Exact Match** | Mean over 50 questions |
| 2 | Each of 5 question-type groups (`mmred_dc_sa_c`, …, `mmred_dc_whs_c`) | **Question-type Length-weighted EM** | Length-weighted mean over 3 sequence lengths |
| 3 | Overall benchmark (`mmred`) | **Overall MMReD Score** | Harmonic mean over 5 question-type groups |

In evaluation logs, level 1 is reported as `exact_match`; levels 2 and 3 as `em.dc_aggregate`.

**Subtask Exact Match** — the proportion of answers that exactly match the reference (a character name, room name, or number) within one subtask. Value from 0 to 1.

**Question-type Length-weighted EM** — combines subtask scores for sequence lengths 32, 64, and 128 with weights **2, 4, and 16** (longer contexts contribute more):

```
question_type_score = (2 * EM_32 + 4 * EM_64 + 16 * EM_128) / (2 + 4 + 16)
```

**Overall MMReD Score** — harmonic mean of the five question-type scores. Penalizes weakness on any single type:

```
overall_score = 5 / (1/(s_1 + eps) + 1/(s_2 + eps) + ... + 1/(s_5 + eps)),  eps = 1e-6
```

where `s_i` is the **Question-type Length-weighted EM** for question type `i`.
