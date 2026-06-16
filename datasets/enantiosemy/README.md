# Enantiosemy

## Task Description

The **Enantiosemy** dataset is a dataset for evaluating the ability of language models to understand words that can have opposite meanings in the same form depending on context. The benchmark tests the model's ability to correctly interpret enantiosemic units in one of four multiple-choice settings: selecting a suitable continuation, selecting an unsuitable continuation, identifying the meaning used in context, or identifying the meaning not used in context. The expected answer is one or several option letters.

Tested model skills: Russian language proficiency, Contextual disambiguation, Linguistic-aware reasoning

Authors: Denis Shevelev, Alexander Astafurov, Alexander Kharitonov

## Motivation

Russian has a phenomenon known as **enantiosemy** (also called contronymy or intralexical antonymy), where the same word or phrase in the same form can have opposite meanings depending on the context in which they occur. For example, the verb "прослушать" can mean both "to listen attentively" and "to fail to hear, to miss." For humans, understanding such words is usually not difficult, but for language models this task remains non-trivial: the model must not only recognize the enantiosemic unit, but also determine which of its opposite meanings is used in the given context, and then select a semantically consistent continuation.

### Limitations

The dataset is intended for Russian-language text models and does not evaluate multimodal abilities, dialogue safety, or factual knowledge outside the provided context. The task focuses on controlled multiple-choice selection, so the results should not be interpreted as a complete evaluation of a model's ability to handle all cases of enantiosemy in open-ended generation.

### Validity

The examples are designed so that the intended meaning of the enantiosemic unit can be recovered from the local context, while the alternative meaning is ruled out by the surrounding utterances. The four task types test both positive and negative recognition of the contextual meaning, which reduces the chance that a model can solve the task only through superficial continuation matching.

## Dataset Description

The dataset contains **506** examples and **159** unique enantiosemic words.

### Distribution by Task Type

Each example is assigned one of four types in the `type` field:

| `type` | Task type | Count | Share |
|--------|-----------|------:|------:|
| 1 | Select the **correct** continuation of the final utterance | 290 | 57.3% |
| 2 | Select the **incorrect** / unsuitable continuation | 188 | 37.2% |
| 3 | Identify the meaning **in which the word is used** | 18 | 3.6% |
| 4 | Identify the meaning **in which the word is not used** | 10 | 2.0% |

The answer format is multiple choice among 6 options labeled А, Б, В, Г, Д, Е. One or several answer options may be correct. When several options are correct, the answer letters are written in alphabetical order and separated by a semicolon and a space.

### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model

- `inputs` — input data that forms the task:
    - `text` [str] — a dialogue or utterance containing an enantiosemic element
    - `enantiosemic_word` [str] — the enantiosemic word in its initial form
    - `option_a` [str] — the first continuation option for the final utterance
    - `option_b` [str] — the second continuation option
    - `option_c` [str] — the third continuation option
    - `option_d` [str] — the fourth continuation option
    - `option_e` [str] — the fifth continuation option
    - `option_f` [str] — the sixth continuation option

- `outputs` [str] — a string containing the letter or letters of the correct answer, written in alphabetical order and separated by a semicolon and a space, for example: А; Б; В; Г; Д; Е

- `meta` — metadata:
    - `id` [int] — example number
    - `type` [int] — task type (1–4, see the table above)

### Data Example

```json
{
      "instruction": "Задача:\\nВыберите тезис, который наиболее логично продолжает заключительную реплику текста и отражает мысль персонажа — исходя из того значения, в котором употреблено энантиосемическое слово.\\n\\nТекст:\\n{text}\\n\\nВарианты ответа:\\nА. {option_a}\\nБ. {option_b}\\nВ. {option_c}\\nГ. {option_d}\\nД. {option_e}\\nЕ. {option_f}\\n\\nЭнантиосемическое слово:\\n{enantiosemic_word}",
      "inputs": {
        "text": "— «Миль Попс, жу-жу-жу, жу-жу-жу…»\n— Таня, прекрати, ты мне на нервы действуешь.\n— «Миль Попс, ах как вкусно, ням-ням-ням…»\n— Таня! Я сейчас ремень достану!\n— Ну что я могу поделать! Весь день эта реклама вертится в голове.",
        "enantiosemic_word": "вертится в голове",
        "option_a": "Я, наверное, и умирать буду, всё равно «Миль Попс» спою.",
        "option_b": "Я, наверное, и умирать буду, но не вспомню.",
        "option_c": "Как её найти? Господи.",
        "option_d": "Как её там? Господи.",
        "option_e": "Уже который день не могу от неё избавиться, словно проклятие наложили.",
        "option_f": "Уже который день вспоминаю, а все равно весело."
      },
      "outputs": "А; Е",
      "meta": {
        "id": 510,
        "type": 1
      }
    }
```

### Prompt Creation

20 prompts were prepared for the task: 5 prompts for each of the 4 task types. They were distributed evenly across questions according to the “one question — one prompt” principle. Templates in curly braces in the prompt are filled from the fields inside the `inputs` field in each question.

### Dataset Creation

The dataset was created by expert annotators with linguistic training in two stages. First, a list of Russian enantiosemic units was collected, and it was checked that each unit can express opposite meanings in the same form depending on context. Then, multiple-choice tasks were created based on this list: dialogue or utterance contexts and answer options for the four task types.

Quality control was also performed by annotators with linguistic training. Each task was checked for the absence of ambiguity: the context had to support the target meaning and rule out the opposite interpretation of the enantiosemic element. Examples with unclear context, overlapping interpretations, or answer options that could be justified under different meanings were edited or excluded.

### Metrics

The following metrics are used for aggregated evaluation of model answers:

- **Exact match (EM)**: This metric computes the proportion of model answers that exactly match the correct answer. The reference answer is one or several letters from А, Б, В, Г, Д, Е; if several letters are correct, they are written in alphabetical order and separated by a semicolon and a space. The value ranges from 0 to 1.