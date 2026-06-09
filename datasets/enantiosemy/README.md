# Enantiosemy

## Task Description

The **Enantiosemy** dataset evaluates whether language models can understand words that, in the same surface form, can have opposite meanings depending on context. The benchmark tests the model's ability to correctly interpret enantiosemic units and select a continuation of an utterance that confirms the intended meaning has been understood.

Tested model skills: Russian language proficiency, Contextual disambiguation, Linguistic-aware reasoning

Authors: Denis Shevelev, Alexander Astafurov, Alexander Kharitonov

## Motivation

Russian has a phenomenon known as **enantiosemy** (also called contronymy or intralexical antonymy), where the same word or phrase in the same form can have opposite meanings depending on the context in which it appears. For example, the verb "прослушать" can mean both "to listen attentively" and "to miss, not hear." Such words are usually easy for humans to understand, but the task remains non-trivial for language models: the model must not only recognize the enantiosemic unit, but also determine which of its opposite meanings is used in the given context, and then select a semantically consistent continuation.

## Dataset Description

The dataset contains **506** examples and **159** unique enantiosemic words.

### Distribution by Task Type

Each example is assigned one of four types in the `type` field:

| `type` | Task type | Count | Share |
|--------|-----------|------:|------:|
| 1 | Select the **correct** continuation of the final utterance | 290 | 57.3% |
| 2 | Select the **incorrect** (unsuitable) continuation | 188 | 37.2% |
| 3 | Identify the meaning **in which the word is used** | 18 | 3.6% |
| 4 | Identify the meaning **in which the word is not used** | 10 | 2.0% |

The answer format is multiple choice among 6 options labeled А, Б, В, Г, Д, Е. One or several options may be correct. When multiple options are correct, the answer letters are listed in alphabetical order and separated by a semicolon followed by a space.

### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model

- `inputs` — input data that forms the task:
    - `text` [str] — a dialogue or utterance containing an enantiosemic element
    - `enantiosemic_word` [str] — the enantiosemic word in the infinitive
    - `option_a` [str] — the first continuation option for the final utterance
    - `option_b` [str] — the second continuation option
    - `option_c` [str] — the third continuation option
    - `option_d` [str] — the fourth continuation option
    - `option_e` [str] — the fifth continuation option
    - `option_f` [str] — the sixth continuation option

- `outputs` [str] — a string containing the letter or letters of the correct answer, written in alphabetical order separated by a semicolon and space, for example: А; Б; В; Г; Д; Е

- `meta` — metadata:
    - `id` [int] — example number
    - `type` [int] — task type (1–4, see the table above)

### Data Example

```json
{
      "instruction": "Задача:\nВыберите тезис, который наиболее логично продолжает заключительную реплику текста и отражает мысль персонажа — исходя из того значения, в котором употреблено энантиосемическое слово.\n\nКонтекст:\nТебе будет предложен текст — скорее всего, диалог или переписка. В нём встречается энантиосемическое слово: слово, которое в одной форме может иметь противоположные значения в зависимости от контекста.\n\nФормат ответа:\nВ качестве ответа укажите только одну строку вида:\n\nОтвет: БУКВА\n\nДопустимые значения БУКВА: А, Б, В, Г, Д, Е. Если верных вариантов несколько — перечислите буквы в алфавитном порядке через точку с запятой и пробел.\n\nТекст:
\n\n{text}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nЭнантиосемическое слово:\n{enantiosemic_word}",
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

20 prompts were prepared for the task: 5 prompts for each of the 4 task types. They were distributed evenly across questions using a one-question–one-prompt scheme. Template placeholders in curly braces in a prompt are filled from the fields inside `inputs` for each question.

### Dataset Creation

The dataset was created by expert linguists in two stages. First, a list of enantiosemic units in Russian was compiled. Then, tasks were built on the basis of this list. Each item was checked for ambiguity: the text must rule out interpreting the enantiosemic element in the opposite meaning.

### Metrics

The following metrics are used for aggregated evaluation of model answers:

- **Exact match (EM)**: This metric computes the proportion of model answers that exactly match the correct answer (a single letter А, Б, В, Г, Д, or Е). The value ranges from 0 to 1.
