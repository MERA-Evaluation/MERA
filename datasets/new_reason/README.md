# NewReason

## Task Description

NewReason is a Russian-language dataset for evaluating a model's ability to solve short reasoning tasks with multiple-choice answers. Each example contains a textual task with a blank marked as `_ _ _` and a set of answer options labeled with Russian letters. The model must select one or more suitable options and return the answer letters in the required format.

The dataset tests whether a model can handle logical traps, condition substitutions, changes in quantities and qualitative properties, and disrupted reasoning chains.

Evaluated skills: Reasoning, Critical Thinking, Deductive Reasoning, Inductive Reasoning, Abductive Reasoning, Analogical Reasoning, Cause-and-Effect Reasoning, Decompositional Reasoning, Case-based Reasoning, Value Reasoning.

Contributors: Denis Shevelev, Alexander Kharitonov, Alexander Astafurov

## Motivation

The task is designed for evaluating Russian-language generative models in cases where solving requires checking the task conditions rather than only recognizing a familiar pattern.

NewReason helps analyze robustness to shallow templates: tasks that look similar on the surface may require different reasoning strategies, so the score reflects how well a model preserves the structure of the condition and follows the required answer format.

### Limitations

The dataset is not intended for evaluating models that do not support Russian or cannot generate answers in the required textual format. The task also does not measure broad mathematical, encyclopedic, or domain-specific knowledge: the examples focus on local reasoning over the provided condition and answer options.

### Validity

Task validity is supported by the controlled format: each example contains an explicit condition, answer options, and a task type in the metadata. Different example types test whether the model preserves the original conditions, detects substitutions, and selects an answer after analyzing the specific formulation rather than relying on a familiar template.

## Dataset Description

The dataset consists of two files with the same structure:

- `test.json` — **372** test examples with ids **1–372**
- `shots.json` — **16** few-shot examples with ids **373–388**

Both files have the form `{"access": "private", "data": [...]}`. Each item contains an `instruction` template, `inputs`, `outputs`, and `meta`.

Each example uses nine answer-option fields (`option_a`–`option_i`). Unused options are stored as empty strings. The `instruction` field contains placeholders only for non-empty options: the "Answer options" block includes exactly as many lines as there are non-empty `option_*` fields in that example. The "Answer format" block explicitly states that answer letters must be uppercase Russian letters from the list (А, Б, В, Г, Д, Е, Ё, Ж, З); in each example, this list contains only the letters that correspond to the non-empty options. One or several options may be correct; multiple correct answers are written in alphabetical order and separated by a semicolon followed by a space.

### Distribution by Task Type

Each example is assigned one of four types in the `task_type` field:

| `task_type` | Task type | Count | Share |
|-------------|-----------|------:|------:|
| `А` | Type А "Task or trick" | 184 | 49.5% |
| `Б` | Type Б "Sequence of steps" | 110 | 29.6% |
| `В` | Type В "Smart answer" | 63 | 16.9% |
| `Г` | Type Г "Matching" | 15 | 4.0% |

The few-shot split contains 4 examples per task type.

### Data Fields

Each example contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model.
- `inputs` — input data that forms the task:
    - `question` [str] — task text with the `_ _ _` blank;
    - `option_a` [str] — answer option А;
    - `option_b` [str] — answer option Б;
    - `option_c` [str] — answer option В;
    - `option_d` [str] — answer option Г;
    - `option_e` [str] — answer option Д;
    - `option_f` [str] — answer option Е;
    - `option_g` [str] — answer option Ё;
    - `option_h` [str] — answer option Ж;
    - `option_i` [str] — answer option З.
- `outputs` [str] — a string containing the letter or letters of the correct answer, written in alphabetical order and separated by a semicolon and a space.
- `meta` — metadata:
    - `id` [int] — example number;
    - `task_type` [str] — task type: А, Б, В, or Г.

### Data Instance

```json
{
    "instruction": "Помогите мне, пожалуйста.\n\nЗадача:\nПрочитайте текст с пропуском, обозначенным как ’_ _ _’. Ознакомьтесь с вариантами заполнения пропуска, и выберите все подходящие — те, что логично отвечают на вопрос в тексте (в том числе в скобках, если он есть).\n\nДля решения нужно оценить ситуацию, применить логическое рассуждение и, при необходимости, элементарную арифметику или сообразительность.\n\nФормат ответа:\nПоследняя строка ответа должна иметь вид:\n\nОтвет: <буква или буквы>\n\nЕсли подходящих вариантов несколько, перечислите буквы в алфавитном порядке через точку с запятой и пробел. \n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е, Ё, Ж, З).\n\nТекст:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\nЁ. {option_g}\nЖ. {option_h}\nЗ. {option_i}\n\nОтвет:",
    "inputs": {
        "question": "Матери 55 лет. У неё три дочери. Первой 15 лет, второй — 7, а третьей — 21 год. Через сколько лет возраст матери будет равен сумме лет её дочерей? Верный ответ: ’_ _ _’.",
        "option_a": "Никогда",
        "option_b": "3,5 года",
        "option_c": "3 года",
        "option_d": "8 лет",
        "option_e": "4 года",
        "option_f": "5 лет",
        "option_g": "6 лет",
        "option_h": "12 лет",
        "option_i": "15 лет"
    },
    "outputs": "Ё",
    "meta": {
        "id": 373,
        "task_type": "А"
    }
}
```

### Prompts

Five prompt variants were prepared and distributed evenly across examples on a one-example–one-prompt basis. For each example, the prompt is adapted to the number of non-empty answer options: the "Answer options" block retains only the corresponding lines, and the "Answer format" block lists the shortened set of allowed letters (for example, six options yield "А, Б, В, Г, Д, Е"). Template placeholders in curly braces are filled from the fields inside `inputs` for each question.

## Dataset Creation

The dataset is built as a collection of short Russian-language reasoning tasks with controlled transformations. It uses original tasks and variants where surface features or the logical structure are changed: conditions are inverted, numbers are modified, units and parameters are replaced, objects are substituted, the required solution type changes, or the order of reasoning steps is disrupted.

Examples are annotated with four task types: А — "Task or trick", Б — "Sequence of steps", В — "Smart answer", and Г — "Matching". These fields allow examples to be filtered by task type and results to be analyzed by group.

## Evaluation

### Metrics

The following metrics are used for aggregated evaluation:

- **Exact match (EM)**: the proportion of model answers that exactly match the reference answer after extracting the string following the `Answer` marker.
- **LLM judge score**: a metric where an LLM judge compares the model answer with the reference answer for each example. If the answer is judged correct, the example receives 1; if it is judged incorrect, it receives 0. The final metric value is the proportion of examples that received 1.
