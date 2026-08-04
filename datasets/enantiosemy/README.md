# Enantiosemy

## Task Description

The **Enantiosemy** dataset evaluates whether language models can understand words that, in the same form, can have opposite meanings depending on context. The benchmark tests the model's ability to correctly interpret enantiosemic units in one of four multiple-choice settings: selecting a suitable continuation, selecting an unsuitable continuation, identifying the meaning used in context, or identifying the meaning not used in context. The expected answer is one or several option letters.

Tested model skills: Russian language proficiency, Contextual disambiguation, Linguistic-aware reasoning

Authors: Denis Shevelev, Alexander Astafurov, Alexander Kharitonov

## Motivation

Russian has a phenomenon known as **enantiosemy** (also called contronymy or intralexical antonymy), where the same word or phrase in the same form can have opposite meanings depending on the context in which they occur. For example, the verb "прослушать" can mean both "to listen attentively" and "to fail to hear, to miss." For humans, understanding such words is usually not difficult, but for language models this task remains non-trivial: the model must not only recognize the enantiosemic unit, but also determine which of its opposite meanings is used in the given context, and then select a semantically consistent continuation.

### Limitations

The dataset is intended for Russian-language text models and does not evaluate general literacy, dialogue safety, or factual knowledge outside the provided context. The task focuses on controlled multiple-choice selection, so the results should not be interpreted as a complete evaluation of a model's ability to handle all cases of enantiosemy in open-ended generation.

### Validity

The examples are designed so that the intended meaning of the enantiosemic unit can be recovered from the local context, while the alternative meaning is ruled out by the surrounding utterances. The four task types test both positive and negative recognition of the contextual meaning, which reduces the chance that a model can solve the task only through superficial continuation matching.

## Dataset Description

The dataset contains **506** examples and **157** unique normalized enantiosemic words and expressions.

### Distribution by Task Type

Each example is assigned one of four types in the `type` field:

| `type` | Task type | Count | Share |
|--------|-----------|------:|------:|
| `1` | Select the **correct** continuation of the final utterance | 290 | 57.3% |
| `2` | Select the **incorrect** / unsuitable continuation | 188 | 37.2% |
| `3` | Identify the meaning **in which the word is used** | 18 | 3.6% |
| `4` | Identify the meaning **in which the word is not used** | 10 | 2.0% |

The answer format is multiple choice. Options are labeled А, Б, В, Г, Д, Е; one or several options may be correct. When multiple options are correct, the answer letters are listed in alphabetical order and separated by a semicolon followed by a space.

Each example stores answer options in the six fields `option_a`–`option_f`. The `instruction` field uses placeholders for these fields in the "Варианты ответа" block and lists the allowed uppercase Russian answer letters (А, Б, В, Г, Д, Е) in the "Формат ответа" block.


### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model

- `inputs` — input data that forms the task:
    - `text` [str] — a dialogue or utterance containing an enantiosemic element
    - `enantiosemic_word` [str] — the enantiosemic word in its initial form
    - `option_a` [str] — the first answer option
    - `option_b` [str] — the second answer option
    - `option_c` [str] — the third answer option
    - `option_d` [str] — the fourth answer option
    - `option_e` [str] — the fifth answer option
    - `option_f` [str] — the sixth answer option

- `outputs` [str] — a string containing the letter or letters of the correct answer, written in alphabetical order and separated by a semicolon and a space, for example: А; Б; В; Г; Д; Е

- `meta` — metadata:
    - `id` [int] — example number
    - `type` [int] — numeric task type, see the task type table above

### Data Example

```json
{
    "instruction": "Помогите, пожалуйста, решить следующую задачу.\n\nЗадача:\nОпределите вариант(ы), который(ые) не мог(ли) бы стать завершающей фразой последней реплики в тексте из-за противоречия заложенному в тексте смыслу энантиосемического слова.\n\nКонтекст:\nВ тексте есть энантиосемическое слово — лексическая единица, способная в одной форме выражать противоположные смыслы.\n\nФормат ответа:\nЗапишите ответ в формате:\n\nОтвет: <выбранная буква или буквы>\n\nУкажите одну букву или несколько букв через точку с запятой и пробел в алфавитном порядке, если неверных вариантов больше одного.\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е).\n\nТекст:\n{text}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nЭнантиосемическое слово:\n{enantiosemic_word}",
    "inputs": {
        "enantiosemic_word": "стишки",
        "option_a": "Простенькие, но такие душевные, такие искренние!",
        "option_b": "Простенькие, неказистые. Одним словом – ужас. Не выйдет из него поэта.",
        "option_c": "Вот чувствуется в них любовь, что ни скажи.",
        "option_d": "Вот чувствуется, что старался.",
        "option_e": "Пусть простые, зато от сердца — такие и запоминаются.",
        "option_f": "Слабенькие, конечно, но мальчишка старался.",
        "text": "— Девочки, ну как малыши вас с днём матери поздравили? Мне вот Саша цветы из бумаги подарила.\n— Ой, мне Ванюша портрет нарисовал. Красота!\n— А мой Лёшка написал замечательные стишки. Такие, знаете…"
    },
    "outputs": "Б",
    "meta": {
        "id": 507,
        "type": 2
    }
}
```

### Prompt Creation

20 prompts were prepared for the task: 5 prompts for each of the 4 task types. They are distributed evenly across examples using a one-example–one-prompt scheme within each type. Template placeholders in curly braces are filled from the fields inside `inputs` for each question. Each prompt template uses a consistent form of address (informal ты or formal вы).

### Dataset Creation

The dataset was created in two stages by linguists invited as expert annotators in collaboration with the AGI NLP team. First, the team developed the task methodology and compiled a list of Russian enantiosemic units. The list includes both individual words and multiword expressions whose identical form can express opposite meanings depending on context. The four task types were selected to test two complementary operations—choosing a continuation and identifying a contextual meaning—in both positive and negative formulations.

The invited linguists then created dialogue or utterance contexts and multiple-choice answer options from this list. The AGI NLP team coordinated the methodology and data-production process, reviewed the resulting structure, and prepared the dataset for inclusion in MERA.

Quality control was also performed by annotators with linguistic training. Each task was checked for the absence of ambiguity: the context had to support the target meaning and rule out the opposite interpretation of the enantiosemic element. Examples with unclear context, overlapping interpretations, or answer options that could be justified under different meanings were edited or excluded.

### Metrics

The following metrics are used for aggregated evaluation of model answers:

- **Exact match (EM)**: This metric computes the proportion of model answers that exactly match the correct answer. The reference answer is one or several letters from А, Б, В, Г, Д, Е; if several letters are correct, they are written in alphabetical order and separated by a semicolon and a space. The value ranges from 0 to 1.

- **LLM judge score**: an LLM judge compares the model answer with the reference answer and evaluates its correctness and completeness. A fully correct and complete answer receives `1`; a partially correct or incomplete answer containing an essential part of the correct answer receives `0.5`; an incorrect or contradictory answer, or one that does not contain the correct answer, receives `0`. The final metric value is the mean score across all examples.