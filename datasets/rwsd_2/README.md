# RWSDNew


## Task Description

**RWSDNew** is a Russian-language dataset for evaluating reference resolution, syntactic ambiguity resolution, and robustness to texts with non-standard but grammatically interpretable vocabulary. The dataset builds on the [Russian Winograd Schema Dataset](https://mera.a-ai.ru/ru/text/tasks/1): one part preserves the idea of Winograd-style reference resolution, while the other part is based on "Glokaya kuzdra"-style texts, where meaning must be inferred from grammatical, derivational, and contextual cues rather than from familiar lexical meanings.

Each example contains a textual context, a question, a reference, and answer options. The model must select one option that the reference refers to, or one option that correctly answers the question based on the context.

Tested model skills: Russian language proficiency, Coreference resolution, Syntactic ambiguity resolution, Morphological reasoning, Contextual reasoning

Authors: Denis Shevelev, Alexander Kharitonov, Alexander Astafurov


## Motivation

The original RWSD task evaluates whether a model can resolve referential ambiguity in short Russian texts using syntax, reasoning, and world knowledge. This dataset uses that idea as one of its foundations and adds examples with potential Russian words, nonce words, and artificial or partly artificial texts where ordinary dictionary semantics is not a reliable shortcut.

This setting tests whether a model can use grammatical structure, morphological markers, derivational patterns, and local context. In texts such as "Syapala kalusha po napushke..." or "Glokaya kuzdra", unknown words are not random strings: their endings, suffixes, prefixes, and syntactic positions carry grammatical information. Humans can infer part of the meaning even without knowing a lexical root. For language models, this is a targeted stress test: rewriting or normalizing such texts may remove exactly the cues that make the task solvable.

### Limitations

The dataset is intended for Russian-language text models and does not evaluate multimodal abilities, dialogue safety, or broad factual knowledge. Results should be interpreted as controlled multiple-choice performance on reference resolution and contextual understanding, not as a full evaluation of open-ended reasoning over nonce words. In the `kusdra` subset, some texts intentionally contain non-standard words, occasionalisms, and language play; low performance may reflect weak morphological analysis or poor robustness to unusual linguistic material.

### Validity

The examples are designed so that the correct answer can be recovered from the context and word forms. In the `rwsd` subset, the model must match a reference to one of the candidate options, including cases where the reference points not only to a word but also to the participant of an action. In the `kusdra` subset, the model answers questions over texts where part of the information is expressed through non-standard words and grammatical constructions. Combining the two subsets reduces the chance that a model can solve the task only by memorizing patterns from the original RWSD setting.


## Dataset Description

The dataset contains **425** examples:

| `task_type` | Task type | Count | Share |
|-------------|-----------|------:|------:|
| `rwsd` | Reference resolution in regular Russian contexts | 212 | 49.9% |
| `kusdra` | Questions over texts with non-standard vocabulary and language play | 213 | 50.1% |

Distribution by context type:

| `context_type` | Count |
|----------------|------:|
| `текст` | 317 |
| `диалог` | 108 |

The number of answer options ranges from 4 to 9.

### Data Fields

Each example contains the following fields:

- `instruction` [str] - a string containing the task formulation for the language model.
- `inputs` - input data forming the task:
    - `question` [str] - the question about the context;
    - `context` [str] - the source text, dialogue, or fragment;
    - `reference` [str] - the reference to be matched with an answer option; in the `kusdra` subset this field may contain a service value when the question is not a reference-resolution task;
    - `option_a` [str] - answer option А;
    - `option_b` [str] - answer option Б;
    - `option_c` [str] - answer option В;
    - `option_d` [str] - answer option Г;
    - `option_e` [str] - answer option Д;
    - `option_f` [str] - answer option Е;
    - `option_g` [str] - answer option Ё;
    - `option_h` [str] - answer option Ж;
    - `option_i` [str] - answer option З.
- `outputs` [str] - the correct answer: one letter from `А, Б, В, Г, Д, Е, Ё, Ж, З`.
- `meta` - metadata:
    - `id` [int] - example number;
    - `task_type` [str] - task type: `rwsd` or `kusdra`;
    - `context_type` [str] - context type: `текст` or `диалог`.

### Data Example

#### `rwsd` Example

```json
{
    "instruction": "Помогите мне, пожалуйста.\n\nЗадача:\nЕсть контекст, референс и несколько вариантов ответа. Нужно определить, к каким из вариантов относится референс.\n\nВ некоторых примерах референс может быть связан не с отдельным словом, а с тем, кто выполняет действие. Если рядом с глаголом встречается несколько существительных, нужно понять, какое из них имеется в виду.\n\nКонтекст:\n{context}\n\nРеференс:\n{reference}\n\nФормат ответа:\nПоследняя строка ответа должна иметь вид:\n\nОтвет: <буква>\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\nЁ. {option_g}",
    "inputs": {
        "question": "Относится ли референс к данному слову среди всех вариантов в указанном контексте?",
        "context": "Дни летели за днями, птицы за птицами, депеши за депешами, а самолёты Аркадия Петровича Авастюрова никак не желали подниматься с полей в небеса, и всёшеньки! Они понуро жались к стерне и что-то надрывно бурчали про себя время от времени, но и только.",
        "reference": "они",
        "option_a": "дни",
        "option_b": "птицы",
        "option_c": "депеши",
        "option_d": "самолёты",
        "option_e": "полей",
        "option_f": "небеса",
        "option_g": "всёшеньки",
        "option_h": "",
        "option_i": ""
    },
    "outputs": "Г",
    "meta": {
        "id": 1,
        "context_type": "текст",
        "task_type": "rwsd"
    }
}
```

#### `kusdra` Example

```json
{
    "instruction": "Помогите мне, пожалуйста.\n\nЗадача:\nЕсть текст и несколько вариантов ответа. Нужно внимательно прочитать текст, ответить на вопрос и выбрать подходящий вариант ответа.\n\nТекст:\n{context}\n\nФормат ответа:\nПоследняя строка ответа должна иметь вид:\n\nОтвет: <буква>\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}",
    "inputs": {
        "question": "Сколько пасиков в пусике?",
        "context": "241. Мряка крючит на пасики и лениво друсит пусики на тасики, которые являются конечной и единственной величиной любой друськи пусиков. На друську одного пусика Мряка тратит полдолгика. Сколько долгиков истратит Мряка на друську шести пусиков?",
        "reference": "‘-‘",
        "option_a": "Три",
        "option_b": "Два",
        "option_c": "Нисколько",
        "option_d": "Невозможно определить",
        "option_e": "",
        "option_f": "",
        "option_g": "",
        "option_h": "",
        "option_i": ""
    },
    "outputs": "В",
    "meta": {
        "id": 4,
        "context_type": "текст",
        "task_type": "kusdra"
    }
}
```

### Prompts

The task uses 10 prompts: 5 prompts for the `rwsd` type and 5 prompts for the `kusdra` type. The prompts are distributed across examples and follow the SAP structure: they explicitly separate the task formulation, context, reference or question, answer format, and answer options. All prompts require a line in the form `Ответ: <letter>`, where the letter is selected from the available options.

### Dataset Creation

The dataset was created from two related linguistic settings. The first group of examples follows the Winograd Schema idea: a text contains an ambiguous reference that must be matched to one of the candidates. The second group adds texts and questions inspired by "Glokaya kuzdra"-style tasks: grammatical structure and derivational cues make it possible to infer meaning even when ordinary lexical semantics is unavailable.

The prompts keep a unified answer format so that evaluation does not depend on long free-form explanations. For each example, answer options are listed explicitly.


## Evaluation

### Metrics

The following metrics are used for aggregated evaluation:

- **Exact match (EM)**: the proportion of model answers that exactly match the reference letter after extracting the string following the `Ответ:` marker.
- **LLM judge score**: a metric where an LLM judge compares the model answer with the reference answer for each example. If the answer is judged correct, the example receives 1; if it is judged incorrect, it receives 0. The final metric value is the proportion of examples that received 1.
