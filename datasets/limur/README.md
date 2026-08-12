# LIMUR


## Task Description

**LIMUR** is a Russian-language dataset for evaluating reference resolution, syntactic ambiguity resolution, and robustness to texts with non-standard but grammatically interpretable vocabulary. It contains two task types. The first follows the [Russian Winograd Schema Dataset](https://mera.a-ai.ru/ru/text/tasks/1) setting and tests Winograd-style reference resolution. The second tests whether meaning can be recovered from grammatical, derivational, and contextual cues independently of familiar lexical meanings. A well-known example of this setting is Academician L. V. Shcherba's phrase "Glokaya kusdra shteko budlanula bokra i kurdyachit bokryonka."

The model receives a textual context, a question, and answer options. Examples of type `rwsd` also provide a reference to be matched with one option; examples of type `kusdra` require the option that correctly answers the contextual question.

Tested model skills: Russian language proficiency, Coreference resolution, Syntactic ambiguity resolution, Lexical ambiguity resolution, Morphological reasoning, Contextual reasoning.

Authors: Denis Shevelev, Alexander Kharitonov, Alexander Astafurov


## Motivation

The original RWSD task evaluates whether a model can resolve referential ambiguity in short Russian texts using syntax, reasoning, and world knowledge. This dataset uses that idea as one of its foundations and adds examples with potential Russian words, nonce words, and artificial or partly artificial texts where ordinary dictionary semantics is not a reliable shortcut.

This setting tests whether a model reasons about relations between participants and events instead of relying on surface lexical patterns. Answering requires combining grammatical structure, morphological markers, derivational patterns, and local context. In texts such as "Syapala kalusha po napushke..." or "Glokaya kusdra", unknown words are not random strings: their endings, suffixes, prefixes, and syntactic positions carry grammatical information. Humans can infer part of the meaning even without knowing a lexical root. For language models, this is a targeted reasoning test under incomplete lexical information: rewriting or normalizing such texts may remove exactly the cues that make the task solvable.

### Limitations

The dataset is intended for Russian-language text models and does not evaluate general literacy, dialogue safety, or broad factual knowledge. Results should be interpreted as controlled multiple-choice performance on reference resolution and contextual understanding, not as a full evaluation of open-ended reasoning over nonce words. In the `kusdra` subset, some texts intentionally contain non-standard words, occasionalisms, and language play; low performance may reflect weak morphological analysis or poor robustness to unusual linguistic material.


## Dataset Description

The dataset contains **423** examples:

| `task_type` | Task type | Count | Share |
|-------------|-----------|------:|------:|
| `rwsd` | Reference resolution in regular Russian contexts | 211 | 49.9% |
| `kusdra` | Questions over texts with non-standard vocabulary and language play | 212 | 50.1% |

Distribution by context type:

| `context_type` | Count |
|----------------|------:|
| `текст` | 315 |
| `диалог` | 108 |

The number of answer options ranges from 4 to 9. The `instruction` field contains placeholders only for non-empty options: the "Варианты ответа" block includes exactly as many lines as there are non-empty `option_*` fields in that example. The "Формат ответа" block states that answer letters may only be uppercase Russian letters from the list (А, Б, В, Г, Д, Е, Ё, Ж, З); in each example this list is shortened to match the non-empty options.

### Data Fields

Each example contains the following fields:

- `instruction` [str] — a string containing the task formulation for the language model.
- `inputs` — input data forming the task:
    - `question` [str] — the question about the context;
    - `context` [str] — the source text, dialogue, or fragment;
    - `reference` [str] — the reference to be matched with an answer option in `rwsd`; the field is unused in `kusdra` and contains the `‘-‘` placeholder in the current data;
    - `option_a` [str] — answer option А;
    - `option_b` [str] — answer option Б;
    - `option_c` [str] — answer option В;
    - `option_d` [str] — answer option Г;
    - `option_e` [str] — answer option Д;
    - `option_f` [str] — answer option Е;
    - `option_g` [str] — answer option Ё;
    - `option_h` [str] — answer option Ж;
    - `option_i` [str] — answer option З.
- `outputs` [str] — the correct answer: one letter from `А, Б, В, Г, Д, Е, Ё, Ж, З`.
- `meta` — metadata:
    - `id` [int] — example number;
    - `task_type` [str] — task type: `rwsd` or `kusdra`;
    - `context_type` [str] — context type: `текст` or `диалог`.

### Data Example

#### `rwsd` Example

```json
{
    "instruction": "Задача:\nПо контексту и референсу нужно определить, к каким вариантам относится референс.\n\nРеференс может указывать как на отдельное слово, так и на того, кто выполняет действие. В последнем случае нужно определить, какое из существительных имеется в виду.\n\nКонтекст:\n{context}\n\nРеференс:\n{reference}\n\nФормат ответа:\nВерните результат в формате\n\nОтвет: <буква>\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е, Ё).\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\nЁ. {option_g}\n",
    "inputs": {
        "question": "К какому из предложенных вариантов ответа относится слово референс в данном контексте?",
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
        "id": 426,
        "context_type": "текст",
        "task_type": "rwsd"
    }
}
```

#### `kusdra` Example

```json
{
    "instruction": "Внимательно прочитай текст и определи ответ.\n\nЗадача:\nОтветь на вопрос по тексту и выбери правильный вариант.\n\nТекст:\n{context}\n\nФормат ответа:\nВыведи только\n\nОтвет: <буква>\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г).\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\n",
    "inputs": {
        "question": "Сколько пасиков в пусике?",
        "context": "241. Мряка крючит на пасики и лениво друсит пусики на тасики, которые являются конечной и единственной величиной любой друськи пусиков. На друську одного пусика Мряка тратит полдолгика. Сколько долгиков истратит Мряка на друську восьми пусиков?",
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
        "id": 429,
        "context_type": "текст",
        "task_type": "kusdra"
    }
}
```

### Prompts

The task uses 10 prompts: 5 prompts for the `rwsd` type and 5 prompts for the `kusdra` type. The prompts are distributed evenly across examples within each task type and follow the SAP structure: they explicitly separate the task formulation, context, reference or question, answer format, and answer options. For each example the prompt is adapted to the number of non-empty answer options. All prompts require a line in the form `Ответ: <letter>`, where the letter is selected from the available options. Each prompt template uses a consistent form of address (informal ты or formal вы).

### Dataset Creation

The dataset was created from two related linguistic settings. The first group contains original reference-resolution examples written by the dataset authors in the Winograd Schema style: a text contains an ambiguous reference that must be matched to one of the candidates. The second group contains original questions and tasks based on texts available online and inspired by "Glokaya kusdra"-style language puzzles: grammatical structure and derivational cues make it possible to infer meaning even when ordinary lexical semantics is unavailable.

The principal literary and educational sources are Academician L. V. Shcherba's "Glokaya kusdra..." phrase; Lyudmila Petrushevskaya's *Puski byatye*; Lewis Carroll's "Jabberwocky" in Russian translations and adaptations, including Dina Orlovskaya's *Barmaglot*, D. Manin's *Ubeshchur*, A. Shcherbakov's *Tarbormoshki*, and Leonid Yakhnin's *Zmeegrych*; Velimir Khlebnikov's poems "Incantation by Laughter," "Siyayushchaya volza...", "Nem lukaet lukom nemnym...", "I ya svirel v svoyu svirel...", and the prologue "Chernotvorskie vestuchki"; Vladimir Mayakovsky's poems "Night" and "And Could You?"; and Grigory Oster's educational book *Physics: An Unvisual Aid*. The dataset authors wrote the questions and answer options; no generative models were used to create the data.

The prompts keep a unified answer format so that evaluation does not depend on long free-form explanations. For each example, answer options are listed explicitly.


## Evaluation

### Metrics

The following metrics are used for aggregated evaluation:

- **Exact match (EM)**: the proportion of model answers that exactly match the reference letter after extracting the string following the `Ответ:` marker. The score ranges from 0 to 1: 0 means no exact matches, while 1 means every example matched.
- **LLM judge score**: an LLM judge compares the model answer with the reference answer and evaluates its correctness and completeness. A fully correct and complete answer receives `1`; a partially correct or incomplete answer containing an essential part of the correct answer receives `0.5`; an incorrect or contradictory answer, or one that does not contain the correct answer, receives `0`. The final metric value is the mean score across all examples.
