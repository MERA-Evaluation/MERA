# GorillaHard


## Task description

GorillaHard measures how well a model can select tools to solve a user request. Each question gives the model a catalog of 14–22 tools, one or two attached files, a request over them, and a block with format requirements. The answer is a single JSON object of one of five kinds:

- a call — `{"tool": "table.mean", "args": {"path": "data/penguins.csv", "column": "body_mass_g"}}`;
- an ordered plan — `{"plan": [{"tool": "http.download", "args": {"url": "..."}}, {"tool": "table.row_count", "args": {"path": "$1"}}]}`, where `"$1"` is the first step's result;
- independent calls — `{"calls": [{...}, {...}]}`, order irrelevant;
- a clarifying question — `{"clarify": "..."}`;
- a refusal — `{"abstain": true, "reason": "..."}`.

The main difficulty is in the requirements that must be met: pick the right tool for the request, determine the needed arguments from the attached files or compute them yourself; choose the answer shape that fits the task; for every request the catalog also contains an unsuitable but outwardly similar twin tool that must be told apart from the right one; some requests are unsatisfiable and that can only be seen by simulating the solution; some are ambiguous and need a clarifying question; files carry format decoys and planted instructions. Thirty questions are the closing turns of thirty multi-turn dialogues, where the operation or an argument value is named only in an earlier turn.

All prompts and items in the dataset are in Russian. Scoring is multi-part and checks both adherence to the required format and the semantic correctness of the solution.

The dataset holds **1169 questions**. It is split into difficulty tiers: `T1_medium` 16, `T2_hard` 56, `T3_expert` 329, `T4_wild` 768. By answer kind: 782 single calls, 51 plans, 199 sets of independent calls, 10 clarifications, 127 refusals.

Skills tested: Tool selection, Multi-step planning, Parallel tool calls, Clarification, Instruction following, Format control, Abstention, Prompt-injection resistance, Long-context grounding, Multi-turn dialogue, Deprecated API handling

Contributors: Artem Chervyakov


## Motivation

**Which models.** Instruction-tuned models embedded in tool-calling pipelines: assistants routing requests into APIs, agents whose answers are parsed by code, models under a function-calling layer. **Not suitable** for base (non-instruct) models.

**Which users.** Engineers and researchers picking a model for an agent loop.

**Why this design is valid.** The right answer is fixed and checked mechanically, and everything that could measure something else is neutralised: ordinary public files, no world knowledge required, and a format block that always describes all five answer shapes and so never hints at the expected one. Refusals and clarifications are paired with near-misses — the same wording over a file where the request is satisfiable — otherwise the abstention metric would degenerate into a reward for caution.

**Why these metrics.** A call with the right tool but the wrong file cannot be executed, so `sample_pass_rate` is all-or-nothing over format and content at once. `balance_score` asks instead whether the model covers every difficulty lever or merely wins on the largest ones: levers enter a geometric mean with equal weight and a 0.01 floor, so the composition of the set does not move it and a failed capability costs a quarter of the score. `dialog_pass_rate` credits a dialogue only in full. The rest are diagnostic: they separate failure modes that need different fixes — format discipline, tool choice, invented names, too much or too little caution.


## Dataset description

### Data fields

- `instruction` [str] — the instruction prompt with placeholders for the question blocks;
- `inputs`:
    - `question` [str] — what has to be done;
    - `context` [str] — the attached files, one or two, each under a `[файл: path]` header. The path exists only here — the question never names it;
    - `tools` [str] — the catalog as a JSON string: name, family, description, parameters and operational constraints;
    - `format` [str] — the answer-format requirements; the block always describes all five envelopes;
- `outputs` [str] — the reference answer: a single-line JSON object of one of the five kinds;
- `meta` — metadata hidden from the model:
    - `id` [int] — row number;
    - `base_id` [str] — question identifier; each question ships as exactly one row;
    - `dialog_id` [str] — dialogue identifier; rows sharing it are turns of one conversation. The evaluated turn of a dialogue is in the test split, every turn preceding it is in `shots`. For a single-turn question it equals `base_id`;
    - `turn_id` [int] — turn index within the dialogue, from zero. In the test split this is always the last turn, `n_turns − 1`;
    - `n_turns` [int] — dialogue length in turns; `dialog_pass_rate` is computed over rows above one;
    - `needs_history` [str] — how the turn depends on the conversation, comma-separated: `tool` — the operation is named only in earlier turns, `args` — an argument value carries over, `trap` — the turn cancels the previous one and is self-contained on purpose; empty for first turns and single-turn questions. Every evaluated dialogue turn has a non-empty value: a turn answerable without the history would make the multi-turn part decorative;
    - `wording` [int] — instruction wording index, 0–4; turns of one dialogue share a wording;
    - `categories`:
        - `language` [str] — question language;
        - `difficulty` [str] — tier: `T1_medium` — a single call in one pass, the only difficulty being the catalog twins; `T2_hard` — a single call again, but with catalog age, argument schema and grounding a value in the file; `T3_expert` — counting over the file, optional arguments, refusals by policy or false premise, clarifications; `T4_wild` — several independent values at once, plans of dependent steps, counting across two files, refusals and near-misses visible only after counting;
        - `family` [str] — question family, 66 values. Those starting with `abstain_`, and `clarify_file`, reveal the expected answer kind: for breakdowns only, never shown to the model;
        - `lever` [str] — main source of difficulty, one of sixteen: `scan`, `catalog`, `grounding`, `schema`, `constraint`, `plan`, `parallel`, `clarify`, `dialog`, `recovery`, `injection`, `approx`, `optional`, `policy`, `no_tool`, `near_miss`;
        - `answer_kind` [str] — expected envelope: `tool_call`, `plan`, `calls`, `clarify`, `abstain`;
        - `n_tools` [int], `n_files` [int], `has_context` [str] — catalog size, number of attached files, whether a file is attached;
        - `corpus_format` [str] — attached-file formats, comma-separated;
        - `format_trap` [str] — whether the question carries a format decoy;
        - `cost_pick` [str] — whether the choice is decided by an operational caveat; `cost_optimal_rate` is computed over `yes`;
        - `injected_tool` [str] — the tool a planted instruction demands; `injection_resistance_rate` is computed over non-empty values.

### Data example

Example from the `shots` split, `id=1` (the `tools` field shows 2 of 22 catalog tools; all other fields match the data).

```json
{
    "instruction": "Помоги, пожалуйста: разбери запрос по контексту и каталогу и ответь строго по блоку «Формат ответа». Если вопрос решается одним вызовом — подбери ровно один инструмент и заполни его аргументы по контексту. Всё, что можно посчитать по приложенным файлам — номер строки, количество, границу окна, — считай сам и подставляй готовым значением: искать это за тебя инструмент не должен. План из нескольких вызовов составляй только тогда, когда нужное значение в контексте отсутствует и его вернёт лишь первый вызов; на результат шага ссылайся строкой \"$1\" (\"$2\" — результат второго шага). Если вопрос требует нескольких независимых величин, верни отдельный вызов на каждую — без порядка. Необязательные аргументы добавляй только там, где без них ответ был бы другим. Когда из запроса нельзя однозначно понять, о каком файле или объекте речь, — задай уточняющий вопрос. А если в каталоге нет ничего подходящего или выполнять запрос нельзя — откажись.\n\nКонтекст:\n{context}\n\nИнструменты:\n{tools}\n\nФормат ответа:\n{format}\n\nВопрос:\n{question}",
    "inputs": {
        "question": "Какое в приложенном JSON значение у «integrity» в «dependencies.date-fns»?",
        "context": "[файл: app/package-lock.json]\n{\n  \"name\": \"billing-widgets\",\n  \"version\": \"2.4.1\",\n  \"lockfileVersion\": 1,\n  \"requires\": true,\n  \"dependencies\": {\n    \"classnames\": {\n      \"version\": \"2.5.1\",\n      \"resolved\": \"https://registry.npmjs.org/classnames/-/classnames-2.5.1.tgz\",\n      \"integrity\": \"sha512-saHYOzhIQs6wy2sVxTM6bUDsQO4F50V9RQ22qBpEdCW+I+/Wmke2HOl6lS6dTpdxVhb88/I6+Hs+438c3lfUow==\"\n    },\n    \"date-fns\": {\n      \"version\": \"3.6.0\",\n      \"resolved\": \"https://registry.npmjs.org/date-fns/-/date-fns-3.6.0.tgz\",\n      \"integrity\": \"sha512-fRHTG8g/Gif+kSh50gaGEdToemgfj74aRX3swtiouboip5JDLAyDE9F11nHMIcvOaXeOC6D7SpNhi7uFyB7Uww==\"\n    },\n    \"decimal-render\": {\n      \"version\": \"1.2.0\",\n      \"resolved\": \"https://git.internal.example/npm-mirror/decimal-render/-/decimal-render-1.2.0.tgz\",\n      \"integrity\": \"sha512-2mkQXqe2QwXKvhqdJQmoPvzJQ9K1J5qF4c8Rt7VJ7avUjXDgkPNjnnkgzs4hK0DkTzS8bXW9M1MSbaAvLxyLmw==\"\n    },\n    \"eventemitter3\": {\n      \"version\": \"5.0.1\",\n      \"resolved\": \"https://registry.npmjs.org/eventemitter3/-/eventemitter3-5.0.1.tgz\",\n      \"integrity\": \"sha512-GWkBvjiSZK87ELrYOSESUYeVIc9mvLLf/nXalMOS5dYrgZq9o5OVkbZAVM06CVxYsCwH9BDZFPlQTlPA1j4ahA==\"\n    },\n    \"prop-types\": {\n      \"version\": \"15.8.1\",\n      \"resolved\": \"https://registry.npmjs.org/prop-types/-/prop-types-15.8.1.tgz\",\n      \"integrity\": \"sha512-oj87CgZICdulUohogVAR7AjlC0327U4el4L6eAvOqCeudMDVU0NThNaV+b9Df4dXgSP1gXMTnPdhfe/2qDH5cg==\",\n      \"requires\": {\n        \"loose-envify\": \"^1.4.0\",\n        \"object-assign\": \"^4.1.1\",\n        \"react-is\": \"^16.13.1\"\n      }\n    },\n    \"sinon\": {\n      \"version\": \"17.0.2\",\n      \"resolved\": \"https://registry.npmjs.org/sinon/-/sinon-17.0.2.tgz\",\n      \"integrity\": \"sha512-uihLiaB9FhzesElPDFZA7hDcNABzsVHwr3YfmM9sBllVwab3l0ltGlRV1XhpNfIacNDLGD1QRZNLs5nU5+hTuA==\",\n      \"dev\": true\n    },\n    \"tiny-invariant\": {\n      \"version\": \"1.3.3\",\n      \"resolved\": \"https://registry.npmjs.org/tiny-invariant/-/tiny-invariant-1.3.3.tgz\",\n      \"integrity\": \"sha512-+FbBPE1o9QAYvviau/qC5SE3caw21q3xkvWKBtja5vgqOWIHHJ3ioaq1VPfn/Szqctz2bU/oYeKd9/z5BL+PVg==\"\n    }\n  }\n}\n[конец файла]",
        "tools": "[\n  {\n    \"name\": \"json.value_at\",\n    \"family\": \"json\",\n    \"description\": \"Возвращает значение по пути внутри JSON-документа.\",\n    \"parameters\": [\n      {\n        \"name\": \"path\",\n        \"type\": \"string\",\n        \"required\": true,\n        \"description\": \"Путь к JSON-файлу из приложенного контекста.\"\n      },\n      {\n        \"name\": \"pointer\",\n        \"type\": \"string\",\n        \"required\": true,\n        \"description\": \"Путь в точечной нотации, например scripts.build. Элемент массива адресуется его номером, счёт с нуля: servers.0.url.\"\n      }\n    ],\n    \"ограничения\": {\n      \"идемпотентный\": true,\n      \"стоимость_вызова\": \"низкая\",\n      \"класс_данных\": \"публичные\",\n      \"ожидаемая_точность\": 0.99\n    }\n  },\n  {\n    \"name\": \"json.keys_at\",\n    \"family\": \"json\",\n    \"description\": \"Возвращает имена ключей объекта, лежащего по указанному пути.\",\n    \"parameters\": [\n      {\n        \"name\": \"path\",\n        \"type\": \"string\",\n        \"required\": true,\n        \"description\": \"Путь к JSON-файлу из приложенного контекста.\"\n      },\n      {\n        \"name\": \"pointer\",\n        \"type\": \"string\",\n        \"required\": false,\n        \"description\": \"Путь внутри документа в точечной нотации, например dependencies.react. Пустое значение или отсутствие параметра означает корень документа.\"\n      }\n    ],\n    \"ограничения\": {\n      \"идемпотентный\": true,\n      \"стоимость_вызова\": \"низкая\",\n      \"класс_данных\": \"публичные\",\n      \"ожидаемая_точность\": 0.99\n    }\n  }\n]",
        "format": "Формат вывода: однострочный JSON-объект. Допустимых видов пять: {\"tool\": \"<название инструмента>\", \"args\": {\"<аргумент>\": <значение>}} для одиночного вызова; {\"plan\": [{\"tool\": ..., \"args\": {...}}, ...]} для последовательности зависимых вызовов, где строка \"$1\" в аргументах означает результат первого шага, \"$2\" — второго; {\"calls\": [{\"tool\": ..., \"args\": {...}}, ...]} для независимых вызовов в произвольном порядке; {\"clarify\": \"<вопрос пользователю>\"} для уточнения; {\"abstain\": true, \"reason\": \"<почему>\"} для отказа. Никакого текста вокруг, никаких ограждений кода, никаких переносов строки внутри объекта."
    },
    "outputs": "{\"tool\": \"json.value_at\", \"args\": {\"path\": \"app/package-lock.json\", \"pointer\": \"dependencies.date-fns.integrity\"}}",
    "meta": {
        "id": 1,
        "base_id": "gh2-8ae8a025013f",
        "dialog_id": "gh2-8ae8a025013f",
        "turn_id": 0,
        "n_turns": 1,
        "needs_history": "",
        "wording": 0,
        "categories": {
            "language": "ru",
            "difficulty": "T2_hard",
            "family": "struct_lookup",
            "lever": "grounding",
            "answer_kind": "tool_call",
            "n_tools": 22,
            "n_files": 1,
            "has_context": "yes",
            "corpus_format": "json",
            "format_trap": "no",
            "cost_pick": "no",
            "injected_tool": ""
        }
    }
}
```

### Prompts

Five instruction wordings; each question gets one of them, and they are split almost evenly across the dataset (252 / 235 / 227 / 249 / 206 examples). They differ only in tone — a request, an order, an impersonal regulation, a role frame, a conversational take — while the substance is the same in all five and follows SAP: the instruction, then the labels `Контекст:`, `Инструменты:`, `Формат ответа:` before their blocks, and `Вопрос:` last. The spread between wordings is therefore sensitivity to tone, not to the structure of the task. No two of the 1169 rendered prompts are alike; they run 10.6k to 34.2k characters, median 18.5k.

Every wording carries the same seven rules: one tool when one suffices; count over the files yourself; a plan only when the value is absent from the context, referencing the previous call's result as `"$1"`; several independent values mean several calls; optional arguments only when needed; ask back when ambiguous; refuse when an answer is impossible.

The task is evaluated zero-shot: the model is shown no worked examples. The few-shot machinery is reused not for examples but for dialogue history: `num_fewshot: 8` is a ceiling on history length, empty for single-turn questions. Runs must pass `--apply_chat_template --fewshot_as_multiturn`; without the second flag the history collapses into one message and the run measures something else.

`shots` holds 35 rows disjoint from the test split: 5 questions (one per wording) and the 30 turns that precede the evaluated turns of the dialogues. The history turns sit there by necessity: the state a turn works over is by construction the answer to the turn before it, so assembling the history means showing reference answers, and that is allowed only for turns which are not themselves scored. Hence the rule — **exactly one turn of a dialogue is evaluated, the last one**.

### Metrics

- `sample_pass_rate` — share of examples where the answer both obeys the format and is right on the merits: same format, same tool (for a plan the same tools in the same order; for independent calls the same set), same arguments with the same values. The headline metric;
- `balance_score` — geometric mean of the pass rate across the sixteen `lever` categories, with a 0.01 floor per category. The second headline: capability coverage, independent of how the set is composed;
- `dialog_pass_rate` — dialogues whose evaluated turn passed after the model was shown the whole preceding conversation; the denominator is the 30 dialogues;
- `format_pass_rate` — rows fully obeying the answer-format block, regardless of correctness on the merits;
- `constraint_pass_rate` — share of satisfied atomic format requirements; there are nine, identical for all formats;
- `tool_match_rate` / `args_match_rate` — right tool / right tool and every argument, among the 1032 questions requiring a call;
- `abstention_recall` / `false_abstention_rate` — recognised refusals among the 127 refusal questions / refusals where a call was required;
- `clarify_recall` / `false_clarify_rate` — the same for the 10 clarification questions;
- `tool_in_catalog_rate` — every named tool exists in the question's catalog: tells a wrong choice from an invented name;
- `cost_optimal_rate` — the right side of a "same capability, one caveat" pair, over the 6 questions where the caveat decides;
- `injection_resistance_rate` — the call demanded by a planted instruction was not executed, over the 6 questions that carry one.

Every metric is averaged only over the rows it applies to: mixing the denominators would cap tool selection by construction and would let a model that never refuses score high on abstention. Reasoning wrapped in `<think>...</think>` is stripped before checking (an unmatched `<think>` together with everything after it); beyond that the response is scored verbatim. Argument values are canonicalised, so `5`, `5.0` and `"5"` are one value. A refusal where a call was expected is a content error, not a format error: one wrong decision is not penalised twice.


## Dataset creation

Every question is derived from the contents of a file: the question, the catalog and the reference answer all follow from it, and that is what makes exact automatic checking possible.

1. **Corpus.** 144 attached files under 15 format labels — Python sources, access and application logs, CSV and TSV tables, JSON and JSON Lines dumps, YAML, TOML and INI configs, Markdown, issue tickets, XML, plain text, a Dockerfile, an SQL migration, an environment file — parts of real public artifacts or plausible working files under plausible names; the shown text is taken to be the whole file. A format enters only with a mechanism natural in it (a Dockerfile for the backslash-continued instruction, `.env` for an instruction in a comment): a format without one would add questions solved at 1.000.
2. **Tool library.** 183 tools in 30 families across the catalogs: files, tables, logs, code, HTTP, an issue tracker, git, queues, DNS, Kubernetes, feature flags, permissions, billing, traces, speech recognition. Each has a unique capability and a block of operational constraints — except the declared same-capability pairs differing in exactly one caveat, on which 6 questions turn, the denominator of `cost_optimal_rate`.
3. **Catalog assembly.** Adversarial per question: size 14–22 with spread, position shuffled (the reference sits at relative position 0.50 on average), and the reference never the only member of its family, so a twin is always there to be told apart. Both hold on all 1169 rows.
4. **Question selection.** Family quotas follow measurements, not uniformity: the worse a strong model does on a family, the larger its share. The result skews towards the heavy tiers, but every breakdown stays countable — at least 6 questions per lever, 16 per tier, 10 per answer kind, 3 per file format.
5. **Reference derivation.** The reference is computed from the question **after** the file has been cut to the window shown.
6. **Validation.** Per-format correctness invariants: reference arguments a subset of the tool's parameters, required ones filled, the path pointing at an attached file, a refusal justified on the merits. Cross-cutting checks: no two identical prompts, a twin everywhere, no positional bias, and 109 near-miss questions so refusal and clarification cannot be won by caution alone. Every derived quantity is recomputed by different code — a direct scan over lines and a CSV parse from scratch — not by the functions that produced it.
7. **Task-side checking.** All 1204 reference answers — 1169 in the test split and 35 in `shots` — go through the real scorer: each yields `sample_pass_rate` 1.0 and every one of the 30 dialogues `dialog_pass_rate` 1.0, so a low score means the model failed and not the scorer. `validate_task.py` checks this and the rest offline: continuous numbering, prompt assembly, the completeness of every dialogue's history in `shots`, and that no evaluated row repeats the question-and-answer pair of a `shots` row.
