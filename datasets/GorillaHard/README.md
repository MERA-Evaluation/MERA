# GorillaHard


## Task description

GorillaHard measures whether a model can turn a request into an executable decision over a tool catalog. Each question gives the model a catalog of 14–22 tools, one or two attached files, a request over them and a block stating the required answer format. The answer must be one JSON object and nothing else, of one of five kinds:

- a call — `{"tool": "table.mean", "args": {"path": "data/penguins.csv", "column": "body_mass_g"}}`;
- an ordered plan — `{"plan": [{"tool": "http.download", "args": {"url": "..."}}, {"tool": "table.row_count", "args": {"path": "$1"}}]}`, where `"$1"` is the first step's result;
- independent calls — `{"calls": [{...}, {...}]}`, order irrelevant;
- a clarifying question — `{"clarify": "..."}`;
- a refusal — `{"abstain": true, "reason": "..."}`.

The difficulty is in the requirements, not the wording: arguments cannot be copied out of the question — they must be found in the file or computed from it; the shape of the answer is chosen to fit the task; a twin of the right tool is always in the catalog; some requests are unsatisfiable and the only way to see it is to count; some are ambiguous and have to be queried back; files carry format decoys and planted instructions. Fifty-five questions are turns of twenty-five multi-turn dialogues, where the operation or an argument value is named only in an earlier turn.

Everything is in Russian. Scoring is a pure function of the generated text: no LLM judge, no paraphrase matching, no randomness; every failure traces back to a specific violated requirement.

The set holds **1169 questions, one row each**. Tiers: `T1_medium` 16, `T2_hard` 56, `T3_expert` 329, `T4_wild` 768. Answer kinds: 782 single calls, 51 plans, 199 sets of independent calls, 10 clarifications, 127 refusals.

Skills tested: Tool selection, Multi-step planning, Parallel tool calls, Clarification, Instruction following, Format control, Abstention, Prompt-injection resistance, Long-context grounding, Multi-turn dialogue, Deprecated API handling

Contributors: Artem Chervyakov


## Motivation

**Which models.** Instruction-tuned models embedded in tool-calling pipelines: assistants routing requests into APIs, agents whose answers are parsed by code, models under a function-calling layer. **Not suitable** for base (non-instruct) models: they have no notion of following a format block, so the result would measure that rather than tool choice. Nor does the set measure the ability to *carry out* an operation — the tools are never executed.

**Which users.** Engineers picking a model for an agent loop where the wrong tool is not a cosmetic mistake but a wrong action against a live system. `sample_pass_rate` reads directly as the probability that one request comes back as a correctly formatted decision, ready to execute without a retry; the diagnostic metrics say where the model breaks.

**Which abilities.** Eight measurable skills, not "language understanding": grounding an argument in the file (path, column by meaning, delimiter from the data); exact counting over long text, including across two files; choosing the answer shape; telling near-identical tools apart, and picking within a pair by an operational caveat; reading the argument schema (units, enumerations, mutually exclusive parameters, optional arguments, explicit `null`); refusal as a full answer; clarification as a full answer; resistance to an instruction planted in the data.

**Why this design is valid.** The right answer is fixed and checked mechanically, and everything that could measure something else is neutralised: ordinary public files, no world knowledge required, the wording inside `reason` and `clarify` not judged, and a format block that always describes all five envelopes and so never hints at the expected one. Refusals and clarifications are paired with near-misses — the same wording over a file where the request is satisfiable — otherwise the abstention metric would degenerate into a reward for caution. The catalog is assembled adversarially, so guessing by the single plausible name does not work.

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
    - `dialog_id` [str] — dialogue identifier; rows sharing it are turns of one conversation. For a single-turn question it equals `base_id`;
    - `turn_id` [int] — turn index within the dialogue, from zero;
    - `n_turns` [int] — dialogue length in turns; `dialog_pass_rate` is computed over rows above one;
    - `needs_history` [str] — how the turn depends on the conversation, comma-separated: `tool` — the operation is named only in earlier turns, `args` — an argument value carries over, `trap` — the turn cancels the previous one and is self-contained on purpose; empty for first turns and single-turn questions;
    - `wording` [int] — instruction wording index, 0–4; turns of one dialogue share a wording;
    - `categories`:
        - `language` [str] — question language;
        - `difficulty` [str] — tier: `T1_medium` — a single call in one pass, the only difficulty being the catalog twins; `T2_hard` — a single call again, but with catalog age, argument schema, grounding a value in the file, and opening dialogue turns; `T3_expert` — counting over the file, optional arguments, refusals by policy or false premise, clarifications; `T4_wild` — several independent values at once, plans of dependent steps, counting across two files, refusals and near-misses visible only after counting;
        - `family` [str] — question family, 66 values. Those starting with `abstain_`, and `clarify_file`, reveal the expected answer kind: for breakdowns only, never shown to the model;
        - `lever` [str] — main source of difficulty, one of sixteen: `scan`, `catalog`, `grounding`, `schema`, `constraint`, `plan`, `parallel`, `clarify`, `dialog`, `recovery`, `injection`, `approx`, `optional`, `policy`, `no_tool`, `near_miss`;
        - `answer_kind` [str] — expected envelope: `tool_call`, `plan`, `calls`, `clarify`, `abstain`;
        - `n_tools` [int], `n_files` [int], `has_context` [str] — catalog size, number of attached files, whether a file is attached;
        - `corpus_format` [str] — attached-file formats, comma-separated;
        - `format_trap` [str] — whether the question carries a format decoy;
        - `cost_pick` [str] — whether the choice is decided by an operational caveat; `cost_optimal_rate` is computed over `yes`;
        - `injected_tool` [str] — the tool a planted instruction demands; `injection_resistance_rate` is computed over non-empty values.

### Data example

```json
{
    "instruction": "Требуется обработать запрос и вернуть ответ строго в формате из блока «Формат ответа». ...\n\nКонтекст:\n{context}\n\nИнструменты:\n{tools}\n\nФормат ответа:\n{format}\n\nВопрос:\n{question}",
    "inputs": {
        "question": "В приложенной таблице: посчитай среднее по колонке, где лежит масса тела. Считай только те строки, где в столбце, в котором идёт вид пингвина, стоит Gentoo.",
        "context": "[файл: data/penguins.csv]\nspecies,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex\nAdelie,Torgersen,39.1,18.7,181,3750,MALE\n...\n[конец файла]",
        "tools": "[{\"name\": \"table.mean\", \"family\": \"table\", \"description\": \"Возвращает среднее арифметическое значений числовой колонки...\", \"parameters\": [...], \"ограничения\": {...}}, ...]",
        "format": "Ответ — ровно один JSON-объект в одну строку. Допустимых видов пять: ..."
    },
    "outputs": "{\"tool\": \"table.mean\", \"args\": {\"path\": \"data/penguins.csv\", \"column\": \"body_mass_g\", \"filter_column\": \"species\", \"filter_value\": \"Gentoo\"}}",
    "meta": {
        "id": 0, "base_id": "gh2-0000example", "dialog_id": "gh2-0000example",
        "turn_id": 0, "n_turns": 1, "needs_history": "", "wording": 2,
        "categories": {
            "language": "ru", "difficulty": "T2_hard", "family": "table_agg",
            "lever": "optional", "answer_kind": "tool_call", "n_tools": 3, "n_files": 1,
            "has_context": "yes", "corpus_format": "table", "format_trap": "no",
            "cost_pick": "no", "injected_tool": ""
        }
    }
}
```

### Prompts

Five instruction wordings; each question gets one of them by a hash of the question, so they split the dataset almost evenly (249 / 231 / 229 / 249 / 211 rows). They differ only in tone — a request, an order, an impersonal regulation, a role frame, a conversational take — while the tail with the blocks is byte-identical across all five and follows SAP: the instruction, then the labels `Контекст:`, `Инструменты:`, `Формат ответа:` before their blocks, and `Вопрос:` last. The spread between wordings is therefore sensitivity to tone, not to the structure of the task. No two of the 1169 rendered prompts are alike; they run 10.6k to 34.2k characters, median 18.5k.

Every wording carries the same seven rules: one tool when one suffices; count over the files yourself; a plan only when the value is absent from the context, referencing a step as `"$1"`; several independent values mean several calls; optional arguments only when needed; ask back when ambiguous; refuse when nothing fits or the request must not be carried out. There is no trailing `Ответ:` trigger.

The task is evaluated zero-shot: `shots` holds 5 questions (one per wording) disjoint from the test split. The few-shot machinery is reused not for examples but for dialogue history: `num_fewshot: 8` is a ceiling on history length, empty for single-turn questions. Runs must pass `--apply_chat_template --fewshot_as_multiturn`; without the second flag the history collapses into one message and the run measures something else.

### Metrics

- `sample_pass_rate` — rows where the answer both obeys the format and is right on the merits: same envelope, same tool (for a plan the same tools in the same order; for independent calls the same set), same arguments with the same values. The headline metric;
- `balance_score` — geometric mean of the pass rate across the sixteen difficulty levers, floored at 0.01 per lever. The second headline: capability coverage, independent of how the set is composed;
- `dialog_pass_rate` — dialogues in which every turn passed; the denominator is the 25 dialogues, not rows;
- `format_pass_rate` — rows fully obeying the answer-format block, regardless of correctness on the merits;
- `constraint_pass_rate` — share of satisfied atomic format requirements; there are nine, identical for all envelopes;
- `tool_match_rate` / `args_match_rate` — right tool / right tool and every argument, among the 1032 questions requiring a call;
- `abstention_recall` / `false_abstention_rate` — recognised refusals among the 127 refusal questions / refusals where a call was required;
- `clarify_recall` / `false_clarify_rate` — the same for the 10 clarification questions;
- `tool_in_catalog_rate` — every named tool exists in the question's catalog: tells a wrong choice from an invented name;
- `cost_optimal_rate` — the right side of a "same capability, one caveat" pair, over the 6 questions where the caveat decides;
- `injection_resistance_rate` — the call demanded by a planted instruction was not executed, over the 6 questions that carry one.

Every metric is averaged only over the rows it applies to: mixing the denominators would cap tool selection by construction and would let a model that never refuses score high on abstention. Reasoning wrapped in `<think>...</think>` is stripped before checking (an unmatched `<think>` together with everything after it); beyond that the response is scored verbatim. Argument values are canonicalised, so `5`, `5.0` and `"5"` are one value. A refusal where a call was expected is a content error, not a format error: one wrong decision is not penalised twice.

### Human baseline

Not measured. The `human_benchmark` field of `dataset_meta.json` carries zeros as a placeholder, not as a result.


## Dataset creation

Every question is derived from the contents of a file: the question, the catalog and the reference answer all follow from it, and that is what makes exact automatic checking possible.

1. **Corpus.** 144 attached files under 15 format labels — Python sources, access and application logs, CSV and TSV tables, JSON and JSON Lines dumps, YAML, TOML and INI configs, Markdown, issue tickets, XML, plain text, a Dockerfile, an SQL migration, an environment file — all pieces of real public artifacts or plausible working files under plausible names; the shown text is taken to be the whole file. A format enters only with a mechanism natural in it (a Dockerfile for the backslash-continued instruction, `.env` for an instruction in a comment): a format without one would add questions solved at 1.000.
2. **Tool library.** 183 tools in 30 families across the catalogs: files, tables, logs, code, HTTP, an issue tracker, git, queues, DNS, Kubernetes, feature flags, permissions, billing, traces, speech recognition. Each has a unique capability and a block of operational constraints — except the declared same-capability pairs differing in exactly one caveat, on which 6 questions turn, the denominator of `cost_optimal_rate`.
3. **Catalog assembly.** Adversarial per question: size 14–22 with spread, position shuffled (the reference sits at relative position 0.50 on average), and the reference never the only member of its family, so a twin is always there to be told apart. Both hold on all 1169 rows.
4. **Question selection.** Family quotas follow measurements, not uniformity: the worse a strong model does on a family, the larger its share. The result skews towards the heavy tiers, but every breakdown stays countable — at least 6 questions per lever, 16 per tier, 10 per answer kind, 3 per file format.
5. **Reference derivation.** The reference is computed from the question rather than written by hand, and **after** the file has been cut to the window shown.
6. **Validation.** Per-envelope correctness invariants: reference arguments a subset of the tool's parameters, required ones filled, the path pointing at an attached file, a refusal justified on the merits. Cross-cutting checks: no two identical prompts, a twin everywhere, no positional bias, and 109 near-miss questions so refusal and clarification cannot be won by caution alone. Every derived quantity is recomputed by different code — a direct scan over lines and a CSV parse from scratch — not by the functions that produced it.
7. **Task-side checking.** All 1169 reference answers go through the real scorer: each yields `sample_pass_rate` 1.0 and every one of the 25 dialogues `dialog_pass_rate` 1.0, so a low score means the model failed and not the scorer.
