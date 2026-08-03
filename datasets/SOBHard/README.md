# SOBHard


## Task description

Structured output between machine-readable formats. Each question hands the model a document in one of eight source notations and asks for a document in one of five target notations, packaged exactly as the prompt specifies. Five task families are represented: converting a document to another notation, extracting values at named paths, repairing a syntactically broken document, applying a declared rewrite rule, and inferring a JSON Schema from a sample.

Nothing is judged by a language model. Every rule the prompt states — the packaging, the syntax, the numeric tolerance, the semantics of the family — is checked by a separate deterministic verifier, so a score always traces back to a named constraint.

Every input is a real file: a Kubernetes manifest, a Cargo lockfile, an airport-code table, an Atom feed, an OpenAPI description. Nothing is invented, so the key names, the value distributions and the awkward corners are the ones that occur in practice. Documents too large for a question are trimmed — a table keeps its first rows, a mapping keeps a subset of its top-level keys, a large tree is entered through one of its own subtrees — which shortens a real document rather than generating a synthetic one.

Evaluated skills: Structured output, Format conversion, Format control, Schema inference, Attention to detail

Contributors: Artem Orlov


## Motivation

Structured output is where a capable model most often fails in production. A model that reasons well about a document can still emit it with a stray sentence of commentary before the code fence, a trailing comma, or a key order that a downstream parser rejects — and in an automated pipeline that failure is total, not partial. Benchmarks that score only the semantic content of an answer are blind to this class of error, because they extract the payload leniently before comparing it and therefore silently repair exactly what they should be measuring.

SOBHard is aimed at instruction-tuned models being considered for pipelines that consume their output programmatically. It is not suitable for judging open-ended generation quality, and it deliberately requires no world knowledge: every question is solvable from the document in front of the model, so a low score means the model could not follow a mechanical specification rather than that it lacked a fact.

The design follows from that goal. Each question is a four-block prompt — task, input data, answer format, instruction — in which the answer-format block states the packaging rules explicitly and the instruction carries the parameters specific to that question. Because the rules are stated, failing them is unambiguously a failure to follow the specification, not a disagreement about convention. The verifiers mirror the stated rules one to one, so the benchmark never penalises a behaviour it did not ask for.

### Where the difficulty comes from

Difficulty is a ladder of *work*, not of obscurity, and the ladder is calibrated against a measured relationship: pass rate falls monotonically with the number of tokens a model spends on a question. Anything that honestly increases the work per question therefore lowers the score without making the wording any harder.

Three properties of the harder questions do that, and all three lean on the same limitation: a model reads and writes left to right, and cannot easily plan what it has not yet seen.

* **The answer depends on the whole input before its first character can be written.** A schema whose array description is derived from the union of *every* element cannot name a single property until the last record has been read. An array's `minItems` is unknown until the array ends. An `enum` needs the distinct values and then a sort.
* **The task is genuinely iterative.** The harder `transform` questions chain two rules, and the hardest chain three: each is applied to the output of the one before it. Filtering and then sorting means the first record of the answer is unknowable until the last record of the input has been classified.
* **The whole document has to be reproduced exactly.** A `repair` question with five breaks scattered through twenty kilobytes cannot be answered by a local edit — every remaining byte has to come back intact as well.

Length carries the rest. The set is 30% short inputs (800-2500 characters), 40% medium (2500-8000) and 30% long (8000-24000), and the length grows through real structure — more keys, deeper nesting, more homogeneous records — rather than through a repeated block. The median input is about 4400 characters and the largest is just under 24000.

### Why the metrics are split

`sample_pass_rate` is all-or-nothing because a document that violates one rule is unusable downstream regardless of how many other rules it honours. But an all-or-nothing headline alone cannot tell a model that solved the task and mispackaged it from a model that never solved it, and those two call for entirely different work from the reader. `format_pass_rate`, `content_pass_rate` and `task_pass_rate` therefore report the packaging, the content and the family-specific semantics separately, and `constraint_pass_rate` gives the partial credit that makes small regressions visible.

The question set is a balanced grid: five families crossed with four difficulty levels, twenty-five questions in each of the twenty cells. Balance matters here because the families have very different failure modes — a schema inference error looks nothing like a broken repair — and an unbalanced set would let a model's aggregate score drift with the composition rather than with its ability.


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [int] — Index of the prompt template in `dataset_meta.json["prompts"]`. The Hugging Face copy carries the template text itself.
- `inputs` — Input data that forms the task for the model.
    - `task` [str] — Description of the operation: which transformation is required, between which notations, and — where the notation has no obvious data model — how the input is read as a structure.
    - `input_data` [str] — The source document the operation is applied to. Given verbatim and never translated.
    - `format` [str] — Answer packaging rules: target notation, fenced block with a tag, no text outside it, whether key order is graded, numeric and string comparison tolerances, and how the target notation holds a document where that is not self-evident.
    - `question` [str] — The concrete instruction for this question, carrying its parameters — extraction paths, rewrite rule, schema inference rules, number of breaks.
- `outputs` [str] — The reference document, packaged exactly as the prompt requires. Scoring reads the reference from `meta.reference`, not from this field.
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — Identification number of the question in the dataset.
    - `base_id` [str] — Content hash of the question, stable across rebuilds.
    - `fence_tag` [str] — The markdown fence tag the answer is required to use.
    - `reference` [str] — The reference document as raw text. The source of truth for scoring.
    - `reference_sha256` [str] — Checksum of the reference, for dataset integrity checks.
    - `task_meta` [str] — JSON string with the question parameters: extraction paths and which of them are absent, break kinds, the rewrite programme, the schema kind.
    - `checks` [str] — JSON string with the input document read under the convention the prompt states, plus an applicability gate for the two checks that compare shapes across notations.
    - `categories` — Categorial features characterizing the test example.
        - `family` [str] — Task family: `convert`, `extract`, `repair`, `transform` or `schema_gen`;
        - `difficulty` [str] — Difficulty level: `easy`, `medium`, `hard` or `expert`;
        - `language` [str] — Language of the question surface;
        - `source_format` [str] — Notation of the source document;
        - `target_format` [str] — Notation the answer must be produced in;
        - `length_tier` [str] — Length band of the input: `short`, `medium` or `long`;
        - `prompt_style` [str] — Register of the prompt wording: `request`, `casual`, `spec`, `formal` or `command`;
        - `origin` [str] — Which corpus file the document came from, and which subtree of it.


### Data formatting example

```json
{
    "instruction": 2,
    "inputs": {
        "task": "Структурное преобразование документа CSV. Правило задано в инструкции, результат нужен в JSON. Таблица читается как список объектов: первая строка задаёт имена колонок, каждая следующая даёт один объект, и все значения в нём остаются строками.",
        "input_data": "iata,name,country\nAMS,Amsterdam Schiphol,NL\nATL,Hartsfield Jackson,US\nBOM,Chhatrapati Shivaji,IN",
        "format": "Формат ответа простой: один блок кода `json`, внутри документ JSON, снаружи ничего. Порядок ключей здесь — часть результата, он проверяется; порядок элементов в списках тоже. Числа сходятся с относительной точностью 1e-6, строки должны быть в точности такими же.",
        "question": "Примени к документу выше это правило и выдай результат как JSON. Сгруппируй записи по значению поля \"country\": результат — объект, где ключ равен этому значению, записанному строкой, а значение — список записей с таким значением поля, в исходном порядке. Ключи в результате идут в лексикографическом порядке, а сами записи не меняются."
    },
    "outputs": "```json\n{\n  \"IN\": [ … ],\n  \"NL\": [ … ],\n  \"US\": [ … ]\n}\n```",
    "meta": {
        "id": 214,
        "base_id": "7c1de0a4b39f5182",
        "fence_tag": "json",
        "reference": "{ … }",
        "reference_sha256": "…",
        "task_meta": "{\"kind\": \"transform\", \"program\": [{\"rule\": \"group_records\", \"params\": {\"field\": \"country\"}}], \"rules\": [\"group_records\"], \"graded_key_order\": true}",
        "checks": "{\"source\": [ … ], \"gates\": {\"convert.lossless_vs_source\": false, \"schema.validates_source\": false}}",
        "categories": {
            "family": "transform",
            "difficulty": "medium",
            "language": "ru",
            "source_format": "csv",
            "target_format": "json",
            "length_tier": "short",
            "prompt_style": "spec",
            "origin": "csv/civic_airport_codes.csv"
        }
    }
}
```


### Prompts

Twenty prompt templates, evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces are filled in from the fields inside `inputs`.

The templates differ only in their opening line. That line carries one of five registers — a polite request, a colloquial aside, a terse specification, an official formulation and a command — with four wordings each. Block labels and block order are identical across all twenty, so prompt wording is a single controlled factor rather than a bundle of confounded ones. Distribution is stratified: each register appears exactly five times in every one of the twenty family-by-difficulty cells, which keeps register from correlating with task type and turns the register breakdown in the audit into a real check on the wording rather than a tautology.

Inside the blocks the wording varies further: nine formulations of the task and nine of the instruction per family, eight of the answer-format block, and two or three of every rewrite rule and every schema-inference rule. Nothing addresses the model as «вы»; the question is always last, and no answer trigger follows it.

Prompt example:

```
Ниже приведены условия задачи, входные данные и требования к формату ответа.

Задача:
{task}

Входные данные:
{input_data}

Формат ответа:
{format}

Инструкция:
{question}
```


### Few-shot examples

The `shots` split holds five questions, one per family, built from corpus documents that no test question uses. Their answers are the reference documents, packaged in the required fence, so a shot demonstrates both a correct solution and correct packaging.


### Dataset creation

The questions are generated by the `sobhard_gen` package that ships next to this file. It reads real documents out of a corpus under a data model it can state in two or three sentences, trims them to the requested size, applies the family's operation to produce the reference, and renders the four prompt blocks. The build is deterministic: the same corpus and the same seed reproduce byte-identical files.

Because the reference is produced by a deterministic procedure that also defines the task, it is correct by construction rather than by annotation — but "by construction" is exactly the assumption that failed in the previous generation, so nothing is taken on trust:

* every reference is written and then **read back** with the target parser and compared with the structure it was built from, and the question is discarded if they differ;
* every input is confirmed to read as the structure the answer was derived from, and every `repair` input is confirmed to genuinely fail to parse;
* for `convert` and `repair`, where losslessness is claimed, **every token of the document** has to appear in the reference;
* handing the input document back unchanged, in the fence the question asks for, must score zero — checked once for all five hundred questions, which is how a `json` to `yaml` pair got removed (JSON is a subset of YAML, so those seven questions were answerable by copying).

Only notations with a data model that fits in a few sentences are admitted, and the model is told what it is. This is the correction that mattered most. The previous generation drew on twenty source notations through parsers whose canonical shapes were internal details: a LaTeX document became `{"type": "heading", "level": 2, "text": "Setup"}` and everything else was dropped, and a Turtle document collapsed into a single string. No engineer reading the prompt could have produced those, so the questions measured guessing rather than structured output. The eight notations kept — JSON, JSONL, YAML, TOML, INI, CSV, TSV, XML — each have their reading convention written into the task block where it is not self-evident, and documents that do not fit their model (a YAML with `yes` as a scalar, a CSV with a repeated column name, an XML element mixing text and children) are refused rather than explained away.

Three defects of the previous generation are fixed at the source rather than patched:

The `transform` family had a single rule and contradicted itself: the question asked for keys sorted recursively while the shared answer-format block told the model that key order is not graded. There are now ten rules, the harder questions chain two of them, and the format block states that key order is graded on exactly the questions whose rule fixes it — checked in both directions by the validator.

The `repair` family had one localised break, and one of its break kinds was unanswerable: it deleted a whole `[section]` line and expected the model to restore the section's name, which had left with the line. There are now nineteen break kinds across five notations, several of which need the model to know what the notation *is* rather than only its grammar — `True` instead of `true` is valid Python and invalid JSON, two records merged onto one line are valid JSON and invalid JSONL, an unquoted string is valid YAML and invalid TOML. No break removes a name, a key or a value; they move, unbalance, mistype or duplicate, and the original is always readable off the damaged text.

The `schema_gen` family had a single kind whose type mapping was spelled out in the question, which made it transcription rather than inference. There are now six, and four of them — a union over every element of an array, array bounds, an enum of distinct values, and a combined kind that asks for all three at once — cannot be produced while reading the document from left to right.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `sample_pass_rate`: The share of questions where the answer satisfies every applicable constraint at once. The headline metric — partial compliance does not count, because a document that breaks one rule is unusable downstream.
- `constraint_pass_rate`: The share of individually satisfied constraints across all questions. Partial credit that separates one missed detail from ignoring the requirements entirely.
- `format_pass_rate`: The share of questions where the answer is packaged correctly — exactly one fenced block with the right tag, nothing outside it, and a body that parses as the target notation.
- `content_pass_rate`: The share of questions where the answer content matches the reference: structure identical, numbers within 1e-6 relative tolerance, strings byte-exact.
- `task_pass_rate`: The share of questions where the family-specific semantic requirements hold — the rules a plain comparison with the reference cannot see, such as whether keys were actually sorted, whether an absent path was reported as `null`, or whether a produced schema really validates its source document.

An empty or unparseable answer scores zero on every metric. Several constraints are vacuously true of an empty string, so without this rule a model that answered nothing would collect partial credit.


### Human baseline

Human baseline measurement is pending. The questions are mechanical and self-contained by design: an engineer who reads the prompt can produce the answer without guessing at an unstated convention, and the validator enforces the properties that make that true — every reading convention stated, every rule unambiguous, no contradiction between the answer-format block and the instruction.
