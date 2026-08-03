# IFHardBench


## Task description

**IFHardBench** measures how precisely a model follows explicit formatting instructions in Russian. Each of the 440 questions pairs a deliberately trivial, knowledge-free writing task with a stack of **three to six** machine-verifiable requirements — exactly forty-seven words, a bulleted list of exactly five items whose first letters spell «берег», no letter «д» anywhere, the seventh word must be «утро», a JSON object with exactly three named keys.

Only compliance is scored, never the meaning of the answer: any text that passes every verifier is accepted. Every requirement is backed by a deterministic verifier (`utils.py`), so there is no LLM judge, no reference-answer matching and no randomness — a score is a pure function of the generated text and always traces back to the specific requirement that was violated.

Two metrics are reported: `sample_pass_rate` (all requirements of a question satisfied at once — the headline metric) and `constraint_pass_rate` (share of individually satisfied requirements — a partial-credit diagnostic).


## Motivation

Instruction following is usually measured on tasks where the instruction and the content compete for the model's attention, so a failure is ambiguous: did the model ignore the format, or did it not know the answer? IFHardBench removes the second possibility. The writing tasks — "describe the place where you rest best" — need no knowledge at all, so everything the score measures is compliance.

The requirement types are chosen against two specific weaknesses of a left-to-right decoder:

* **Committing to a number before the text exists.** An exact word count, an exact character budget, "exactly six words in every sentence", "the seventh word must be X" — all of these have to be true of text that has not been written yet, and none of them can be repaired at the end without rewriting.
* **Planning the right-hand side in advance.** An acrostic fixes the first letter of every item before the first item is written. A closing phrase fixes the tail. A lipogram must hold for every character still to come.

Both are things a person does with a pencil in two minutes, and both are trivially checkable — which is exactly the combination a benchmark wants.


## Data description


### Data fields

- `instruction` — a prompt template with placeholders for the fields of `inputs`;
- `inputs` — a dictionary of the question's parts:
    - `question` — the writing task: a neutral, knowledge-free request;
    - `constraints` — the requirements, one per line, each starting with `- `;
    - `context` — background notes, irrelevant to scoring; an empty string for half the questions;
- `outputs` — a witness: a natural Russian answer that satisfies every requirement of the question. It is a proof of solvability, not the unique correct answer;
- `meta` — technical metadata:
    - `id` — the question's number in the dataset;
    - `base_id` — the generator's identifier for the question;
    - `constraints` — a JSON string with the machine-readable requirements: category, family, verifier parameters. Duplicates the content of `inputs.constraints` and is used solely for automatic scoring;
    - `categories` — `language`, `tier` (stack depth), `length_tier` (with or without the background block), `n_constraints`, `constraint_families`, `prompt_style`, `topic`;
    - `annotation` — `witness` (the generator proved solvability constructively), `solvability_review` / `correctness_review` (an independent review of a sample, see *Dataset creation*), `reviewer`.


### Data formatting example

```json
{
    "instruction": "Сделай ровно то, что написано ниже.\n\nЧто должно быть в ответе:\n{constraints}\n\nЗадача: {question}\n\nПроверь себя по каждому пункту, прежде чем отвечать.",
    "inputs": {
        "question": "Опиши маршрут, по которому тебе нравится ходить пешком.",
        "constraints": "- Ответь одним объектом JSON — и ничем больше. Ключей ровно три: «что», «где», «когда». В каждом значении не больше 3 слов.\n- Во всём ответе не должно быть ни одной буквы «в».\n- Не забудь про слово «арка»: оно должно быть в тексте, форма неважна.",
        "context": ""
    },
    "outputs": "{\"что\": \"Лёгкая усталость\", \"где\": \"Спокойный темп\", \"когда\": \"Эхо под аркой\"}",
    "meta": {
        "id": 107,
        "base_id": "033191e5bf20",
        "constraints": "[{\"category\": \"format:json_object\", \"family\": \"format\", \"params\": {\"keys\": [\"что\", \"где\", \"когда\"], \"max_words\": 3, \"no_commas_in_values\": true}, \"is_terminal\": true}, {\"category\": \"style:forbid_letter\", \"family\": \"style\", \"params\": {\"letter\": \"в\"}, \"is_terminal\": false}, {\"category\": \"lexical:include_keyword\", \"family\": \"lexical\", \"params\": {\"word\": \"арка\", \"stem\": \"арк\"}, \"is_terminal\": false}]",
        "categories": {
            "language": "ru",
            "tier": "easy",
            "length_tier": "short",
            "n_constraints": 3,
            "constraint_families": "format,lexical,style",
            "prompt_style": "order",
            "topic": "walk"
        },
        "annotation": {
            "witness": "Да",
            "solvability_review": "Да",
            "correctness_review": "Да",
            "reviewer": "model:x-ai/grok-4.3"
        }
    }
}
```


### Counting conventions

Three definitions decide almost every verdict. They are written to match how a person counts rather than how a tokeniser does, because a requirement a careful reader would satisfy and the scorer would reject is a defect in the benchmark, not in the model.

- **word** — a whitespace-separated token containing at least one letter or digit. `«чёрно-белый»` is one word, `«2025»` is one word, a standalone dash is not a word. An answer is never penalised for containing a number or a Latin name.
- **sentence** — text between terminators `.`/`!`/`?`/`…` that are followed by whitespace or the end of the answer. A lone `.` after a single-letter fragment does not terminate, so `«и т.д. дальше»` stays one sentence and `«3.14»` is not split. Trailing text with no terminator still counts as a sentence.
- **list item** — a line beginning with `-`/`*`/`•` or `1.`/`1)`.

Word matching (required words, forbidden words, positional words, acrostics) is case-insensitive and treats `ё` as `е`. "Use the word X" and "do not use X" both match **any inflected or derived form**, which is what the wording promises: Russian drops vowels, and a rule that caught `дня` but not `день` would be a trap.


### Prompts

40 prompt templates: five tonalities — a polite request, a direct order, an impersonal specification, a casual chat message, and a strict "your answer will be checked" framing — in four different wordings each, in layouts with and without the `{context}` block. They are distributed evenly, 88 questions per tonality, so a measured score reflects instruction following rather than sensitivity to one phrasing.

Every template follows the Semantic Anchor Prompting block order, with the question as the final semantic block:

```
[обращение] → Контекст → Требования к ответу → Вопрос → [напоминание]
```

Placing the requirements immediately before the question keeps the answer-format block out of the middle of the prompt, where a long context would bury it. The closing reminder restates "satisfy all of the above" in the tone of that wording; it adds no requirement of its own, so the requirement set of a question is fully contained in `inputs.constraints`.

No template ends with an `Ответ:` continuation cue. The benchmark targets instruct models, where the user turn is closed by an end-of-turn token — a trailing trigger cannot be continued and therefore buys nothing.

Prompt example:

```
Контекст задачи: К середине августа вечера стали заметно короче. Днём всё ещё жарко, но после заката нужна кофта. Так каждый год.

Ниже список требований. Ответ будет проверяться по каждому из них:
- Уложись в 190–198 символов, считая пробелы и знаки препинания.
- В тексте должны встретиться «иней», «трава» и «туман» — в любых формах.
- Обойдись без буквы «ы» — ни в одном слове.
- Шестое слово ответа — «сухая».
- Ни «время», ни «дождь», ни «лето», ни «осень», ни родственных им слов в ответе быть не должно.

Задание: Какое время года тебе ближе всего? Напиши почему.

Несоответствие хотя бы одному требованию означает, что ответ не принят.
```

The requirement wordings are not canonical either: each requirement type has three to five different phrasings, so a model has to read the requirement rather than recognise a template.


### Few-shot examples

The task is evaluated zero-shot (`num_fewshot: 0`). Because every question carries its own requirement set, a demonstration cannot teach the answer format of the next question — and risks anchoring the model on the demonstration's constraints. The `shots.json` split therefore holds only 5 questions, built from a different generator seed and checked against the test split so a shot can never leak a test question.


### Dataset creation

Questions are produced by a deterministic generator (`ifhb9`) rather than collected from a source corpus, which is what makes exact automatic verification possible. The same seed rebuilds the dataset byte for byte.

1. **Constraint catalogue.** 25 requirement types across four families: `structure` (exact word, sentence, paragraph and character counts; words per sentence; words per list item; sentences per paragraph), `lexical` (a required word in any form, three required words, forbidden words and their cognates, a required opening or closing phrase, a word pinned to the N-th position, an acrostic over the list items), `style` (a forbidden letter, no commas, lowercase only, a required final punctuation mark, all sentences starting with different letters) and `format` (a bulleted or numbered list of exactly N items and nothing else, a JSON object with exactly three named keys and a word cap on each value, a markdown table of N rows by M columns). Each type ships a deterministic verifier, three to five Russian wordings and an English mirror.

2. **Stack assembly.** A writing task is combined with 3–6 requirements. Requirements may co-occur only if they constrain distinct quantities and belong to disjoint incompatibility groups, so contradictory stacks are excluded by construction. Every stack draws on at least two families, contains at least one requirement that pins how much text there must be (otherwise a stack of pure prohibitions would be satisfied by the word «да»), and contains at least one — two, from four requirements up — of the types a strong model actually fails.

3. **Solvability proof.** For every question the composer searches for a **witness**: a natural Russian answer that satisfies the whole stack. The composer never invents text — it selects whole sentences and list items that were written as ordinary Russian and is allowed only three cosmetic operations (joining with a space or newline, lowercasing, replacing the final punctuation mark). A question with no witness is discarded. This is the lesson of the previous version, where 416 questions had a mechanical witness and human annotators still rejected them: every passing answer was degenerate («да», «463»).

4. **Free-rate filtering.** 108 unconstrained answers to the dataset's own writing tasks were collected from a live model, and every requirement type was measured against them: how often is it satisfied by an answer that never tried? Types above 0.15 were re-parameterised or dropped. This is what killed the previous version's `exclude_word` (0.99 free), `char_count le 280` (0.97) and `no_numerals` (0.90). Required words are now chosen from words a model *rarely* reaches for on that topic; forbidden words from the ones it leans on hardest.

5. **Difficulty calibration.** For each type ~30 questions were generated using that type alone and run against `deepseek/deepseek-v4-flash-0731`, which is what decides the parameter ranges and which types the stack policy treats as hard.

6. **Independent review.** A sample of questions goes through the same two-question review as the previous version ("is it solvable?" / "is it correctly worded?"), recorded in `meta.annotation`. See *Limitations*.

The final set spans 25 requirement types (24 of them used) over 440 questions, mean 4.2 requirements and 3.1 families per question, in 40 prompt templates and 54 writing tasks over 9 topics.

All 445 shipped witnesses are re-verified against the shipped verifier library at build time, so the invariant "every question is satisfiable, and the scorer agrees" is checked mechanically rather than assumed.


### Limitations

The solvability review that ships in `meta.annotation` was performed by a **model** (`x-ai/grok-4.3`), not by human annotators, and it returned 63 % "solvable" against a 90 % bar.

That number should be read carefully. Every question the reviewer rejected ships a witness that satisfies every one of its requirements and reads as ordinary Russian — the reviewer called tasks impossible while a valid answer to them exists in the dataset. So the 63 % is not evidence that the questions are broken. It is also not evidence that they are fine: a model judging "could this be answered" is a weak instrument in both directions, and the previous version of this dataset is the cautionary tale for trusting mechanical satisfiability alone.

**Human annotation of a ≥100-question sample is the one outstanding check**, and it is the check that decides whether the shipped configuration stands or is relaxed. `mera_results/history/` holds a measured, validated, easier alternative (0.550 instead of 0.459 on gpt-5.6-terra) for exactly that eventuality.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `sample_pass_rate`: Share of questions where the model response satisfies every requirement of the question at once. The headline metric: partial instruction following does not count.
- `constraint_pass_rate`: Share of individually satisfied requirements, averaged over questions. A diagnostic metric: it separates missing one requirement out of five from missing all of them.

Scoring details that affect comparability between runs: reasoning traces wrapped in `<think>...</think>` are stripped before verification, since they are scaffolding rather than part of the answer; beyond that and trimming outer whitespace the response is scored verbatim, with no lenient extraction of an "answer part", because peeling off a preamble would credit a response the instruction did not ask for. An empty response never counts as a pass — prohibition-style requirements such as "do not use the word X" are vacuously true of the empty string.

Because `sample_pass_rate = Π pᵢ` over the requirements of a question, the stack-size distribution bounds the score directly: at a per-requirement pass probability of 0.80 the ceiling is 0.41, at 0.75 it is 0.32.


### Human baseline

TODO
