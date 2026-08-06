# IFHardBench


## Task description

IFHardBench measures precise instruction following in Russian. Each question
pairs a deliberately trivial, knowledge-free writing task ("напиши про место, в
котором ты по-настоящему отдыхаешь") with a stack of three to six
machine-verifiable requirements on the answer: an exact word count, a forbidden
letter, a required word in a given position, a fixed number of commas, a list of
a given shape, a rule that applies only if the supplied context mentions rain.
Only compliance is scored — never the meaning of the answer and never knowledge.

Every requirement is backed by a pure deterministic verifier, so there is no LLM
judge, no reference-answer matching and no randomness, and every point lost
traces back to a named unsatisfied requirement. Every question also ships a
**witness**: a natural Russian answer, assembled from human-written sentences,
that satisfies the whole stack — proof that the question is solvable, never
compared against during scoring.

The dataset has two halves: 1260 single-turn questions, and a dialogue half of
340 turns in 136 dialogues where a rule is stated once and must survive the turns
that follow without being restated.

Evaluated skills: Instruction following, Format control, Constraint satisfaction,
Attention to detail, Self-verification

Contributors: Artem Chervyakov


## Motivation

The dataset targets **instruction-tuned** models whose output is consumed under
rules — an assistant told to answer in three sentences without markup, a
generation step whose result is parsed downstream, a system prompt that fixes a
house style. It is **not** suitable for base (non-instruct) models, which have no
notion of obeying a requirement block, and it does not measure writing quality,
knowledge or reasoning: the writing tasks are chosen to be answerable by anyone
about anything.

Results are aimed at engineers picking a model to put behind a fixed output
contract. The headline reads directly: `sample_pass_rate` is the probability that
a single request comes back satisfying every stated requirement at once, with no
retry and no repair pass.

Three capabilities are isolated by design:

1. **Holding several independent requirements at once.** Stacks mix families
   (counting, lexical, stylistic, format, conditional), so a model cannot satisfy
   them with one habit; it has to track each separately while writing.
2. **Self-verification.** Exact counts, acrostics, rings and checksums cannot be
   produced by fluent writing alone — they require the model to count what it
   wrote and correct it before answering.
3. **Evaluating a rule rather than executing it.** Conditional requirements state
   both branches and make which one holds depend on the supplied context, so
   blind then-execution is visible as a failure.

The design keeps the measurement valid by neutralising everything else. The
writing tasks need no world knowledge and have no correct content, so a wrong
answer cannot be a knowledge gap; requirement wordings state their own counting
convention; contradictory requirements are excluded by construction; and every
question's witness proves the stack is satisfiable, so a failure is the model's,
not the question's. Requirement types are also filtered by **free rate** — how
often an unconstrained answer satisfies them by accident — so no type can be
passed without reading it (bar 0.15; worst shipped type 0.125).

Stacks follow the **concentration principle**: one or two deliberately hard
requirements plus clear, easily satisfiable ones, rather than many moderately
hard ones. Because the headline is a product over the stack, this buys difficulty
without buying ambiguity — a question is hard because one requirement is hard,
not because five of them are vague.

The metrics follow from how such an answer is consumed. A response that misses
one requirement of five is unusable in the same way as one that misses all five,
so the headline `sample_pass_rate` is all-or-nothing. But those two failures call
for different fixes, so `constraint_pass_rate` reports the share of individually
satisfied requirements as a diagnostic. Both average over occurrences and are
therefore weighted by how common a requirement is — which hides the failure mode
engineers care about most, a *kind* of instruction the model cannot follow at
all. `balance_score` closes that gap: per-type pass rates combined geometrically
with equal weight per type, so concentrated failures cost more than spread-out
ones. Read together, the headline says how often the whole answer is right and
the balance says whether the model has a blind spot.


## Data description

### Data fields

- `instruction` [str] — prompt template with placeholders for the fields of
  `inputs`;
- `inputs` — the task shown to the model:
    - `question` [str] — the writing task;
    - `constraints` [str] — the requirements, one per line;
    - `context` [str] — background notes, never scored, empty for roughly half
      the questions;
- `outputs` [str] — the witness: a valid answer proving the question is solvable;
- `meta` — metadata hidden from the model:
    - `id` [int], `base_id` [str];
    - `constraints` [str] — the machine-readable requirement list the scorer
      reads (JSON);
    - `categories` — `topic`, `prompt_style`, `n_constraints`,
      `constraint_families`, `length_tier`, `language`, `tier` (quartile of
      predicted stack difficulty: `easy` / `medium` / `hard` / `expert`, 315
      questions each), `stratum`;
    - `annotation` — how the witness was verified.

`stratum` splits the file: `core` (440 questions) is the ordinary generation
policy, whose stack-size mix, tonality balance and conditional branch split are
checked exactly; `topup` (820) are questions added afterwards so that every
requirement type reaches at least 35 uses and a per-type pass rate is readable.
Nothing was removed to make room, and reports can quote either.

### Data formatting example

```json
{
    "instruction": "Слушай, помоги.\n\nТребования к ответу:\n{constraints}\n\nЗадание:\n{question}\n\nПеред тем как отвечать, проверь, что ничего не потерялось.",
    "inputs": {
        "question": "Что ты читал или смотрел за последнее время? Поделись впечатлением.",
        "constraints": "- Ответ должен состоять ровно из 2 абзацев; между абзацами — пустая строка.\n- Посмотри, каким словом кончается первый абзац: это же слово должно прозвучать и в последнем абзаце.\n- Уложись ровно в 246 символов, считая пробелы и знаки препинания.\n- Запятых в ответе — ровно 4; считается каждая.\n- Во всём ответе не должно быть ни одной буквы «п».",
        "context": ""
    },
    "outputs": "Финальная сцена снята без единого слова. Тираж у книги крошечный, еле нашёл в магазинах. Книга оказалась короче, чем я думал.\n\nРазве не в этом сила хорошей истории? Такие мелочи и делают обычный день хорошим. Я думал, что угадал финал, но ошибся.",
    "meta": {
        "id": 19,
        "base_id": "e321d89ce751",
        "categories": {
            "language": "ru", "tier": "hard", "length_tier": "short",
            "n_constraints": 5,
            "constraint_families": "lexical,structure,style",
            "prompt_style": "casual", "topic": "story"
        }
    }
}
```

### Prompts

40 templates are distributed evenly over the questions. Every block is introduced
by a fixed label and by nothing else, in a fixed order:

```
[обращение]

Контекст:
{context}

Требования к ответу:
{constraints}

Задание:
{question}

[напоминание]
```

The labels are identical in all templates; what varies is the opening address and
the closing reminder, which carry five tonalities — просьба, приказ,
обезличенная спецификация, разговорная, строгая. The impersonal one has neither
and renders as bare labelled blocks. A question without background ships a
template without the `Контекст:` block rather than an empty one. The task is
always the last block, so the requirements are never buried in the middle, and
there is no `Ответ:` cue: these prompts go to instruct models, where the user
turn is closed by an end-of-turn token.

### Dataset creation

Questions are produced by a deterministic generator — the same seed rebuilds the
dataset byte for byte.

1. **Catalogue.** 61 requirement types over five families (structure, lexical,
   style, format, logic). Each carries a deterministic verifier, three to five
   Russian wordings, an English mirror, and declared incompatibility groups, so a
   stack can never contain two requirements that contradict each other. Wordings
   state their own counting convention wherever a person might count differently.
2. **Tasks and context.** Writing tasks and background paragraphs come from a
   hand-written bank over nine everyday topics. Contexts are whole paragraphs
   used verbatim, never stitched together; some are **decoys** that describe a
   formatting habit the requirements then override, so background that looks like
   an instruction has to lose to a real one.
3. **Stack assembly.** Three to six requirements over at least two families;
   every stack pins how much text the answer must contain; shape quotas hold the
   intended mix against composability drift.
4. **Solvability proof.** The witness is assembled from human-written bank
   sentences. The composer never invents text — it selects, and may only reorder,
   lowercase, wrap lines, embolden a chosen word, prepend the task line or append
   a count. A question for which no witness can be built is discarded, so no
   unsatisfiable question can reach the dataset.
5. **Free-rate filtering.** Every type was measured against a fresh corpus of 162
   unconstrained model answers to the dataset's own tasks: how often is the
   requirement satisfied by accident, without being asked? Every shipped type
   sits under the 0.15 bar; types measured over it were built, verified and not
   shipped. Conditional types are measured per branch, because averaging the two
   would hide the only number that decides whether the type ships.
6. **Validation.** Every shipped witness is re-scored with the shipped scorer;
   every verifier is checked to reject a targeted corruption of a passing answer
   (1221 distinct parameter pairs); every conditional's branch is re-derived from
   the context it ships with. The verifier suite is 403 cases, and all of it runs
   at build time rather than being assumed.

### Limitations

- Human annotation of a ≥100-question sample is the outstanding acceptance check.
  Every question carries a constructive witness, which is a stronger mechanical
  guarantee, but it is not the same thing.
- **The false branch of a conditional requirement is a control, not a
  difficulty**: a model that never read the rule passes it, because the word it
  forbids was not going to appear anyway. Branches ship 81 % true in `core` and
  69 % overall.
- Two types are implemented and verified but thin in the shipped build:
  `lexical:sentence_acrostic` (14 uses) and `lexical:epiphora` (1). They compete
  for the same slot as the numeric totals every stack carries; reports mark such
  cells instead of quoting them.


## Evaluation

### Metrics

- `sample_pass_rate` — share of questions where the response satisfies every
  requirement at once. The headline metric. An empty response is a plain failure.
- `balance_score` — the pass rate of each requirement type, combined as a
  geometric mean with equal weight per type and floored at `0.01`. The second
  headline: it asks whether a *kind* of instruction is missing rather than how
  many requirements were missed. The same number of failures costs more when
  concentrated in one type; a type the model never gets right costs about 7 % of
  the score rather than zeroing it; a type with fewer than 5 verdicts in the run
  is excluded and named in the report. **It is read against
  `constraint_pass_rate`, not against `sample_pass_rate`** — both are built from
  per-requirement verdicts, while the headline is a product over the whole stack
  and is structurally far lower.
- `constraint_pass_rate` — share of individually satisfied requirements; a
  partial-credit diagnostic separating "missed one requirement of five" from
  "missed all of them". **Computed over non-empty responses only**: per sample the
  value is `null` for an empty response and a custom aggregation averages the
  rest. A reasoning model that spends its whole token budget in the trace returns
  an empty answer, which honestly fails the headline metric — but zeroing five
  constraint verdicts for it would turn the diagnostic into a proxy for token
  budgeting. The run audit reports the empty share separately.

Because `sample_pass_rate = Π pᵢ` over the requirements of a question, the
stack-size distribution bounds the score directly: at a per-requirement pass
probability of 0.80 the ceiling is 0.41 for a five-requirement stack.

Reasoning traces wrapped in `<think>...</think>` are stripped before
verification; beyond that the response is scored verbatim, with no lenient
extraction of an "answer part". An **unterminated** trace — the model was cut off
inside its own reasoning — leaves its text as the answer and fails: the answer is
not absent, it is wrong.

### Human baseline

TODO
