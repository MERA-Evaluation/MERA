# IFHardBench

## Task description

**IFHardBench** measures precise instruction following in Russian. Each of the
1260 questions pairs a deliberately trivial, knowledge-free writing task
(«напиши про место, в котором ты по-настоящему отдыхаешь») with a stack of three
to six machine-verifiable requirements on the answer: an exact word count, a
forbidden letter, a required word in a given position, a fixed number of commas,
a list of a given shape, a rule that applies only if the supplied context
mentions rain. Only compliance is scored — never the meaning of the answer and
never knowledge.

Every requirement is backed by a deterministic verifier, so there is no LLM
judge, no reference-answer matching and no randomness, and every point lost
traces back to a named unsatisfied requirement.

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

Everything else is neutralised to keep the measurement valid. The writing tasks
need no world knowledge and have no correct content, so a wrong answer cannot be
a knowledge gap; requirement wordings state their own counting convention;
contradictory requirements are excluded by construction; and every question was
built together with a **witness** — a natural Russian answer satisfying the whole
stack — so a failure is the model's, not the question's. Requirement types are
also filtered by **free rate**, how often an unconstrained answer satisfies them
by accident, so no type can be passed without reading it (bar 0.15; worst shipped
type 0.125).

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
    - `constraints` [str] — the requirements, one per line, each starting with
      `- `;
    - `context` [str] — background notes, never scored; a non-empty paragraph for
      717 of the 1260 questions and an empty string for the rest;
- `outputs` [str] — the witness: an answer satisfying the whole stack, kept as
  evidence that the question is solvable. It is never compared against during
  scoring, so the split ships it empty in `test` and filled in `shots`;
- `meta` — metadata hidden from the model:
    - `id` [int] — running number over the whole dataset;
    - `base_id` [str] — stable identifier of the question;
    - `constraints` [str] — the machine-readable requirement list the scorer
      reads, as a JSON string;
    - `categories` — `language`, `tier` (quartile of predicted stack difficulty:
      `easy` / `medium` / `hard` / `expert`, 315 questions each), `length_tier`,
      `n_constraints`, `constraint_families`, `prompt_style`;
    - `annotation` — `is_solvable` and `language_correctness`, reserved for
      human acceptance marks and currently unfilled.

### Data formatting example

```json
{
    "instruction": "Контекст:\n{context}\n\nТребования к ответу:\n{constraints}\n\nЗадание:\n{question}\n\nПроверка выполняется по каждому требованию отдельно.",
    "inputs": {
        "question": "Расскажи, как выглядит идеальный вечер после тяжёлого дня.",
        "constraints": "- Раздели мысль ровно на 8 предложений.\n- Сколько слов в первом предложении, ровно столько же должно быть и в последнем.\n- Мне нужен текст длиной ровно 272 символа; пробелы тоже считаются.\n- Не используй букву «я» — нигде в ответе.",
        "context": "Сосед по переписке отвечает всегда телеграфно: «вечер удался», «чай, плед». Я перенял у него эту манеру. Длинные описания вечеров кажутся мне теперь излишеством."
    },
    "outputs": "Мелким удачам радуюсь всерьёз. Лампа даёт тёплый круг света. Радио шумело вполголоса на кухне. Телевизор мы не включаем почти никогда. В подъезде хлопнула дверь, и снова стало тихо. Настроение было ровное. Всё вышло тихо, и мне это подошло. Никаких сложностей не возникло.",
    "meta": {
        "id": 18,
        "base_id": "d5f59cea723c",
        "constraints": "[{\"category\": \"structure:sentence_count\", \"family\": \"structure\", \"params\": {\"op\": \"eq\", \"value\": 8}, \"is_terminal\": false}, {\"category\": \"structure:mirror_sentences\", \"family\": \"structure\", \"params\": {}, \"is_terminal\": false}, {\"category\": \"structure:char_count\", \"family\": \"structure\", \"params\": {\"op\": \"eq\", \"value\": 272}, \"is_terminal\": false}, {\"category\": \"style:forbid_letter\", \"family\": \"style\", \"params\": {\"letter\": \"я\"}, \"is_terminal\": false}]",
        "categories": {
            "language": "ru",
            "tier": "hard",
            "length_tier": "long",
            "n_constraints": 4,
            "constraint_families": "structure,style",
            "prompt_style": "spec"
        },
        "annotation": {"is_solvable": null, "language_correctness": null}
    }
}
```

### Prompts

40 templates are distributed over the questions. Every block is introduced by a
fixed label and by nothing else, in a fixed order:

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
обезличенная спецификация, разговорная, строгая — eight templates each. The
impersonal one drops the opening address and begins directly with the first
labelled block. A question without background ships a
template without the `Контекст:` block rather than an empty one. The task is
always the last block, so the requirements are never buried in the middle, and
there is no `Ответ:` cue: these prompts go to instruct models, where the user
turn is closed by an end-of-turn token.

### Dataset creation

Questions are produced by a deterministic generator — the same seed rebuilds the
dataset byte for byte.

1. **Catalogue.** 61 requirement types over five families (`structure`,
   `lexical`, `style`, `format`, `logic`). Each carries a deterministic verifier,
   three to five Russian wordings, an English mirror, and declared incompatibility
   groups, so a stack can never contain two requirements that contradict each
   other. Wordings state their own counting convention wherever a person might
   count differently.
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
5. **Free-rate filtering.** Every type was measured against a corpus of 162
   unconstrained model answers to the dataset's own tasks: how often is the
   requirement satisfied by accident, without being asked? Every shipped type
   sits under the 0.15 bar. Conditional types are measured per branch, because
   averaging the two would hide the only number that decides whether the type
   ships.
6. **Validation.** Every witness is re-scored with the shipped scorer; every
   verifier is checked to reject a targeted corruption of a passing answer; every
   conditional's branch is re-derived from the context it ships with. All of it
   runs at build time rather than being assumed.

### Limitations

- Human annotation of a ≥100-question sample is the outstanding acceptance
  check. Every question carries a constructive witness, which is a stronger
  mechanical guarantee, but it is not the same thing.
- **The false branch of a conditional requirement is a control, not a
  difficulty**: a model that never read the rule passes it, because the word it
  forbids was not going to appear anyway. Branches ship 69 % true.
- Two types are implemented and verified but thin in the shipped build:
  `lexical:sentence_acrostic` (14 uses) and `lexical:epiphora` (1). They compete
  for the same slot as the numeric totals every stack carries; reports should
  mark such cells rather than quote them.
- The dataset is single-turn. It says nothing about whether a rule stated once
  survives the turns that follow.

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
  is excluded. **It is read against `constraint_pass_rate`, not against
  `sample_pass_rate`** — both are built from per-requirement verdicts, while the
  headline is a product over the whole stack and is structurally far lower.
- `constraint_pass_rate` — share of individually satisfied requirements; a
  partial-credit diagnostic separating "missed one requirement of five" from
  "missed all of them". **Computed over non-empty responses only**: per sample the
  value is `null` for an empty response and a custom aggregation averages the
  rest. A reasoning model that spends its whole token budget in the trace returns
  an empty answer, which honestly fails the headline metric — but zeroing five
  constraint verdicts for it would turn the diagnostic into a proxy for token
  budgeting. Because empty responses are excluded here and counted as failures
  there, the share of empty responses belongs next to the three metrics whenever
  they are quoted.

Because `sample_pass_rate = Π pᵢ` over the requirements of a question, the
stack-size distribution bounds the score directly: at a per-requirement pass
probability of 0.80 the ceiling is 0.33 for a five-requirement stack, and 0.37 at
the shipped mean of 4.43 requirements per question.

Reasoning traces are stripped before verification: everything up to and
including the first `</think>` is dropped. The trace is delimited by its
**closing** tag rather than by a balanced `<think>...</think>` pair, because the
opening tag often never reaches the generation — chat templates for reasoning
models routinely pre-fill `<think>` at the end of the prompt, so the model emits
only `trace</think>answer`, and a server that re-inserts a separately returned
trace can leave a leaked token in front of it. Any further balanced blocks are
removed too, so a model that interleaves several traces keeps the answer between
them. Beyond that the response is scored verbatim, with no lenient extraction of
an "answer part". An **unterminated** trace — the model was cut off inside its
own reasoning, so there is no closing tag — leaves its text as the answer and
fails: the answer is not absent, it is wrong.

### Human baseline

TODO
