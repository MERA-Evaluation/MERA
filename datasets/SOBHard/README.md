# SOBHard

## Task Description

SOBHard measures structured output between machine notations. The model is given
a real document — sometimes two — and must return a document in another notation,
packaged exactly as the prompt demands.

825 questions in Russian: 11 families of work × 3 difficulty levels × 25
questions per cell. That grid is the one axis that is balanced exactly; the
notations, the lengths and the corpus are not, on purpose — see "What is
balanced". Fifteen source notations (`csv`, `fwf`, `html`, `ini`, `json`,
`jsonc`, `jsonl`, `logline`, `mdtable`, `properties`, `scsv`, `toml`, `tsv`,
`xml`, `yaml`) and six target ones (`json`, `jsonl`, `yaml`, `toml`,
`properties`, SQL `INSERT`).

| Family | What is asked |
|---|---|
| `convert` | the same document in another notation, data unchanged |
| `extract` | collect the values at listed paths into one object |
| `repair` | fix a broken document, then rewrite it by a rule |
| `transform` | rewrite the document by a rule of several ordered steps |
| `schema_gen` | a draft-07-style schema of what the document becomes after a rule |
| `patch` | apply a numbered list of `set` / `delete` / `insert` operations |
| `diff` | the difference between two versions, as `added`/`removed`/`changed` |
| `apply_diff` | apply such a difference object to version 1 |
| `merge` | join two sources on a stated key |
| `template` | render one line per record from a text template |
| `session` | a chain of steps spread over several turns of a conversation |

Difficulty is length plus composition: an easy question is one operation over
some 7 000 characters, a hard one a chain of steps over some 16 500. Just under
half the sources run past 12 000 characters, the longest to 62 741; in just over
a quarter of the questions the answer is longer than the source.

Nothing is judged by a language model. Every rule the prompt states is checked by
a separate deterministic verifier — 31 constraints in four layers: packaging (4),
the parse of the target notation (1), the comparison of values (3), the family's
own semantics (23). Each notation's reading convention is written into the prompt,
so no knowledge of a foreign standard is required, and a run is exactly
reproducible.

In nine of the 825 questions the reference answer is contained verbatim in the
input document: these are questions where the correct action is to return an
already-correct fragment unchanged. They are kept in the set on purpose — they
test that the model does not break valid data.

Evaluated skills: Structured output, Format conversion, Format control,
Instruction following, Long-form exact generation

Contributors: Artem Chervyakov

## Motivation

**Who this is for.** Instruct models that get put into a data pipeline, and the
people deciding whether one can sit between two systems where the format cannot
be broken. Such a reader takes `sample_pass_rate` as the share of documents that
would pass downstream without a hand edit, and `balance_score` as the answer to a
different question — whether there is a class of work the model cannot do at all.
It does not suit base models or models with a short context.

**What it evaluates.** Not text comprehension, but holding a structure through a
rewrite: read a document under a stated convention, apply a stated operation,
write the result out in another notation in full and without a single divergence.
Composition and carrying an intermediate state across turns are evaluated
separately. Because the answer is a whole document, scoring does not reduce to
guessing: one wrong value in a thousand is visible and is named, and the
shortcuts score exactly zero — handing the input back, an empty document, a chain
abandoned halfway.

**Why two headline metrics.** The first is binary per question: a document that
breaks even one rule is no good downstream, so partial credit would mislead. But
an arithmetic mean cannot tell an even model from an uneven one, and for a
pipeline those are different things — hence a geometric mean over the cells,
which penalises failing a whole class of work.

## Data Description

Two splits, divided by role rather than by size. `test` holds the 825 graded
questions. `shots` holds the 203 rows that may be shown with their answers: ten
demonstrations, and the 193 earlier turns of the 75 conversations. Those answers
are the history a graded turn continues.

### What Is Balanced

One axis is exact, and the rest are not. Reading a rate off an axis that is not
balanced is fine where the levels are large and misleading where they are not,
so here are the counts.

**Exact.** Family × difficulty: 33 cells of exactly 25 questions, 275 per
difficulty level. This is what `balance_score` averages over, and the only slice
that supports comparing cells against each other directly.

**Even to within a question.** The prompt register: five registers, and in 30 of
the 33 cells each appears exactly five times. The three `session` cells are
uneven — 4 to 6 per register — because a conversation speaks in one voice, so the
register belongs to the whole chain, and 25 turns come from 25 chains of
different lengths. Over the whole split the five registers are 164 to 167.

**Deliberately uneven.** Document length, because length is half of what
difficulty means here: 50 short, 275 medium, 500 long, and the mix moves with the
tier — easy is 50/145/80, medium 0/100/175, hard 0/30/245. The mean input grows
9 700 → 14 000 → 17 200 characters.

**Uneven, and not a target.** The notations follow what the corpus can carry:

| | most | least |
|---|---|---|
| source | `yaml` 153, `json` 134, `jsonl` 88 | `ini` 8, `properties` 4 |
| target | `yaml` 273, `json` 242, `jsonl` 157 | SQL `INSERT` 37, `properties` 3 |

A rate over `properties` as a target is three questions wide and means nothing;
`ini` and `properties` as sources are barely more. The corpus is uneven the same
way: 114 files, a median of 5 questions each, and one table behind 93 of them.

### Data Fields

- `instruction` [str] — the prompt template with slots for the question's parts
- `inputs` [dict] — the four prompt blocks: `task` (what the work is and in which
  notations), `input_data` (the source document, or two under fixed labels),
  `format` (the packaging requirements and the comparison rules), `question` (the
  specific instruction)
- `outputs` [str] — the reference answer in a code block with the right tag.
  Filled in on `shots`, empty on every graded row
- `meta` [dict] — `id` and `base_id`; `fence_tag`, the tag the answer's block must
  carry; `reference`, the reference document as raw text — **scoring reads the
  reference from here, not from `outputs`** — and its `reference_sha256`;
  `task_meta` and `checks`, JSON strings holding the question's own parameters
  and the canonical structures the verifiers derive the answer from;
  `dialogue_id`, `turn_index`, `n_turns`, the coordinates of a turn in its
  conversation and `null` elsewhere; and `categories` — family, difficulty,
  language, source and target notations, length tier, prompt register, corpus
  file

### Prompts

Four blocks with text labels, always in the same order, the instruction last:

```
Задача:
<what the work is and in which notations>

Входные данные:
<the source document>

Формат ответа:
<the packaging requirements and the comparison rules>

Инструкция:
<the specific instruction>
```

20 templates: five registers of speech — `casual`, `request`, `spec`, `formal`,
`command` — in four editions each. How they are spread is under "What is
balanced".

### Multi-turn Questions

The `session` family is 75 conversations over 268 turns — thirty-six of three
turns, thirty-five of four, four of five. The first turn shows the document;
every later turn works on the previous turn's result, which is **not** repeated
in the prompt.

One turn of a conversation is graded, and it is the last one. That is not a
choice but a consequence: the state a turn starts from is the answer to the turn
before it, so grading two turns of one conversation would mean publishing the
answer to one of them. The 193 earlier turns are in `shots` with their answers,
and the few-shot sampler replays them as the graded turn's history.

Underneath, a turn is an ordinary question of an ordinary family — 165 of the 268
are `patch` work, 103 `transform` — checked by that family's constraints plus one
more: an answer that merely restates the state already on the table is refused.

Such a run must be started with `--apply_chat_template --fewshot_as_multiturn`.
The few-shot count is an upper bound on the number of replayed turns, not a
number of examples: history goes to the graded turns of conversations only, and
the other 750 questions stay single-turn under the same command.

### Dataset Creation

**Sources.** 114 real files from open repositories — configurations, data dumps,
tables, schemas — read as 139 distinct documents, since a large file is entered
through one of its own subtrees. Selection favoured byte dirt: emoji and
surrogate pairs, escaped quotes and line breaks inside values, number-strings,
near-twin keys. A document is admitted only when its data model fits in the two
or three sentences that can be written into the prompt; anything that reads
ambiguously — mixed content in XML, scalars whose type YAML decides for itself —
is refused rather than explained away. How much each file supplies was not
levelled: the median is five questions, the largest table stands behind 93.

**Assembly.** For each question the document is rendered in the source notation,
the operation applied, and the reference written out and then **read back**: if
the parse does not match the structure it was built from, the question is
discarded. The reference is then scored by the same checks that later grade the
model and kept only at 1.0 on every applicable constraint. Each dialect's
semantics is implemented twice, independently, and the two are reconciled on
every question.

**Before release.** Every constraint is shown to reject something as well as
accept a reference, the metrics stay inside their own bounds, and key order, line
endings, byte-order marks and trailing whitespace change no verdict where the
question does not grade them. One check exists for the split by role: no graded
answer occurs anywhere in `shots` — not in an answer, not inside an earlier
turn's prompt.

## Evaluation

### Metrics

- **`sample_pass_rate`** — the share of questions where every applicable
  constraint holds at once. The headline metric.
- **`balance_score`** — the geometric mean over the 33 family × difficulty cells.
  A cell at zero counts as 0.01, so one dead cell multiplies the score by 0.87
  rather than zeroing it. A model at an even 0.60 in every cell scores 0.60; a
  model averaging 0.64 with three dead cells scores 0.48.
- `constraint_pass_rate` — the share of satisfied constraints; partial credit
  that separates one broken rule from an ignored format.
- `format_pass_rate` — packaging: one code block, the right tag, nothing outside
  it, a body that parses as the target notation.
- `content_pass_rate` — content: the structure matches the reference, numbers are
  within a 1e-6 relative tolerance, strings match byte for byte.
- `task_pass_rate` — the family's own semantics: what a comparison with the
  reference cannot see.
