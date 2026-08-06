# SOB-Hard: Structured Output Benchmark Hard

## Task description

SOB-Hard measures structured output between machine formats: the model is given
one or two real documents and has to produce a document in another notation,
packaged exactly as the prompt specifies.

825 questions in Russian, 11 families of work × 3 difficulty levels × 25
questions per cell. Fifteen source notations, six target ones. Every input
document is a real file from an open source.

Nothing is judged by a language model: every rule the prompt states is checked
by a separate deterministic verifier — 31 constraints in four layers
(packaging, the parse of the target notation, the comparison of values, the
family's semantics). The reading convention of each notation is stated in the
prompt itself.

Model skills tested: Structured output, Format conversion, Format control,
Instruction following, Long-form exact generation

Contributors: Artem Chervyakov

## Motivation

**Which models this is for.** Instruct models that get put into a data
pipeline. The set aims at strong models: the easiest level was removed because
it handed frontier models a steady bonus while ranking nobody. It does not suit
base models without instruction tuning, or models with a short context: half
the questions are documents over twelve thousand characters, and the answer is
often longer than the input.

**Who the results are for.** People deciding whether a model can sit between
two systems where the format cannot be broken. Such a reader takes
`sample_pass_rate` as the share of documents that would pass downstream without
a hand edit, and `balance_score` as the answer to a different question — is
there a class of work where the model is useless outright.

**Which abilities it evaluates.** Not text comprehension, but holding a
structure through a rewrite: read a document under a stated convention, apply a
stated operation, and write the result out in another notation in full, without
a single divergence. Composition is evaluated separately — edits, then a
rewrite of their result, then a change of notation — as is the ability to carry
the intermediate state before the first character of the answer is written.

**Why the questions are built this way.** The answer is a whole document rather
than a label, so scoring does not reduce to guessing: one wrong value out of a
thousand is visible and is named. Each notation's reading convention is written
in the prompt, so the task does not test knowledge of foreign standards — an
engineer reading the prompt can answer without guessing at anything. Validity
rests on two facts: every question's reference passes all applicable checks at
1.0 under the same scorer, and the shortcuts (handing the input back, an empty
document, a pipeline abandoned halfway) score exactly zero.

**Why these metrics.** The headline metric is binary per question: a document
that breaks even one rule is no good downstream, so partial credit would
mislead here. But an arithmetic mean cannot tell an even model from an uneven
one, and for a pipeline those are different things — hence the second metric, a
geometric mean over the cells, which penalises failing a whole class of work.

## Data description

### Data fields

- `instruction` [str] — the prompt template with slots for the question's parts
- `inputs` [dict] — the four prompt blocks:
  - `task` [str] — what the work is and in which notations
  - `input_data` [str] — the source document
  - `format` [str] — the packaging requirements and the comparison rules
  - `question` [str] — the specific instruction
- `outputs` [str] — the reference answer in a code block with the right tag
- `meta` [dict] — service fields: the identifier, the reference and its hash,
  the task parameters, the canonical structures the verifiers need, the
  categories (family, difficulty, notations, length, register); multi-turn
  questions additionally carry `dialogue_id`, `turn_index`, `n_turns`

### Prompts

Four blocks with text labels, always in the same order, the question last:

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

20 templates: five registers of speech in four editions each, spread evenly
inside every cell. On a full run the spread across registers came out inside
the sampling noise.

### Multi-turn questions

The `session` family is 21 dialogues, 75 turns. The first turn shows the
document; every later turn works on the previous turn's result, which is not
repeated in the prompt. The history holds the reference answers, so every turn
is reproducible and is scored on its own. Run with `--apply_chat_template
--fewshot_as_multiturn`; the other 750 questions stay single-turn.

### Dataset creation

**Sources.** Real files from open repositories — configurations, data dumps,
tables, schemas. Selection favoured byte dirt: emoji and surrogate pairs,
escaped quotes and line breaks inside values, number-strings, near-twin keys.
The provenance of every file is recorded in `corpus/`.

**Selection.** A document is admitted only when its data model fits in the two
or three sentences that can be written into the prompt. Anything that reads
ambiguously — mixed content in XML, scalars whose type YAML decides for itself
— is refused by the reader rather than explained away in the prompt.

**Assembly.** Questions are generated by a program: the document is rendered in
the required notation, the operation is applied, the reference is written and
then **read back** — if the parse does not match the structure it was built
from, the question is discarded. The reference is then scored by the same
scorer that later grades the model, and is accepted only at 1.0 on every
applicable constraint. Each dialect's semantics is implemented twice — in the
generator and in the scorer, independently — and the validator reconciles both
implementations on every question.

**Validation of the set.** Before release the following run: the generator's
self-test (every reader, writer, rule and verifier in accept-and-reject pairs);
a check that the data matches the prompts; a probe for shortcuts; and an audit
of the measuring instrument itself — every constraint not only accepts the
reference but also rejects something, the metrics stay inside their own bounds,
scoring does not depend on key order where the question does not grade it, and
line endings, byte-order marks and trailing whitespace change no verdict.

## Evaluation

### Metrics

- **`sample_pass_rate`** — the share of questions where every applicable
  constraint holds at once. The headline metric.
- **`balance_score`** — the geometric mean over the 33 family × difficulty
  cells. A cell at zero counts as 0.01, so one failed cell multiplies the score
  by 0.87 rather than zeroing it. A model scoring an even 0.60 in every cell
  gets 0.60; a model averaging 0.64 with three failed cells gets 0.48.
- `constraint_pass_rate` — the share of satisfied constraints, partial credit.
- `format_pass_rate` — packaging: one code block, the right tag, nothing
  outside it.
- `content_pass_rate` — content: structure, numbers within tolerance, strings
  byte for byte.
- `task_pass_rate` — the family's semantics.

### Human baseline

Not measured. Every constraint is deterministic and checked by program, and
each question's reference scores 1.0 by construction.
