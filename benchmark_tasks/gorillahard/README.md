# The GorillaHard dataset

## Description

GorillaHard measures whether a model can turn a request into an executable decision over a tool catalog. Each of the 1169 Russian questions gives the model a catalog of 14–22 tools, one or two attached files, a request over them and a block stating the required answer format. The answer must be a single JSON object of one of five kinds: a call `{"tool": ..., "args": {...}}`, an ordered plan `{"plan": [...]}` whose later steps reference an earlier result as `"$1"`, independent calls `{"calls": [...]}`, a clarifying question `{"clarify": ...}`, or a refusal `{"abstain": true, "reason": ...}`.

Arguments cannot be copied out of the question — they have to be found in the attached file or computed from it; a twin of the right tool is always in the catalog; some requests are unsatisfiable and the only way to see it is to count; files carry format decoys and planted instructions. Fifty-five of the rows are turns of twenty-five multi-turn dialogues.

Scoring is a pure function of the generated text (`utils.py`): no LLM judge, no paraphrase matching, no randomness, so a run is exactly reproducible and every failure traces back to a specific violated requirement. All 1169 reference answers score `sample_pass_rate` 1.0 through this scorer, and every one of the 25 dialogues scores `dialog_pass_rate` 1.0.

## Homepage

https://mera.a-ai.ru

## Tasks

| Task name | Dataset source | Test size | N-shots | Headline metrics |
| --- | --- | --- | --- | --- |
| `gorillahard` | `MERA-evaluation/GorillaHard` | 1169 | 0 | `sample_pass_rate` / `balance_score` |

`num_fewshot: 8` in the YAML is **not** a few-shot count: the task is zero-shot, and the few-shot machinery is reused to carry dialogue history. `dialog_sampler.py` returns the previous turns of the same dialogue instead of random examples, and 8 is the ceiling on how many of them are kept. A single-turn question gets an empty history.

## Running

Both flags are mandatory. `--fewshot_as_multiturn` is what splits the history into separate `user`/`assistant` messages; without it the history collapses into one message, the run does not fail, and the numbers measure something else.

```bash
lm_eval --model hf --model_args pretrained=<model> \
        --tasks gorillahard --include_path ./benchmark_tasks \
        --apply_chat_template --fewshot_as_multiturn \
        --output_path ./mera_results --log_samples
```

Against an OpenAI-compatible endpoint:

```bash
lm_eval --model local-chat-completions \
        --model_args "model=<model>,base_url=<url>/v1/chat/completions,num_concurrent=8,tokenized_requests=False,timeout=900" \
        --tasks gorillahard --include_path ./benchmark_tasks \
        --apply_chat_template --fewshot_as_multiturn \
        --output_path ./mera_results --log_samples
```

`fewshot_split` must stay equal to `test_split`: only under that condition does lm-eval hand the sampler the current document, and without it there is nothing to build the history from. The task works unchanged on the MERA fork of lm-eval 0.4.9.2 and on upstream 0.4.13 — `dialog_sampler.py` implements both sampler surfaces, and the rendered prompts are byte-identical across the two.

Prompts are long: 10.6k to 34.2k characters, median 18.5k. Against a local server, raise the context window explicitly (Ollama, for example, defaults to 4096 tokens and truncates silently) — otherwise the run measures the truncation, not the model.

## Scoring notes

* **Reasoning traces are stripped.** `<think>...</think>` is removed before checking, and an unmatched `<think>` — the trace of a generation cut off by the token limit — is removed together with everything after it, so a truncated answer is not scored as a malformed one.
* **No lenient extraction.** Beyond stripping reasoning traces and outer whitespace the response is scored verbatim: unwrapping a code fence or peeling off a preamble would forgive exactly the violations the format block forbids.
* **Denominators are honest.** A metric that does not apply to a question is omitted rather than set to zero, so `tool_match_rate` stays a share of the 1032 questions requiring a call, `abstention_recall` a share of the 127 refusal questions and `clarify_recall` a share of the 10 clarification questions.
* **Format and content are never mixed.** A refusal where a call was expected is a decision error, counted once under `sample_pass_rate` and the abstention metrics; it never inflates the format failure count.
* **Answers are stripped in the copy on the Hub.** `outputs` is empty there, so a run against it reports only `format_pass_rate` and `constraint_pass_rate`, and the dialogue history arrives with empty assistant turns. The content metrics need a copy with reference answers; the full evaluation is run by the benchmark organiser.

## Metrics

| metric | meaning |
| --- | --- |
| `sample_pass_rate` | format and content both right — headline |
| `balance_score` | geometric mean of the pass rate over the 16 difficulty levers, floored at 0.01 — capability coverage, independent of how the set is composed |
| `dialog_pass_rate` | dialogues in which every turn passed; denominator is the 25 dialogues |
| `format_pass_rate` | the answer-format block fully obeyed |
| `constraint_pass_rate` | share of the nine atomic format requirements satisfied |
| `tool_match_rate` / `args_match_rate` | right tool / right tool and every argument |
| `abstention_recall` / `false_abstention_rate` | refusals recognised / refusals where a call was required |
| `clarify_recall` / `false_clarify_rate` | the same for clarifications |
| `tool_in_catalog_rate` | every named tool exists in the question's catalog — a wrong choice versus an invented name |
| `cost_optimal_rate` | the right side of a "same capability, one caveat" pair (6 questions) |
| `injection_resistance_rate` | the call demanded by a planted instruction was not made (6 questions) |

## Dataset files

`datasets/GorillaHard/` holds the documentation (`README.md`, `README_ru.md`, `dataset_meta.json`) and the upload notebook. The question files `test.json` and `shots.json` are **not** committed to the repository — they live only on the Hub, which is what `dataset_path` in `gorillahard.yaml` points at.

## License

MERA License
