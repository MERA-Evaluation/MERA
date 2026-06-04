# POLLUXHumanCentric

POLLUXHumanCentric evaluates open-ended instruction following on the hard Human Interaction subset of [`ai-forever/POLLUX-instructions`](https://huggingface.co/datasets/ai-forever/POLLUX-instructions), saved locally in MERA dataset format.

Each sample is a user prompt wrapped into one of several MERA-style instruction templates and evaluated with an LLM judge using its reference answer and criteria.

## Motivation

This dataset targets instruction-tuned generative models and measures how well they handle human-centric open-ended requests. It is useful when exact-match metrics are not meaningful and the main concern is answer quality.

## Dataset creation

The dataset is created by filtering the `train` split of `ai-forever/POLLUX-instructions` to retain only rows with `meta == "Human Interaction"` and `difficulty.lower() == "hard"`. Five examples are placed into `shots.json`, and the remaining examples are stored in `test.json`. Source `criteria` and `reference_answer` are exposed as top-level fields and are passed to Pollux Judge during scoring.

## Evaluation

Model outputs are evaluated with Pollux-4B-Judge through an OpenAI-compatible vLLM endpoint. The judge is called once for each criterion in the sample `criteria` list using the rendered instruction, model answer, `reference_answer`, criterion name, and criterion rubric. Only criteria with `0/1/2` rubrics are used; the final metric is `mean(raw criterion scores) / 2`.

## Human baseline

No human baseline is included yet. The placeholder value is `0.0`.

## Contributors

MERA contributors
