# POLLUXReasoning

POLLUXReasoning evaluates open-ended reasoning and technical problem solving on the hard Technical Problems subset of [`ai-forever/POLLUX-instructions`](https://huggingface.co/datasets/ai-forever/POLLUX-instructions), saved in MERA dataset format.

## Dataset creation

The dataset is created by filtering the `train` split of `ai-forever/POLLUX-instructions` to retain only rows with `meta == "Technical Problems"` and `difficulty.lower() == "hard"`. Five examples are placed into `shots.json`, and the remaining examples are stored in `test.json`. Only criteria with `0/1/2` rubrics are used.

## Evaluation

Model outputs are evaluated with Pollux-4B-Judge through an OpenAI-compatible vLLM endpoint. The final metric is `mean(raw criterion scores) / 2`.

## Human Baseline

No human baseline is included yet. The placeholder value is `0.0`.

## Contributors

MERA contributors
