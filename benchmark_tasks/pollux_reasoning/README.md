# POLLUX Reasoning

This task evaluates `pollux_reasoning`, a MERA-format subset derived from
`ai-forever/POLLUX-instructions`.

The dataset contains only examples where:

- `meta == "Technical Problems"`
- `difficulty.lower() == "hard"`

The generated answer is scored by `llm_as_judge`: Pollux Judge is called once per criterion from
the sample `criteria` list using the sample `reference_answer`. The final metric is
`mean(raw 0/1/2 criterion scores) / 2`.

Scoring requires the `openai` Python package and an OpenAI-compatible vLLM endpoint.
