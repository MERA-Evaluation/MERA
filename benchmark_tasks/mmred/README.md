# MMReD: Dense Context Reasoning Benchmark

MMReD (Multimodal Reasoning Evaluation Dataset) is a synthetic benchmark for evaluating long-context dense reasoning in language models. This integration provides the **Dense Context (DC)** subset for the MERA leaderboard.

## Tasks

| Task Code | Question Type | Description |
|-----------|--------------|-------------|
| `DC-SA-C` | spend_alone | Who spent the most/least time alone? |
| `DC-SR-I` | steps_in_room | How many steps did X spend in room Y? |
| `DC-CC-I` | crowd_count | How many times did a crowd (3+ people) appear? |
| `DC-WS-R` | where_spend | In which room did X spend the most/least time? |
| `DC-WHS-C` | who_spend | Who spent the most/least time in room X? |

Each task is evaluated at three sequence lengths: **32**, **64**, **128** steps.

## Evaluation

- **Input**: A textual sequence of room occupancy states followed by a question.
- **Output**: A single word (person or room name) or a number.
- **Metric**: Exact Match with case-insensitive comparison and answer normalization.
- **Aggregate**: Weighted average — 1× weight for length 32, 2× for 64, 4× for 128.

## Usage

```bash
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL \
    --tasks mmred \
    --batch_size auto
```

## Files

```
mmred/
├── _group.yaml          # Task group with weighted aggregate metric
├── mmred_base.yaml      # Base task configuration (inherited by all subtasks)
├── mmred_dc_*_*.yaml    # Individual subtask configs (15 files)
├── utils.py             # Answer extraction and metric computation
└── README.md            # This file
```

## Homepage

https://mera.a-ai.ru

## License

Apache-2.0
