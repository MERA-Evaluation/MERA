# MMReD: Dense Context Reasoning Benchmark

MMReD (Multi-Modal Reasoning in Dense context) is a benchmark for evaluating long-context reasoning in vision-language models. This integration provides the **Dense Context (DC)** subset for the MERA leaderboard.

## Tasks

| Task Code | Question Type | Description |
|-----------|--------------|-------------|
| `DC-SA-C` | spend_alone | Who spent the most/least time alone? |
| `DC-SR-I` | steps_in_room | How many steps did X spend in room Y? |
| `DC-CC-I` | crowd_count | How many times did a crowd appear? |
| `DC-WS-R` | where_spend | In which room did X spend the most/least time? |
| `DC-WHS-C` | who_spend | Who spent the most/least time in room X? |

Each task is evaluated at three sequence lengths: **32**, **64**, **128**.

## Evaluation

### Input Format
- **Context**: Sequence of images showing room occupancy at each step
- **Question**: Natural language question about the sequence
- **Answer**: Single word (person/room name) or number

### Metric
- **Exact Match** with case-insensitive comparison and answer normalization
- **Aggregate**: Weighted average with 2× weight on length-128 samples

## Usage

```bash
# Run evaluation on a model
lm_eval --model hf \
    --model_args pretrained=YOUR_MODEL \
    --tasks mmred \
    --batch_size auto
```

## Files

```
mmred/
├── _group.yaml          # Task group configuration
├── mmred_base.yaml      # Base task configuration
├── mmred_dc_*_*.yaml    # Individual task configs (15 files)
├── utils.py             # Answer extraction & metrics
└── README.md            # This file
```

## Citation

```bibtex
@article{mmred2024,
  title={MMReD: A Cross-Modal Benchmark for Dense Context Reasoning},
  author={...},
  year={2024}
}
```
