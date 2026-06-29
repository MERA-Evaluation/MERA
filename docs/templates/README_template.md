f"""# {meta["dataset_name"]}


## Task description

{custom["Task description"]}

Evaluated skills: {computed["skills"]}{computed["contributors"]}


## Motivation

{custom["Motivation"]}


## Data description

### Data fields

Each dataset question includes data in the following fields:

{computed["data_field_descriptions"]}


### Data formatting example

```json
{computed["data_example"]}
```


### Prompts

For the task, {len(meta["prompts"])} prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
{computed["prompts"]}
```


### Dataset creation

{custom["Dataset creation"]}


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

{computed["metrics"]}
"""