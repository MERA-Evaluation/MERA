# **LCS**

## Task Description

The longest common subsequence is an algorithmic task from [BIG-Bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/cs_algorithms/lcs). This problem consists of pairs of strings as input, and language models are expected to predict the length of the longest common subsequence between them correctly.

LCS is a prototypical dynamic programming problem and this task measures the model's ability to capture that approach.

**Keywords:** algorithms, numerical response, context length

**Authors:** Harsh Mehta, Behnam Neyshabur

### Motivation

Recently, large language models have started to do well on simple algorithmic tasks like few-shot arithmetic, so we want to extend this evaluation to more complicated algorithms.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is an example of two sequences to be compared;
- `outputs` is a string containing the correct answer, the length of the longest common subsequence;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Запишите в виде одного числа длину самой длинной общей подпоследовательности для следующих строк: \"{inputs}\".\nОтвет:",
    "inputs": "RSEZREEVCIVIVPHVLSH VDNCOFYJVZNQV",
    "outputs": "4",
    "meta": {
        "id": 138
    }
}
```

### Data Splits

The public test includes `320` examples, and the closed test set includes `500` examples.

### Prompts

10 prompts of varying difficulty were created for this task. Example:

```json
"Решите задачу нахождения длины наибольшей общей подпоследовательности для следующих строк:\n\"{inputs}\"\nОтвет (в виде одного числа):".
```

### Dataset Creation

Sequences of length in the range [4; 32) were generated with a Python script for open public test and closed test sets.

For the open public test set we use the same seed for generation as in the Big-Bench.

## Evaluation

### Metrics

The task is evaluated using Accuracy.

### Human Benchmark

The human benchmark is measured on a subset of size 100 (sampled with the same original distribution). The accuracy for this task is `0.56`.
