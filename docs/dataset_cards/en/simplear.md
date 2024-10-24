# **SimpleAr**

## Task Description

Simple arithmetic is a mathematical task from [BIG-Bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/simple_arithmetic). The task itself tests language models' basic arithmetic capabilities by asking them to perform n-digit addition for a range of n.

**Warning:** This is a diagnostic dataset with an open test and is not used for general model evaluation on the benchmark.

**Keywords:** arithmetic, example task, free response, mathematics, numerical response, zero-shot

### Motivation

The goal of the task is to analyze the ability of the model to solve simple mathematical addition tasks.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is the example of arithmetic expression;
- `outputs` is a string containing the correct answer of summation of two numbers;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Напишите ответ для математического выражения.\n{inputs}",
    "inputs": "663 + 806 = ",
    "outputs": "1469",
    "meta": {
        "id": 412
    }
}
```

### Data Splits

The train set consists of `1000` examples of arithmetic expressions. The test set consists of `1000` examples of arithmetic expressions.

### Prompts

The number of prompts used for the task is 10. Example:

```json
"Реши математическую задачу на сложение чисел. Выведи ответ в формате \"number\", где number - число, которое является результатом сложения.\nОтвет:"
```

### Dataset Creation

N-digit addition was created for n in the range [1;5] for both train and test sets.

## Evaluation

### Metrics

The task is evaluated using the Exact Match (EM). For each example, 1.0 is given for the target sequence that EXACTLY matches the predicted sequence. Else, 0.0.

### Human Benchmark

The human benchmark is measured on a subset of size `200` (sampled with the same original distribution). The final score equals `1.0`.
