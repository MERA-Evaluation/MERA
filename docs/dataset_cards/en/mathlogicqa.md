# **MathLogicQA**

## Task Description

The task is to solve mathematical problems formulated in natural language.

Mathematical problems can be divided into several types:

- forming and solving equations,
- forming and solving systems of equations,
- solving problems on proportions and comparison,
- comparing the objects described in the problem with the variables in the equation.

### Motivation

The goal of the task is to analyze the ability of the model to solve mathematical tasks using simple operations such as addition, subtraction, multiplication, division, and comparison operations.

## Dataset Description

Each dataset sample consists of the problem text and 4 answer options, only one of which is correct.

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format. All used products are presented in the project repository;
- `inputs` is a dictionary containing input data for the model:
    - `id` is an integer indicating the index of the example;
    - `option_a` is a string containing answer option A;
    - `option_b` is a string containing answer option B;
    - `option_c` is a string containing answer option C;
    - `option_d` is a string containing answer option D;
- `outputs` is a string containing the letter of the correct answer;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example;
    - `task` is a string containing information about the task type: `math` includes solving systems of equations and comparing quantities, `logimath` includes matching the objects described in the problem with the variables in the equation and solving it.

### Data Instances

Below is an example from the dataset:

```json
{
	"instruction": "{text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nУкажите только букву правильного ответа.\nОтвет:",
	"inputs": {
		"text": "Если из 17 вычесть 26, то получится 3, умноженное на q. Рассчитайте значение переменной q.",
		"option_a": "-3",
		"option_b": "3",
		"option_c": "14",
		"option_d": "14.3"
	},
	"outputs": "A",
	"meta": {
		"id": 1,
		"task": "math"
	}
}
```

### Data Splits

The train set consists of  `680` examples. The test set consists of `1143` examples. Train and test sets are balanced in class labels.

### Prompts
10 prompts of varying difficulty were created for this task. Example:

```json
"Решите математичеcкую задачу: {text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nВыберите один правильный ответ. В ответе укажите только букву правильного ответа.\nОтвет:"
```

### Dataset Creation

The dataset includes two types of problems: `logic` and `math`.

#### logic

Logic problems are mathematical problems formulated in natural language. To solve this type of problem, it is necessary to construct a system of equations (or one equation) and solve it by comparing the objects described in the problem with the variables in the equation. Problems of this type were formed using open sources containing databases of mathematical problems.

#### math

Math problems consist of a mathematical expression (a linear equation or a system of linear equations) and a question about that expression. One must solve a linear equation or system of linear equations to answer the question. For some tasks, it is also necessary to perform a comparison operation. Mathematical expressions are synthetic data generated using an open-source library using the linear_1d and linear_2d modules. The resulting generated expressions were manually rewritten by experts from mathematical language into natural Russian. Next, the experts formulated a question in natural language and the correct answer for each expression.

When creating the dataset, experts added instructions in natural language to some tasks. The experts also formulated 3 incorrect answer options for each task from the dataset.

### Validation

All examples from the dataset have been validated on the Yandex.Toloka platform. Tolokers checked the correctness of the problem conditions and the answer. The dataset included 2000 examples of type `math` and 570 examples of type `logic`. Each example had a 3-person overlap, which could increase to 5 if the agreement on the task answer was below 70%. The responses of the Toloka annotators who showed labeling accuracy of less than 50% on control tasks were excluded.

As a result of validation, the final test set included examples with complete consistency between the annotators. The training set included the remaining examples with agreement above 60%.

## Evaluation

### Metrics

Models’ performance is evaluated using the Accuracy score. The choice of this metric was due to the balance of classes.

### Human Benchmark

Human-level score is measured on a test set with the Yandex.Toloka project with the overlap of 5 reviewers per task. The human accuracy score is `0.99`.
