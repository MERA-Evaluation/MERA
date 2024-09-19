# **MaMuRAMu**

## Task Description

**Massive Multitask Russian AMplified Understudy  (MaMuRAMu)** is a dataset designed to measure model professional knowledge acquired during pretraining in various fields. The task covers 57 subjects (subdomains) across different topics (domains): HUMANITIES; SOCIAL SCIENCE; SCIENCE, TECHNOLOGY, ENGINEERING, AND MATHEMATICS (STEM); OTHER. The dataset was created based on the English MMLU proposed in [1] and follows its methodology in instruction format. Each example contains a question from one of the categories with four possible answers, only one of which is correct.

**Warning:** to avoid data leakage for MaMuRAMu, we created the NEW closed dataset that follows the original MMLU design. Thus, **results on the MMLU and MaMuRAMu datasets cannot be directly compared with each other.**

**Keywords**: logic, world knowledge, factual, expert knowledge

### Motivation

This set is a continuation of the idea GLUE [2] and SuperGLUE [3] benchmarks, which focus on generalized assessment of tasks for understanding the language (NLU). Unlike sets like ruWorldTree and ruOpenBookQA (where questions are similar to MMLU format), which cover tests of the school curriculum and elementary knowledge, MaMuRAMu is designed to test professional knowledge in various fields.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is a dictionary that contains the following information:
    - `text` is the test question;
    - `option_a` is the option A;
    - `option_b` is the option B;
    - `option_c` is the option C;
    - `option_d` is the option D;
    - `subject` is the topic of the question (generalization of a group of subdomains by meaning);
- `outputs` is the result: can be one of the following string variables: "A", "B", "C", "D";
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example;
    - `domain` is question subdomain.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Задание содержит вопрос по теме {subject} и 4 варианта ответа A, B, C, D, из которых только один правильный.\n{text}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}\nЗапишите букву правильного ответа\nОтвет:",
    "inputs": {
        "text": "Чему равен косинус угла 0 градусов?",
        "option_a": "-1",
        "option_b": "0",
        "option_c": "1",
        "option_d": "π/2",
        "subject": "Математика"
    },
    "outputs": "C",
    "meta": {
        "id": 6,
        "domain": "high_school_mathematics"
    }
}
```

### Data Splits

The private test set (test split) contains `4250` examples. The few-shot set (train split) `285` hand-written examples.

### Prompts

For this task 10 prompts of varying difficulty were created. Example:

`"Ниже приведен вопрос на определенную профессиональную тематику {subject} и даны варианты ответа A, B, C, D. Гарантируется, что только один из ответов правильный.\nПравильно ответьте на вопрос, выбрав букву A, B, C или D:\n{text}\nA {option_a}\nB {option_b}\nC {option_c}\nD {option_d}\nОтвет:"`.

### Dataset Creation

The test set is based on the [the original MMLU dataset](https://github.com/hendrycks/test) methodology. The set was assembled manually according to the original format with domains as close as possible to the original set. The set is adapted for the Russian language and culture. The distribution of tasks across individual specific domains and subjects are balanced and corresponds to the distribution of the original MMLU.

## Evaluation

### Metrics

The dataset is evaluated using Accuracy and, following the original methodology, is evaluated in the few-shot format with five shots.

### Human benchmark

According to the original article, for English test human-level accuracy varies:
"Unspecialized humans from Amazon Mechanical Turk obtain 34.5% accuracy on English test. Meanwhile, expert-level performance can be far higher. For example, real-world test-taker human accuracy at the 95th percentile is around 87% for US Medical Licensing Examinations, and these questions make up our “Professional Medicine” task. If we take the 95th percentile human test-taker accuracy for exams that build up our test, and if we make an educated guess when such information is unavailable, we then estimate that expert-level accuracy is approximately 89.8%.".

Accuracy of the annotation on the test set is `84.4%`.

## Limitations

The questions relate to human knowledge relevant on October 31, 2023.

## References

[1] Hendrycks, Dan, et al. "Measuring Massive Multitask Language Understanding." International Conference on Learning Representations. 2020.

[2] Wang, Alex, et al. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." *International Conference on Learning Representations*. 2018.

[3] Wang, Alex, et al. "Superglue: A stickier benchmark for general-purpose language understanding systems." *Advances in neural information processing systems* 32 (2019).

[4] The original MMLU translated into Russian (without filtering) https://github.com/NLP-Core-Team/mmlu_ru.

[5] The 🤗 Open LLM Leaderboard (содержит внутри MMLU, замеры происходят по 5-шотам) https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard.
