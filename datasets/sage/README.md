# SAGE


## Task description

The SAGE dataset is designed to evaluate the ability of language models to automatically correct errors in Russian text.

The task is formulated as transforming a text containing errors into a grammatically correct and natural version. Models are expected to correct spelling, punctuation, grammar, and letter case errors without adding any comments or explanations.

This dataset is an enhanced version of the open SAGE dataset (v1.1.0), published on Hugging Face: https://huggingface.co/ai-forever/sage-v1.1.0.

The benchmark is oriented toward text-to-text models and enables quantitative evaluation of automatic text correction quality.

Evaluated skills: Grammatical error correction, Spelling correction, Punctuation correction, Casing correction, Error detection, Text editing


## Motivation

### Target models
The dataset is intended for generative language models capable of editing text and performing automatic error correction.

### Evaluated capabilities
The benchmark evaluates a model’s ability to:
- detect different types of errors
- correct them accurately
- preserve the original meaning of the text

This task goes beyond basic language understanding and focuses on language editing capabilities.

### Target audience
The results are intended for NLP researchers and developers evaluating text generation and post-editing performance.

### Interpretation of metrics
The metrics reflect correction quality across different error categories (letter case, spelling, punctuation, etc.), enabling analysis of model strengths and weaknesses.


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — Instruction prompt template with question elements placeholders.
- `inputs` — Input data that forms the task for the model. Can include one or multiple modalities - video, audio, image, text.
    - `source` [str] — Text containing errors that must be corrected.
- `outputs` [str] — The correct answer to the question.
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — Identification number of the question in the dataset.


### Data formatting example

```json
{
    "instruction": "ЗАДАЧА\nИсправь орфографические, пунктуационные и грамматические ошибки в тексте, а также ошибки регистра букв.\n\nФОРМАТ ОТВЕТА\nВыведи только исправленный текст без пояснений и комментариев.\n\nТЕКСТ\n{source}",
    "inputs": {
        "source": "Очень жалею что сделала заказ в этом магазине. Вещи абсолютно не соответствуют описанию в профиле продавца. Деньги можно просто выкинуть на помойку, а не ждать, когда придёт посылка, ибо при примерке и после шока, вы все равно эту вещь выкинете в мусорку. Сшито криво-косо-размер не соответствует. Вещи из этого магазина можно, не сомневаясь, отправлять в обзор 'Ожидание и реальность от Aliexpress - смех и разочарование'. Короткая и квадратная, как будто сшита вместо наволочки на подушку. Не рекомендую."
    },
    "outputs": "Очень жалею, что сделала заказ в этом магазине. Вещи абсолютно не соответствуют описанию в профиле продавца. Деньги можно просто выкинуть на помойку, а не ждать, когда придёт посылка, ибо при примерке и после шока вы всё равно эту вещь выкинете в мусорку. Сшито криво-косо, размер не соответствует. Вещи из этого магазина можно не сомневаясь отправлять в обзор «Ожидание и реальность от Aliexpress — смех и разочарование». Короткая и квадратная, как будто сшита вместо наволочки на подушку. Не рекомендую.",
    "meta": {
        "id": 1
    }
}
```


### Prompts

For the task, 5 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
ЗАДАЧА
Исправь орфографические, пунктуационные и грамматические ошибки в тексте, а также ошибки регистра букв.

ФОРМАТ ОТВЕТА
Выведи только исправленный текст без пояснений и комментариев.

ТЕКСТ
{source}
```


### Dataset creation

### Data sources
The data were collected from various Russian-language text sources, including open corpora, internet texts, user-generated content, and synthetically generated examples.

### Annotation
A portion of the dataset was manually re-annotated to ensure the correctness of reference corrections.

### Data cleaning
Empty, duplicate, and invalid samples were removed from the dataset.

### Error categories
The dataset includes the following error categories (percentage of samples containing at least one error of the given type):

- CASE — letter case errors (uppercase/lowercase mismatch): 13.6%
- YO — incorrect usage of the letters 'е' and 'ё': 33.2%
- SPELL — spelling errors: 58.0%
- PUNCT — punctuation errors: 43.2%

The percentages reflect the proportion of samples containing at least one error of the corresponding type. A single sample may contain multiple error types simultaneously.

### Dataset splitting
After preprocessing, the data were randomly shuffled and divided into the final evaluation splits.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `CASE_F1`: CASE_F1 measures the F1-score for edits related to letter case correction (uppercase and lowercase). The metric is computed based on edit-level precision and recall within this category.
- `YO_F1`: YO_F1 measures the F1-score for edits related to correcting the use of the letters 'е' and 'ё'. The metric is computed based on edit-level precision and recall within this category.
- `SPELL_F1`: SPELL_F1 measures the F1-score for spelling-related edits. The metric is computed based on edit-level precision and recall within this category.
- `PUNCT_F1`: PUNCT_F1 measures the F1-score for punctuation-related edits. The metric is computed based on edit-level precision and recall within this category.
- `errant`: errant is a weighted aggregated F1-score across error categories (CASE, YO, SPELL, PUNCT). Each category-specific F1-score is weighted proportionally to its distribution in the dataset and normalized by the sum of weights. The metric reflects the overall quality of text correction while accounting for the relative frequency of different error types.
