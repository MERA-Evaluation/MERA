# SAGE


## Task description

The SAGE dataset is designed to evaluate the ability of language models to automatically correct errors in Russian text. The task is formulated as a text-to-text transformation problem: converting erroneous text into its corrected version. Models are expected to correct spelling and punctuation errors without adding comments or explanations. This dataset is an improved version of the publicly available SAGE dataset (v1.1.0), published on [Hugging Face](https://huggingface.co/ai-forever/sage-v1.1.0). The dataset is intended for text-to-text models and enables quantitative evaluation of automatic text correction quality.

Some examples in the dataset may already be correct and require no modifications. This is an expected and intentional property of the benchmark: models should be able not only to correct errors, but also to leave correct text unchanged.

Evaluated skills: Spelling correction, Punctuation correction, Error detection, Text editing.

Contributors: Danil Astafurov, Ulyana Isaeva, Alena Fenogenova, Anastasia Mordasheva, Natalia Atnagulova, Olga Kun, Olga Tabolina, Kristina Eremeeva, Nikita Martynov, Alexander Astafurov


## Motivation

### Target models
The dataset is intended for generative language models capable of text editing and automatic error correction.

### Limitations
The benchmark is not intended to evaluate:
- open-ended text generation quality
- reasoning capabilities or factual knowledge
- stylistic rewriting or paraphrasing
- semantic text improvement
- multilingual text correction

### Evaluated capabilities
The task evaluates a model’s ability to:
- detect different types of errors
- correct them accurately
- preserve the original meaning of the text

This goes beyond basic language understanding and focuses on practical text editing ability.

### Intended users
The results are intended for NLP researchers and practitioners evaluating text generation and post-processing quality.

### Interpretation of metrics
The metrics reflect the quality of correcting spelling and punctuation errors, allowing evaluation of model strengths and weaknesses.

### Validity
The text-to-text task formulation combined with edit-level metrics enables separate evaluation of a model’s ability to detect errors (recall) and correct them accurately (precision), which would be difficult in classification-based or span-based settings. This design also reflects real-world usage scenarios in which LLMs act as text editors.
The benchmark design controls key evaluation factors: stratified sampling based on input text length is applied, and spelling and punctuation errors are evaluated separately.
At the same time, the benchmark has validity limitations. Some cases of Russian punctuation may allow multiple acceptable interpretations in the gold annotations. Mandatory restoration of the letter \"ё\" is an intentional design choice and may affect evaluation results for models trained on alternative writing conventions. In addition, the aggregated metric `ERRANT_F1` does not capture relationships between different error types and should be interpreted together with task-specific metrics.",


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — Instruction prompt template with question elements placeholders.
- `inputs` — Input text data that forms the task for the model.
    - `source` [str] — Text containing errors that must be corrected.
- `outputs` [str] — The correct answer to the question.
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — Identification number of the question in the dataset.


### Data formatting example

```json
{
    "instruction":"Задача:\nИсправь ошибки в тексте, сохранив смысл и стиль автора.\n\nИсправляй следующие ошибки:\n- орфографические\n- пунктуационные\n- синтаксические\n- морфологические\n- лексические\n\nНе исправляй:\n- стиль и формулировки\n- сленг, разговорную речь, диалектизмы\n- эмоции и намеренные искажения\n- логические или фактические ошибки\n- текст не на русском языке\n\nОграничения:\n- не добавляй и не удаляй слова без необходимости\n- не делай первую букву заглавной без причины\n- не добавляй точку в конце\n- всегда используй \"ё\"\n- сохраняй форматирование\n\nФормат ответа:\nНапиши только исправленный текст.\n\nТекст:\n{source}\n",
    "inputs": {
        "source": "Я заказала размер 35, себя замерив, почитая отзывы. На рос. размер 50-52 заказала, но не посоветовавшись со мной продавец отправил 36 размер. Ну, ооочень большой, одела с застегнутым замком, а пуговицы вообще не застегнула, так как петли не прорезаны. Я выразила претензию. Продавец долго морочил мне голову, что мой заказ мне должен подойти, не спросив даже мой размер. Короче, предложил взамен 3 доллара, я возмутилась и открыла спор. Спасибо AliExpress, вернули деньги полностью. Джинсы валяются, запах ужасный, хорошо, хоть деньги вернули"
    },
    "outputs": "Я заказала размер 35, себя замерив, прочитав отзывы. На рос. размер 50–52 заказала, но, не посоветовавшись со мной, продавец отправил 36 размер. Ну ооочень большой, надела с застёгнутым замком, а пуговицы вообще не застегнула, так как петли не прорезаны. Я выразила претензию. Продавец долго морочил мне голову, что мой заказ мне должен подойти, не спросив даже мой размер. Короче, предложил взамен 3 доллара, я возмутилась и открыла спор. Спасибо AliExpress, вернули деньги полностью. Джинсы валяются, запах ужасный, хорошо хоть деньги вернули",
    "meta": {
        "id": 1
    }
}
```

### Prompts

For the task, 5 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
Задача:
Исправь ошибки в тексте, сохранив смысл и стиль автора.

Исправляй следующие ошибки:
- орфографические
- пунктуационные
- синтаксические
- морфологические
- лексические

Не исправляй:
- стиль и формулировки
- сленг, разговорную речь, диалектизмы
- эмоции и намеренные искажения
- логические или фактические ошибки
- текст не на русском языке

Ограничения:
- не добавляй и не удаляй слова без необходимости
- не делай первую букву заглавной без причины
- не добавляй точку в конце
- всегда используй \"ё\"
- сохраняй форматирование

Формат ответа:
Напиши только исправленный текст.

Текст:
{source}
```


### Dataset creation

### Data Sources

The SAGE dataset is based on the original SAGE dataset [[1](https://aclanthology.org/2024.findings-eacl.10/),[2](https://dialogue-conf.org/media/5914/martynovnplusetal056.pdf)] and was constructed using a variety of Russian-language textual sources, including open corpora, web texts, and user-generated queries. To ensure diversity in linguistic structures, data from multiple domains were incorporated, including news articles, literary texts, social media content, medical texts, and technical documents.

The dataset was built using existing Russian-language corpora and benchmark datasets, including RUSpellRU, MultidomainGold, MedSpellChecker, and GitHubTypoCorpusRu. In addition, texts from publicly available online sources and user-generated content were included to expand coverage of spelling and punctuation phenomena.

### Annotation and Validation

For each example, both the original text and the corresponding reference correction were recorded. The dataset underwent manual verification to ensure correction accuracy and compliance with modern Russian language standards. Ambiguous cases allowing multiple valid corrections were excluded from the final dataset.

### Data Cleaning

Empty, duplicate, and invalid records were removed from the dataset, along with examples containing ambiguous corrections. Additional checks were performed to ensure diversity in text length, domain coverage, and error categories.

### Error Types
The dataset includes two types of errors: spelling and punctuation errors. Multiple error types may appear within a single text.

### Data Splitting
After preprocessing, the data was shuffled and split into final evaluation sets.
The initial pool consisted of 2,500 sample, of which 1,000 examples were selected for the final evaluation set (40% of the data). To preserve diversity in text length, stratified sampling based on input text length was applied.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `SPELL_F1`: F1 score measuring the quality of spelling error correction.  
- `PUNCT_F1`: F1 score measuring the quality of punctuation error correction.  
- `ERRANT_F1`: an aggregated metric defined as the average of SPELL_F1 and PUNCT_F1, reflecting the overall quality of text correction.