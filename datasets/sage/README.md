# SAGE


## Task description

The SAGE dataset is designed to evaluate the ability of language models to automatically correct errors in Russian text. The task is formulated as a text-to-text transformation problem: converting erroneous text into its corrected version. Models are expected to correct spelling and punctuation errors without adding comments or explanations. This dataset is an improved version of the publicly available SAGE dataset (v1.1.0), published on Hugging Face: https://huggingface.co/ai-forever/sage-v1.1.0. The dataset is intended for text-to-text models and enables quantitative evaluation of automatic text correction quality.

Evaluated skills: Spelling correction, Punctuation correction, Error detection, Text editing

Contributors: Danil Astafurov, Ulyana Isaeva, Alena Fenogenova, Anastasia Mordasheva, Natalia Atnagulova, Olga Kun, Olga Tabolina, Kristina Eremeeva, Nikita Martynov, Alexander Astafurov


## Motivation

### Target models
The dataset is intended for generative language models capable of text editing and automatic error correction.

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
    "instruction": "<role>\nТы — идеальный спеллчекер.\n</role>\n\n<task>\nИсправь ошибки в тексте, сохранив смысл и стиль автора.\n</task>\n\n<input_text>\n{source}\n</input_text>\n\n<rules>\nИсправляй:\n- орфографические\n- пунктуационные\n- синтаксические\n- морфологические\n- лексические ошибки\n\nНе исправляй:\n- стиль и формулировки\n- сленг, разговорную речь, диалектизмы\n- эмоции и намеренные искажения\n- логические или фактические ошибки\n- текст не на русском языке\n</rules>\n\n<constraints>\n- не добавляй и не удаляй слова без необходимости\n- не делай первую букву заглавной без причины\n- не добавляй точку в конце\n- всегда используй \"ё\"\n- сохраняй форматирование\n</constraints>\n\n<output_format>\nВерни только исправленный текст.\n</output_format>",
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
<role>
Ты — идеальный спеллчекер.
</role>

<task>
Исправь ошибки в тексте, сохранив смысл и стиль автора.
</task>

<input_text>
{source}
</input_text>

<rules>
Исправляй:
- орфографические
- пунктуационные
- синтаксические
- морфологические
- лексические ошибки

Не исправляй:
- стиль и формулировки
- сленг, разговорную речь, диалектизмы
- эмоции и намеренные искажения
- логические или фактические ошибки
- текст не на русском языке
</rules>

<constraints>
- не добавляй и не удаляй слова без необходимости
- не делай первую букву заглавной без причины
- не добавляй точку в конце
- всегда используй "ё"
- сохраняй форматирование
</constraints>

<output_format>
Верни только исправленный текст.
</output_format>
```


### Dataset creation

### Data Sources
The data is collected from various Russian-language text sources, including open corpora, web texts and user queries.

### Annotation
The dataset was manually re-annotated to ensure the correctness of reference corrections.

### Data Cleaning
Empty, duplicate, and invalid samples were removed from the dataset.

### Error Types
The dataset includes two types of errors: spelling and punctuation errors. Multiple error types may appear within a single text.

### Data Splitting
After preprocessing, the data was shuffled and split into final evaluation sets.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `SPELL_F1`: F1 score measuring the quality of spelling error correction.  
- `PUNCT_F1`: F1 score measuring the quality of punctuation error correction.  
- `ERRANT_F1:` an aggregated metric defined as the average of spell_f1 and punct_f1, reflecting the overall quality of text correction.