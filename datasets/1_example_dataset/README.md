### ❗️ Некоторые файлы приведены здесь только для образца, в папке реального датасета в репозитории должны делать только такие файлы:

```
├── README.md  # автособираемая документация (англ)
├── README_ru.md  # автособираемая документация (рус)
├── dataset_meta.json  # автособираемая метаинформация
├── raw_readme_en.json  # документация вручную (англ)
├── raw_readme_ru.json  # документация вручную (рус)
├── raw_dataset_meta.json  # метаинформация вручную
```


# XXX


## Task description

Task description.

Evaluated skills: Counterfactual robustness, Static counting

Contributors: Name Surname, Name Surname


## Motivation

This section should comprehensively answer the question “Why is this task important?”. Guiding questions (refer to [ECBD](https://aclanthology.org/2024.acl-long.861/) for details on benchmark design motivation):
- What models is this dataset oriented towards evaluating? For which models is it NOT suitable (limitations)?
- What users are the evaluation results targeted at? How will these users interpret the results obtained from evaluating on this dataset?
- What model capabilities does the task assess? What exactly is meant by these capabilities? Not just language understanding, but specifically what and in what context? Why evaluate these particular capabilities?
- How are the questions in the dataset structured, and why are they structured this way? How can one understand that this specific task design evaluates the model capabilities that need to be assessed? Experimental validity.
- How are the metrics chosen (especially when the metric is not simply the proportion of correct answers), and why are they the way they are? How does this choice of metrics help the user interpret the results?


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — Instruction prompt template with question elements placeholders.
- `inputs` — Input data that forms the task for the model. Can include one or multiple modalities - video, audio, image, text.
    - `question` [str] — Text of the question.
    - `image` [str] — Path to the image file related to the question.
    - `option_a` [str] — Answer option A.
    - `option_b` [str] — Answer option B.
    - `option_c` [str] — Answer option C.
    - `option_d` [str] — Answer option D.
- `outputs` [str] — The correct answer to the question.
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — Identification number of the question in the dataset.
    - `categories` — Categorial features characterizing the test example.
        - `task_type` [str] — Task type according to the task classification in the XXX dataset;
    - `image` — Image metadata.
        - `synt_source` [list] — Sources used to generate or recreate data for the question, including names of generative models.
        - `source` [list] — Information about the origin of the image — according to the image classification for MERA datasets.
        - `type` [list] — Image type — according to the image classification for MERA datasets.
        - `content` [list] — Image content — according to the image classification for MERA datasets.
        - `context` [list] — Accompanying context present in the image — according to the image classification for MERA datasets.


### Data formatting example

```json
{
    "instruction": "Посмотри на картинку <image> и ответь на вопрос, выбрав вариант ответа из предложенных. Напиши только букву правильного ответа.\nВопрос: {question}.\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d)\nОтвет:",
    "inputs": {
        "question": "Сколько автомобилей изображено на фото?",
        "image": "image0001.jpg",
        "option_a": "три",
        "option_b": "два",
        "option_c": "ни одного",
        "option_d": "пять"
    },
    "outputs": "C",
    "meta": {
        "id": 1,
        "categories": {
            "task_type": "counterfactual"
        },
        "image": {
            "synt_source": [
                "model-name"
            ],
            "source": [
                "photo"
            ],
            "type": [
                "visual"
            ],
            "content": [
                "view",
                "objects"
            ],
            "context": [
                "no_context"
            ]
        }
    }
}
```


### Prompts

For the task, 10 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
prompt_0
```


### Dataset creation

Dataset creation methodology.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Accuracy`: Accuracy is the proportion of correct model predictions among the total number of cases processed.


### Human baseline

To compare the quality of the model responses and human responses, measurements of human responses were taken using the following methodology.

Human baseline evaluation methodology.

Evaluation results:

- Accuracy – 1.00
