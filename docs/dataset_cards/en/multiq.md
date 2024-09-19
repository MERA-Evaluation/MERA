# **MultiQ**

## Task Description

MultiQ is a multi-hop QA dataset for Russian, suitable for general open-domain question answering, information retrieval, and reading comprehension tasks. The dataset is based on the [dataset](https://tape-benchmark.com/datasets.html#multiq) of the same name from the TAPE benchmark [1].

**Keywords:** Multi-hop QA, World Knowledge, Logic, Question-Answering

**Authors:** Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

### Motivation

Question-answering has been an essential task in natural language processing and information retrieval. However, certain areas in QA remain quite challenging for modern approaches, including the multi-hop one, which is traditionally considered an intersection of graph methods, knowledge representation, and SOTA language modeling.

## Dataset Description

### Data Fields

- `meta` is a dictionary containing meta-information about the example:
    - `id` is the task ID;
    - `bridge_answer` is a list of entities necessary to answer the question contained in the `outputs` field using two available texts;
- `instruction` is an instructional prompt specified for the current task;
- `inputs` is a dictionary containing the following information:
    - `text` is the main text line;
    - `support_text` is a line with additional text;
    - `question` is the question, the answer to which is contained in these texts;
- `outputs` is a string containing the answer.

### Data Instances

Each dataset sample consists of two texts (the main and the supporting ones) and a question based on these two texts. Below is an example from the dataset:

```json
{
    "instruction": "Даны два текста:\nТекст 1: {support_text}\nТекст 2: {text}\nОпираясь на данные тексты, ответьте на вопрос: {question}\nВаш ответ не должен содержать дополнительные объяснения.\nОтвет:",
    "inputs": {
        "text": "Нижний Новгород (в разговорной речи часто — \"Нижний\", c XIII по XVII век — Новгород Низовской земли, с 7 октября 1932 по 22 октября 1990 года — Горький) — город в центральной России, административный центр Приволжского федерального округа и Нижегородской области. Второй по численности населения город в Приволжском федеральном округе и на реке Волге.\\n\\nКультура.\\nИсторический центр Нижнего Новгорода, расположенный в Нагорной части города, несмотря на значительные перестройки, сохранил значительное число исторических гражданских строений XVIII — начала XX веков, включая многочисленные памятники деревянного зодчества. Дмитриевская башня Кремля выходит на историческую площадь Минина и Пожарского. Нижегородский кремль является официальной резиденцией Городской думы Нижнего Новгорода и правительства Нижегородской области. Зоопарк \"Лимпопо\". Зоопарк \"Лимпопо\" — первый частный зоопарк в России, расположенный в Московском районе.",
        "support_text": "Евгений Владимирович Крестьянинов (род. 12 июля 1948, Горький) — российский государственный деятель.",
        "question": "Как называется законодательный орган города, где родился Евгений Владимирович Крестьянинов?"
    },
    "outputs": "Городской думы",
    "meta": {
        "id": 0,
        "bridge_answers": "Горький"
    }
}
```

### Data Splits

The dataset consists of `1056` training examples (train set) and `900` test examples (test set).

### Prompts

We prepared 10 different prompts of various difficulties for this task.
An example of the prompt is given below:

```json
"Текст 1: {support_text}\nТекст 2: {text}\nОпираясь на данные тексты, ответьте на вопрос: {question}\nЗапишите только ответ без дополнительных объяснений.\nОтвет:"
```

### Dataset Creation

The dataset was created using the corresponding dataset from the TAPE benchmark [1] and was initially sampled from Wikipedia and Wikidata. The whole pipeline of the data collection can be found [here](https://tape-benchmark.com/datasets.html#multiq).

## Evaluation

### Metrics

To evaluate models on this dataset, two metrics are used: F1-score and complete match (Exact Match — EM).

### Human Benchmark

The F1-score / EM results are `0.928` / `0.91`, respectively.

## References

[1] Taktasheva, Ekaterina, et al. "TAPE: Assessing Few-shot Russian Language Understanding." Findings of the Association for Computational Linguistics: EMNLP 2022. 2022.
