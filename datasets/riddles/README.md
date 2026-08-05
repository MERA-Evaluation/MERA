# Riddles


## Task Description

**Riddles** is a Russian-language dataset for evaluating a model's ability to solve traditional and modern riddles. It consists of two separate sets: an open set of classic riddles collected from publicly available books and online sources, and a closed set of newly written riddles based on the same text-formation patterns but unavailable online. Each set contains 500 examples. Only the closed set is included in the MERA Text 2.0 leaderboard; the open set is maintained separately and may be used for supplementary diagnostic evaluation.

Riddles encode information about an object or phenomenon through a compact set of distinctive properties, metaphor, personification, metonymy, sound imitation, and other forms of language play. The examples cover subjects ranging from traditional concepts such as the sky, road, night, day, moon, and stars to tools, occupations, and objects found in a modern home. Solving them requires the model to combine linguistic reasoning with attention to every clue in the question.

Tested model skills: General reasoning, Problem solving, Metaphorical Thinking, Folklore, Language Games, Attention to Detail, Pattern recognition

Authors: Denis Shevelev, Artem Chervyakov, Alexander Astafurov, Alexander Kharitonov


## Motivation

The dataset evaluates whether advanced language models can interpret and decode information expressed through riddles, one of the oldest forms of knowledge transmission. In this format, information is compressed into a short and vivid description through metaphor and other linguistic devices.

The results may be useful to AI researchers studying linguistic and creative reasoning and to developers of educational systems for school and preschool learning. Performance indicates whether a model can act as a solver of traditional and modern riddles, which can inform research on language-based reasoning and the development of intelligent assistants and tutoring tools.

### Limitations

The dataset is intended for Russian-language generative text models that can follow instructions and produce a short answer. The result reflects the ability to solve riddles and should not be interpreted as a general measure of reasoning, factual knowledge, or understanding of folklore.

The number of acceptable reference answers varies from 1 to 11 across examples. Examples with more variants therefore have a higher probability of an exact match, which should be considered when interpreting the metric.

### Validity

The task evaluates the following abilities:

1. Broad Russian vocabulary and knowledge of Russian word-formation patterns.
2. Interpretation of locally constructed figurative meanings and conventional metaphors.
3. Reconstruction of an object, phenomenon, or person from a compact set of indirect clues.
4. Logical and compositional reasoning, including recognition of a whole from its parts.
5. Understanding of anthropocentric descriptions and language play.

Each example has a set of acceptable reference answers, including synonyms and, where appropriate, modern equivalents for objects described in archaic riddles. This supports generation-based evaluation and reduces, but does not eliminate, subjectivity and ambiguity in answer matching.


## Dataset Description

The dataset contains two sets of 500 Russian-language riddles each:

- the closed set is used in the MERA diagnostic benchmark;
- the open set is maintained separately and may be used for supplementary diagnostic evaluation.

Each example uses one of five prompt templates and requires a short answer in the form `Ответ: <word or phrase>`.

### Data Fields

Each example contains the following fields:

- `instruction` [str] - a string containing the task formulation for the language model.
- `inputs` - input data forming the task:
    - `question` [str] - the riddle text.
- `outputs` [str] - a string containing one or more acceptable answer words or phrases separated by the sequence `; `. A model answer is correct if it matches at least one variant.
- `meta` - metadata:
    - `id` [int] - example number.

### Data Example

```json
{
    "instruction": "По тексту загадки нужно найти ответ в виде слова или фразы.\n\nВ качестве ответа укажите строку\n\nОтвет: СЛОВО ИЛИ ФРАЗА\n\n{question}",
    "inputs": {
        "question": "Серёжки для простаков."
    },
    "outputs": "Лапша",
    "meta": {
        "id": 504
    }
}
```

### Prompts

The task uses five prompt templates, distributed evenly across the 500 examples in the closed set according to the principle of one prompt per example. Each template consistently uses either informal or formal address. The `{question}` placeholder is filled with the value of the `inputs.question` field.

All prompts ask the model to provide a word or phrase and use an answer line beginning with the Russian marker `Ответ:`.

### Dataset Creation

The dataset was created in two parts of 500 examples each. For the open set, classic riddles were collected from publicly available publications, including well-known collections by Yu. G. Illarionova and M. A. Rybnikova, reprints from older children's magazines, and entertainment websites for children. For the closed set, new riddles were written using the text-formation patterns of classic riddles. These texts are not available online and were supplied with sets of acceptable reference answers.

The collection and validation process included internal expert reviews and data quality analysis. Additional quality checks were performed by evaluating the dataset with advanced language models capable of processing and generating Russian text.


## Evaluation

### Metrics

The following metrics are used for aggregated evaluation:

- **Exact match (EM)**: the proportion of model answers that exactly match one of the reference answers after extracting the text following the `Ответ:` marker. The score ranges from 0 to 1: 0 means no exact matches, while 1 means every example matched.
- **LLM judge score**: an LLM judge compares the model answer with the reference variants and evaluates its correctness and completeness. A fully correct and complete answer receives `1`; a partially correct or incomplete answer containing an essential part of the solution receives `0.5`; an incorrect or contradictory answer, or one that does not contain the correct solution, receives `0`. The final metric value is the mean score across all examples.
