# The SAGE Dataset

To run the SAGE evaluation, additional dependencies are required.

Install the required packages:

```bash
python -m spacy download ru_core_news_lg
pip install sage-spelling[errant]
```

## Description

SAGE is a Russian text error correction benchmark designed to evaluate language models' ability to automatically edit and correct natural language text.

The task is formulated as transforming an input text containing errors into a grammatically correct and natural version without adding any comments or explanations. Models are expected to correct spelling errors, punctuation mistakes, letter case inconsistencies, and incorrect usage of the letters “е/ё”.

Correction quality is evaluated using edit-level F1 scores for different error categories (CASE, YO, SPELL, PUNCT), as well as a weighted aggregated metric reflecting their distribution in the dataset.

## Homepage

https://mera.a-ai.ru

## License

MERA Private License
