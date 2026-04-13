# The SAGE Dataset

## Description

[SAGE](https://huggingface.co/ai-forever/sage-v1.1.0) is a Russian text error correction benchmark designed to evaluate language models' ability to automatically edit and correct natural language text.

The task is formulated as transforming an input text containing errors into a grammatically correct and natural version without adding any comments or explanations. Models are expected to correct spelling errors, punctuation mistakes, letter case inconsistencies, and incorrect usage of the letters “е/ё”.

Correction quality is evaluated using the ERRANT_F1 metric, which measures the overall F1-score for spelling and punctuation corrections at the edit level.

## Homepage

https://mera.a-ai.ru

## License

MERA Private License
