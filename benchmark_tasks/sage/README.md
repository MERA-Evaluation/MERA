# The SAGE Dataset

## Description

[SAGE](https://huggingface.co/ai-forever/sage-v1.1.0) is a Russian text error correction benchmark designed to evaluate language models' ability to automatically edit and correct natural language text.

The task is formulated as transforming an input text containing errors into a grammatically correct and natural version without adding any comments or explanations. Models are expected to correct spelling and punctuation errors.

Correction quality is evaluated using multiple metrics. SPELL_F1 and PUNCT_F1 measure the quality of spelling and punctuation error correction, respectively. ERRANT_F1 is defined as the average of these metrics and reflects the overall quality of text correction.

## Homepage

https://mera.a-ai.ru

## License

MERA Private License
