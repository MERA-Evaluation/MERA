# The T-Math Dataset

## Description

T-Math is a task based on the original [dataset](https://huggingface.co/datasets/t-tech/T-math). It consists of Russian-language olympiad mathematics problems for evaluating the multi-step reasoning abilities of language models.


The task is formulated as generating a solution with a final answer. The model is expected to reason step by step and produce the final answer, which is then extracted from the generation and compared against the gold one.

Quality is measured with the `exact_match` metric computed by the [`math_verify`](https://github.com/huggingface/Math-Verify) library following the methodology of the original T-math: the final answer is extracted from the generation (prioritizing `\boxed{...}`, with a fallback to plain expressions), converted to a sympy expression, and compared against the gold answer for symbolic equivalence (`1/2` == `0.5` == `\frac{1}{2}`). Under greedy decoding the metric is equivalent to pass@1. Computing the metric requires an extra dependency:

```
pip install math_verify
```

## Homepage

https://mera.a-ai.ru

## License

apache-2.0
