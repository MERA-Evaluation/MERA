import datasets
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

# Как в run_vllm_prompts.py: обёртка вокруг поля prompt из датасета
PROMPT_TEMPLATE = """Один из вариантов графического диктанта — рисование по клеточкам в заданном порядке. При этом линии проводят строго по инструкции. В процессе этой несложной игры на листе бумаги в клетку выполняют линии, и в результате получается картинка. Это может быть домик, кораблик или какое-либо животное. Какой рисунок получится в результате этого графического диканта?
{prompt}
В конце своего ответа напиши "Ответ: <правильный ответ>".
"""

_ANSWER_MARKER = "Ответ:"


def _text_after_last_marker(s: str, marker: str = _ANSWER_MARKER) -> str:
    s = s if s is not None else ""
    idx = s.rfind(marker)
    if idx == -1:
        return s.strip()
    return s[idx + len(marker) :].strip()


@register_filter("cellpicturebench_extract")
class CellPictureBenchExtract(Filter):
    """Всё, что идёт после *последнего* вхождения «Ответ:» в сгенерированном тексте."""

    def apply(self, resps, docs):
        return [
            [_text_after_last_marker(completion) for completion in sample] for sample in resps
        ]


def _normalize_row(doc: dict) -> dict:
    row = dict(doc)
    if "Prompt" in row and "prompt" not in row:
        row["prompt"] = row["Prompt"]
    if "Answer" in row and "answer" not in row:
        row["answer"] = row["Answer"]
    return row


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_normalize_row)


def doc_to_text(doc: dict) -> str:
    p = doc.get("prompt", "")
    if p is None:
        p = ""
    return PROMPT_TEMPLATE.format(prompt=str(p).strip())
