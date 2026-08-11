"""Task utils for the Humor dataset.

Two metrics, ``exact_match`` and ``judge_score``, scored the same way as the
other generative tasks: the filter chain only normalises the response, and the
metric reads it several ways and keeps the best score.

Humor is the one task with a genuinely different answer shape. The model is
asked two questions at once and answers them on one line as
``<class>,<letter>`` — the kind of humour, then the option — announced with the
``ОТВЕТ`` and ``РЕШЕНИЕ`` markers rather than the ``Ответ:`` the rest of the
suite uses. So on top of the two generic readings this task adds a third
candidate: the pair reassembled by :func:`parse_humor_response`. The generic
readings still count, because a model that simply wrote ``ирония,Г`` with no
markers at all has given a perfectly good answer.
"""

import logging
import os
import re
from typing import List

from lm_eval.api.answer_extraction import answer_candidates, best_score
from transformers.data.metrics import squad_metrics

eval_logger = logging.getLogger(__name__)


def humor_candidates(response):
    """The generic readings of the response plus the reassembled pair."""
    candidates = answer_candidates(response)
    parsed = parse_humor_response(response)
    if parsed and parsed not in candidates:
        candidates.append(parsed)
    return candidates


def process_results(doc, results):
    gold = doc.get("outputs") or ""
    if not gold:
        return {"exact_match": 0, "judge_score": 0.0}

    candidates = humor_candidates(results[0] if results else "")

    return {
        "exact_match": best_score(
            lambda c: squad_metrics.compute_exact(gold, c), candidates),
        "judge_score": best_score(
            lambda c: compute_judge_score(doc, c), candidates),
    }


def compute_judge_score(doc, model_answer):
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    judge_prompt = os.getenv(
        "LM_EVAL_HUMOR_JUDGE_PROMPT",
        "Оцени, верно ли модель определила тип юмора и вариант ответа, по шкале от 1 до 10.\n"
        "Эталон: {reference}\nОтвет модели: {prediction}\nОдно число:",
    )

    if not doc.get("outputs") or not judge_api_base or not judge_model:
        return 0.0

    try:
        from lm_eval.api.metrics_generative import compute_llm_judge

        return compute_llm_judge(
            [model_answer],
            [doc["outputs"]],
            api_base=judge_api_base,
            model=judge_model,
            judge_prompt=judge_prompt,
        )["llm_judge"]
    except Exception:
        eval_logger.warning("%s: llm judge failed, scoring 0.0", __name__, exc_info=True)
        return 0.0


def doc_to_text(doc):
    return doc["instruction"].format(**doc["inputs"])


# --- Humor-specific answer parser (kept per the final review) -----------------

_ANSWER_RE = re.compile(r"ОТВЕТ\s*[:\-—]?\s*(.+)", re.IGNORECASE)
_SOLUTION_RE = re.compile(r"РЕШЕНИЕ\s*[:\-—]?\s*([АБВГ])", re.IGNORECASE)
# A response that is already in the final "<class>,<letter>" shape.
_FINAL_RE = re.compile(r"^\s*(.+?)\s*,\s*([АБВГ])\s*\.?\s*$", re.DOTALL)


def parse_humor_response(resp: str) -> str:
    """Extract ``<class>,<letter>`` from a model response.

    Strategy:
    1. Look for the ОТВЕТ and РЕШЕНИЕ markers anywhere in the text
       (last occurrence wins, so chain-of-thought repeats are tolerated).
    2. If markers are absent but the response already looks like the final
       ``<class>,<letter>`` string, return it normalized as is
       (fixes the ``ирония,Г`` -> ``ирония,Г,`` bug).
    3. Otherwise return the stripped response unchanged; exact_match will
       simply fail, but no artificial commas are appended.
    """
    if not resp or not str(resp).strip():
        return ""

    text = str(resp)

    answer_matches = []
    for line in text.split("\n"):
        m = _ANSWER_RE.search(line)
        if m:
            candidate = m.group(1).strip(" \t:—-")
            # Cut a trailing РЕШЕНИЕ part if both markers share one line.
            candidate = re.split(r"РЕШЕНИЕ", candidate, flags=re.IGNORECASE)[0]
            candidate = candidate.strip(" \t:—-,.")
            if candidate:
                answer_matches.append(candidate)

    solution_matches = [m.group(1).upper() for m in _SOLUTION_RE.finditer(text)]

    if answer_matches or solution_matches:
        answer = answer_matches[-1] if answer_matches else ""
        solution = solution_matches[-1] if solution_matches else ""
        return f"{answer},{solution}"

    final = _FINAL_RE.match(text.strip())
    if final:
        return f"{final.group(1).strip()},{final.group(2).upper()}"

    return text.strip()
