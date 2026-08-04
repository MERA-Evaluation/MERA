"""Task utils for the Humor dataset.

Two metrics: EM and
LLM judge. The scoring code follows benchmark_tasks/rwsd_2/utils.py (PR 36,
approved reference): ``process_results`` returns ``exact_match`` +
``judge_score``, the judge is configured through the generic
``LM_EVAL_JUDGE_API_BASE`` / ``LM_EVAL_JUDGE_MODEL`` /
``LM_EVAL_JUDGE_PROMPT_PATH`` env vars and receives the rendered task text
as ``instruction``.


"""

import os
import re
from typing import List

from transformers.data.metrics import squad_metrics
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, FILTER_REGISTRY


def process_results(doc, results):
    # The Humor answer parser runs in the filter, so results[0] is already
    # the canonical "<class>,<letter>" string (rwsd_2 calls extract_answer
    # here instead).
    model_answer = results[0] if results and results[0] else ""
    exact_score = squad_metrics.compute_exact(doc["outputs"], model_answer) if doc["outputs"] else 0
    judge_score = compute_judge_score(doc, model_answer)

    return {
        "exact_match": exact_score,
        "judge_score": judge_score,
    }


def compute_judge_score(doc, model_answer):
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    judge_prompt_path = os.getenv("LM_EVAL_JUDGE_PROMPT_PATH")

    if not doc.get("outputs") or not judge_api_base or not judge_model:
        return 0.0

    try:
        from lm_eval.api.metrics_generative import compute_llm_judge

        return compute_llm_judge(
            [model_answer],
            [doc["outputs"]],
            api_base=judge_api_base,
            model=judge_model,
            judge_prompt_path=judge_prompt_path,
            instruction=doc_to_text(doc),
        )["llm_judge"]
    except Exception:
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


if "remove_whitespace_and_nones" not in FILTER_REGISTRY:

    @register_filter("remove_whitespace_and_nones")
    class RemoveWhitespaceAndNones(Filter):
        def apply(self, resps: List[List[str]], docs: List[dict]) -> List[List[str]]:
            def filter_set(inst):
                return [parse_humor_response(resp) for resp in inst]

            return [filter_set(resp) for resp in resps]
