import re
import sys
from pathlib import Path
from typing import Any

from lm_eval.api.answer_extraction import answer_candidates, best_score

# lm-eval executes this file by path, so benchmark_tasks is not importable by
# name from here (see lm_eval.utils.import_function); the judge, shared with the
# other tasks that report judge_score, is picked up from the directory above.
_BENCHMARK_TASKS = str(Path(__file__).resolve().parent.parent)
if _BENCHMARK_TASKS not in sys.path:
    sys.path.insert(0, _BENCHMARK_TASKS)

from mera_judge import compute_judge_score  # noqa: E402


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["instruction"].format(**doc["inputs"])


def normalize(text: str) -> str:
    return text.strip().lower()


def process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, int | float]:
    gold = doc.get("outputs") or ""
    if not gold:
        return {"exact_match": 0, "judge_score": 0.0}

    # A riddle has several accepted answers, separated by ";".
    gold_variants = [x.strip() for x in re.split(r";", gold) if x.strip()]
    normalized_gold = {normalize(x) for x in gold_variants}

    # Both readings of the response are scored and the better one kept: the
    # model may answer without the marker, or restate it while thinking out
    # loud. See lm_eval.api.answer_extraction.
    candidates = answer_candidates(results[0] if results else "")

    exact_match = int(
        best_score(lambda c: float(normalize(c) in normalized_gold), candidates))

    return {
        "exact_match": exact_match,
        "judge_score": best_judge_score(doc, candidates, gold_variants),
    }


def best_judge_score(
    doc: dict[str, Any], candidates: list[str], gold_variants: list[str]
) -> float:
    """The judge's best score over every reading of the answer against every
    accepted answer.

    The judge sees one accepted answer at a time. Handing it the whole
    ``"нож; ножницы"`` string as the reference invites it to expect every
    reading listed back, when naming one of them solves the riddle.

    That is a call to the judge per pair, so scoring stops at the first perfect
    one — nothing later can beat it.
    """
    best = 0.0
    for candidate in candidates:
        for variant in gold_variants:
            best = max(
                best, compute_judge_score(doc, candidate, reference=variant))
            if best == 1.0:
                return best
    return best
