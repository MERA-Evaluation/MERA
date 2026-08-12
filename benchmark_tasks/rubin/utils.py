import sys
from pathlib import Path
from typing import Dict, List, Any
from lm_eval.api.answer_extraction import answer_candidates, best_score
from transformers.data.metrics import squad_metrics

# lm-eval executes this file by path, so benchmark_tasks is not importable by
# name from here (see lm_eval.utils.import_function); the judge, shared with the
# other tasks that report judge_score, is picked up from the directory above.
_BENCHMARK_TASKS = str(Path(__file__).resolve().parent.parent)
if _BENCHMARK_TASKS not in sys.path:
    sys.path.insert(0, _BENCHMARK_TASKS)

from mera_judge import compute_judge_score  # noqa: E402


def doc_to_text(doc: Dict[str, Any]) -> str:

    return doc["instruction"].format(**doc["inputs"])


def process_results(doc, results):
    # Both readings of the response are scored and the better one kept: the
    # model may answer without the marker, or restate it while thinking out
    # loud. See lm_eval.api.answer_extraction.
    candidates = answer_candidates(results[0] if results else "")
    gold = doc.get("outputs") or ""
    if not gold:
        return {"exact_match": 0, "judge_score": 0.0}

    exact_score = best_score(
        lambda c: squad_metrics.compute_exact(gold, c), candidates)
    judge_score = best_score(
        lambda c: compute_judge_score(doc, c), candidates)

    return {
        "exact_match": exact_score,
        "judge_score": judge_score,
    }
