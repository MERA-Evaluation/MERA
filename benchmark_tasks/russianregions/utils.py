import sys
from pathlib import Path

from lm_eval.api.answer_extraction import answer_candidates, best_score
from transformers.data.metrics import squad_metrics

# lm-eval executes this file by path, so benchmark_tasks is not importable by
# name from here (see lm_eval.utils.import_function); the judge, shared with the
# other tasks that report judge_score, is picked up from the directory above.
_BENCHMARK_TASKS = str(Path(__file__).resolve().parent.parent)
if _BENCHMARK_TASKS not in sys.path:
    sys.path.insert(0, _BENCHMARK_TASKS)

from mera_judge import compute_judge_score  # noqa: E402


def process_results(doc, results):
    group_id = doc["meta"]["group_id"]
    gold = doc.get("outputs") or ""
    if not gold:
        return {
            "exact_match": 0,
            "group_exact_match": [0, group_id],
            "judge_score": 0.0,
            "group_judge_score": [0.0, group_id],
        }

    # Both readings of the response are scored and the better one kept: the
    # model may answer without the marker, or restate it while thinking out
    # loud. See lm_eval.api.answer_extraction. The maximum is taken over the
    # readings first; the group metrics then carry that single best score, so
    # a group is judged on the same number as the ungrouped metric.
    candidates = answer_candidates(results[0] if results else "")
    exact_score = best_score(
        lambda c: squad_metrics.compute_exact(gold, c), candidates)
    judge_score = best_score(
        lambda c: compute_judge_score(doc, c), candidates)

    return {
        "exact_match": exact_score,
        "group_exact_match": [exact_score, group_id],
        "judge_score": judge_score,
        "group_judge_score": [judge_score, group_id],
    }


def aggregate_group_score(items):
    unzipped = list(zip(*items))
    scores = unzipped[0]
    groups = unzipped[1]
    dct_agg = {}
    for i in range(len(groups)):
        dct_agg.setdefault(groups[i], []).extend([scores[i]])
    metric_list = []
    for group_id in dct_agg:
        if sum(dct_agg[group_id]) == len(dct_agg[group_id]):
            metric_list.extend([1])
        else:
            metric_list.extend([0])
    return sum(metric_list) / len(metric_list)


def doc_to_text(doc):
    return doc["instruction"].format(**doc["inputs"])
