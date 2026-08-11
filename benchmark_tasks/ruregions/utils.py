import logging
import os
from lm_eval.api.answer_extraction import answer_candidates, best_score
from transformers.data.metrics import squad_metrics

eval_logger = logging.getLogger(__name__)


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



def compute_judge_score(doc, model_answer):
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    judge_prompt = os.getenv(
        "LM_EVAL_RUREGIONS_JUDGE_PROMPT",
        "Оцени правильность ответа модели о регионе России по шкале от 1 до 10.\n"
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
