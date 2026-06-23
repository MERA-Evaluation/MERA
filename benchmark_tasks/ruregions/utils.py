import os
import sys
from transformers.data.metrics import squad_metrics

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def process_results(doc, results):
    model_answer = extract_answer(results[0])
    group_id = doc["meta"]["group_id"]
    exact_score = squad_metrics.compute_exact(doc["outputs"], model_answer) if doc["outputs"] else 0
    judge_score = compute_judge_score(doc, model_answer)

    return {
        "exact_match": exact_score,
        "group_exact_match": [exact_score, group_id],
        "judge_score": judge_score,
        "group_judge_score": [judge_score, group_id],
    }


def extract_answer(model_answer):
    if not model_answer:
        return ""
    return model_answer.rsplit("Ответ:", 1)[-1].strip()


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