import os
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, FILTER_REGISTRY
from transformers.data.metrics import squad_metrics


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


if "remove_whitespace_and_nones" not in FILTER_REGISTRY:
    @register_filter("remove_whitespace_and_nones")
    class RemoveWhitespaceAndNones(Filter):
    
        def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
            def filter_set(inst):
                filtered_resp = []
                for resp in inst:
                    if not resp:
                        resp = ""
                    else:
                        resp = resp.lstrip().split('Ответ: ')[-1].strip()
                    filtered_resp.append(resp)
                return filtered_resp
    
            filtered_resps = [filter_set(resp) for resp in resps]
    
            return filtered_resps