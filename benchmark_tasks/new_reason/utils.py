import os
from typing import Any, Dict, List

from lm_eval.api.filter import Filter
from lm_eval.api.registry import FILTER_REGISTRY, register_filter
from transformers.data.metrics import squad_metrics


def doc_to_text(doc: Dict[str, Any]) -> str:
    return doc["instruction"].format(**doc["inputs"])


def extract_answer(model_answer: str) -> str:
    """
    Extracts the answer from the model's response.
    """
    if not model_answer:
        return ""
    return model_answer.rsplit("Ответ:", 1)[-1].strip()


def compute_judge_score(doc: Dict, model_answer: str) -> float:
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


def process_results(doc: Dict, results: List[str]) -> Dict:
    model_answer = extract_answer(results[0]) if results else ""
    if len(doc.get("outputs", "")) > 0:
        em = squad_metrics.compute_exact(doc["outputs"], model_answer)
        judge_score = compute_judge_score(doc, model_answer)
        return {"exact_match": em, "judge_score": judge_score}
    return {"exact_match": 0, "judge_score": 0.0}


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
                        resp = resp.lstrip().split("Ответ: ")[-1].strip()
                    filtered_resp.append(resp)
                return filtered_resp

            return [filter_set(resp) for resp in resps]
