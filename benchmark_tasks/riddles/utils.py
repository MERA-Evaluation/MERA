import os
import re
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import FILTER_REGISTRY, register_filter


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["instruction"].format(**doc["inputs"])


def normalize(text: str) -> str:
    return text.strip().lower()


def process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, int | float]:
    if len(doc["outputs"]) == 0:
        return {"exact_match": 0, "judge_score": 0.0}

    gold = doc["outputs"]
    pred = results[0]

    gold_variants = [
        normalize(x) for x in re.split(r";", gold) if x.strip()
    ]

    pred = normalize(pred)

    exact_match = int(pred in gold_variants)
    judge_score = compute_judge_score(doc, pred)

    return {
        "exact_match": exact_match,
        "judge_score": judge_score,
    }


def compute_judge_score(
    doc: dict[str, Any], model_answer: str
) -> float:
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    judge_prompt_path = os.getenv("LM_EVAL_JUDGE_PROMPT_PATH")

    if not doc.get("outputs") or not judge_api_base or not judge_model:
        return 0.0

    try:
        from lm_eval.api.metrics_generative import compute_llm_judge

        return float(
            compute_llm_judge(
                [model_answer],
                [doc["outputs"]],
                api_base=judge_api_base,
                model=judge_model,
                judge_prompt_path=judge_prompt_path,
                instruction=doc_to_text(doc),
            )["llm_judge"]
        )
    except Exception:
        return 0.0


if "remove_whitespace_and_nones" not in FILTER_REGISTRY:

    @register_filter("remove_whitespace_and_nones")
    class RemoveWhitespaceAndNones(Filter):
        def apply(
            self,
            resps: list[list[str]],
            docs: list[dict[str, Any]],
        ) -> list[list[str]]:
            def filter_set(inst: list[str]) -> list[str]:
                filtered_resp: list[str] = []
                for resp in inst:
                    if not resp:
                        resp = ""
                    else:
                        resp = resp.lstrip().split("Ответ: ")[-1].strip()
                    filtered_resp.append(resp)
                return filtered_resp

            filtered_resps = [filter_set(resp) for resp in resps]

            return filtered_resps
