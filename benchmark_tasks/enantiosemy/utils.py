import logging
import os
from typing import Any, Dict, List

from lm_eval.api.answer_extraction import answer_candidates, best_score
from transformers.data.metrics import squad_metrics

eval_logger = logging.getLogger(__name__)


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



def compute_judge_score(doc: Dict, model_answer: str) -> float:
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    judge_prompt = os.getenv(
        "LM_EVAL_ENANTIOSEMY_JUDGE_PROMPT",
        "Оцени, верно ли модель определила значение энантиосемичного слова, по шкале от 1 до 10.\n"
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
