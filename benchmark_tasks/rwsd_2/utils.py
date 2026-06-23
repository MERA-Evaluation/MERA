import os
from transformers.data.metrics import squad_metrics


def process_results(doc, results):
    model_answer = extract_answer(results[0])
    exact_score = squad_metrics.compute_exact(doc["outputs"], model_answer) if doc["outputs"] else 0
    judge_score = compute_judge_score(doc, model_answer)

    return {
        "exact_match": exact_score,
        "judge_score": judge_score,
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


def doc_to_text(doc):
    return doc["instruction"].format(**doc["inputs"])