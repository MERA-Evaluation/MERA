from typing import Dict, List, Any
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, FILTER_REGISTRY
from transformers.data.metrics import squad_metrics
import logging
import os

import sacrebleu
from transformers.data.metrics import squad_metrics

from lm_eval.api.metrics import (
    compute_rouge_fn,
    exact_match_hf_evaluate,
    metric_max_over_ground_truths,
)

eval_logger = logging.getLogger(__name__)

eval_logger.disabled = True


def doc_to_text(doc: Dict[str, Any]) -> str:

    return doc["instruction"].format(**doc["inputs"])


def parse_embedding_env():
    raw = os.getenv("LM_EVAL_EMBEDDING_MODELS", "")
    if not raw:
        return []

    models = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue

        try:
            base, model = item.split("|", 1)
            models.append((base.strip(), model.strip()))
        except ValueError:
            eval_logger.warning(
                "Invalid embedding config: %s. "
                "Expected format: api_base|model_name",
                item,
            )

    return models


def _zeros_generative_bundle() -> Dict[str, float]:
    return {
        "em": 0.0,
        "f1": 0.0,
        "f1_gen": 0.0,
        "exact_match": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "sentence_bleu": 0.0,
        "levenshtein": 0.0,
        "token_overlap_f1": 0.0,
        "meteor": 0.0,
        "bertscore": 0.0,
        "comet": 0.0,
        "bleurt": 0.0,
        "embedding_cosine_qwen_8b": 0.0,
        "embedding_cosine_embeddinggemma_300m": 0.0,
        "embedding_cosine_ruroberta_large": 0.0,
        "llm_judge": 0.0,
    }

def process_results_generative_metrics(doc: Dict, results: List[str]) -> Dict:
    """CheGeKa-style max over `;`-separated references, with extended MERA generative metrics.

    *Corpus* metrics (`bleu` / `chrf` / `ter` passthrough) are not computed here; use a task without
    custom ``process_results`` if you need corpus BLEU.

    **Embedding similarity metrics** are configured through:

    - ``LM_EVAL_EMBEDDING_MODELS``
    
    Format:
    
        api_base|model_name;api_base|model_name
    
    Example:
    
        http://localhost:2000|Qwen/Qwen3-Embedding-8B;
        http://localhost:2001|google/embeddinggemma-300m
    
    **LLM judge metric** is configured through:
    
    - ``LM_EVAL_JUDGE_API_BASE``
    - ``LM_EVAL_JUDGE_MODEL``
    - Optional: ``LM_EVAL_JUDGE_PROMPT``
    
    If API configuration is missing or metric computation fails,
    the corresponding score defaults to ``0.0``.
    """
    if not doc.get("outputs"):
        return _zeros_generative_bundle()

    gold_label_set = [doc["outputs"]]
    if not gold_label_set:
        return _zeros_generative_bundle()

    pred = results[0]

    em = metric_max_over_ground_truths(squad_metrics.compute_exact, pred, gold_label_set)
    f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, pred, gold_label_set)

    def max_exact_match_hf(**kwargs) -> float:
        return max(
            float(
                exact_match_hf_evaluate(predictions=[pred], references=[r], **kwargs)[
                    "exact_match"
                ]
            )
            for r in gold_label_set
        )

    out: Dict[str, float] = {
        "em": em,
        "f1": f1,
        "f1_gen": f1,
        "exact_match": max_exact_match_hf(),
    }

    for rouge_type, key in (
        ("rouge1", "rouge1"),
        ("rouge2", "rouge2"),
        ("rougeL", "rougeL"),
    ):
        out[key] = max(
            float(
                compute_rouge_fn([pred], [r], rouge_type=rouge_type)["rouge"]
            )
            for r in gold_label_set
        )


    out["sentence_bleu"] = max(
        float(sacrebleu.sentence_bleu(pred, [r]).score) for r in gold_label_set
    )

    try:
        from lm_eval.api.metrics_generative import compute_levenshtein

        out["levenshtein"] = max(
            compute_levenshtein([pred], [r])["levenshtein"] for r in gold_label_set
        )
    except ImportError:
        eval_logger.debug("levenshtein skipped: install rapidfuzz (lm_eval[generative_metrics])")
        out["levenshtein"] = 0.0

    from lm_eval.api.metrics_generative import compute_token_overlap_f1

    out["token_overlap_f1"] = max(
        compute_token_overlap_f1([pred], [r])["token_overlap_f1"] for r in gold_label_set
    )

    try:
        from lm_eval.api.metrics_generative import compute_meteor

        out["meteor"] = max(
            compute_meteor([pred], [r])["meteor"] for r in gold_label_set
        )
    except Exception as e:
        eval_logger.debug("meteor skipped: %s", e)
        out["meteor"] = 0.0

    try:
        from lm_eval.api.metrics_generative import compute_bertscore

        out["bertscore"] = max(
            compute_bertscore([pred], [r])["bertscore"] for r in gold_label_set
        )
    except Exception as e:
        eval_logger.debug("bertscore skipped: %s", e)
        out["bertscore"] = 0.0

    try:
        from lm_eval.api.metrics_generative import compute_comet

        out["comet"] = max(compute_comet([pred], [r])["comet"] for r in gold_label_set)
    except Exception as e:
        eval_logger.debug("comet skipped: %s", e)
        out["comet"] = 0.0

    try:
        from lm_eval.api.metrics_generative import compute_bleurt

        out["bleurt"] = max(compute_bleurt([pred], [r])["bleurt"] for r in gold_label_set)
    except Exception as e:
        eval_logger.debug("bleurt skipped: %s", e)
        out["bleurt"] = 0.0

    from lm_eval.api.metrics_generative import compute_embedding_cosine

    embedding_models = parse_embedding_env()
    for base, model in embedding_models:
        safe_name = model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        key = f"embedding_cosine_{safe_name}"
    
        try:
            out[key] = max(
                compute_embedding_cosine(
                    [pred],
                    [r],
                    api_base=base,
                    model=model,
                    use_cache=True,
                )["embedding_cosine"]
                for r in gold_label_set
            )
        except Exception as e:
            # print(e)
            eval_logger.debug("embedding %s failed: %s", model, e)
            out[key] = 0.0


    j_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    j_model = os.getenv("LM_EVAL_JUDGE_MODEL")
    j_prompt_path = os.getenv("LM_EVAL_JUDGE_PROMPT")
    
    if j_base and j_model:
        from lm_eval.api.metrics_generative import compute_llm_judge
    
        out["llm_judge"] = max(
            compute_llm_judge(
                [pred],
                [r],
                api_base=j_base,
                model=j_model,
                judge_prompt_path=j_prompt_path,
                instruction=doc["instruction"],
            )["llm_judge"]
            for r in gold_label_set
        )
    else:
        out["llm_judge"] = 0.0
    return out


def process_results(doc: Dict, results: List[str]) -> Dict:
    if len(doc["outputs"]) > 0:
        gold_label = doc["outputs"]
        pred_label = results[0]

        em = squad_metrics.compute_exact(gold_label, pred_label)

        return {"em": em}
    return {"em": 0}


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
                        resp = resp.lstrip()
                    filtered_resp.append(resp)
                return filtered_resp
    
            filtered_resps = [filter_set(resp) for resp in resps]
    
            return filtered_resps
