import logging
import os
from typing import Dict, List

import sacrebleu
from transformers.data.metrics import squad_metrics

from lm_eval.api.metrics import (
    compute_rouge_fn,
    exact_match_hf_evaluate,
    metric_max_over_ground_truths,
)

eval_logger = logging.getLogger(__name__)


def process_results(doc: Dict, results: List[str]) -> Dict:
    # - Pick the maximum likelihood prediction entity
    # - Evaluate the accuracy and token F1 PER EXAMPLE
    # - Average over all examples
    if len(doc["outputs"]) > 0:
        gold_label_set = doc["outputs"].split(";")
        pred = results[0]

        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, pred, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, pred, gold_label_set
        )

        return {"f1": f1, "em": em}
    return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)


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
        "embedding_cosine": 0.0,
        "llm_judge": 0.0,
    }


def process_results_generative_metrics(doc: Dict, results: List[str]) -> Dict:
    """CheGeKa-style max over `;`-separated references, with extended MERA generative metrics.

    *Corpus* metrics (`bleu` / `chrf` / `ter` passthrough) are not computed here; use a task without
    custom ``process_results`` if you need corpus BLEU.

    **API metrics** (``embedding_cosine``, ``llm_judge``): set env vars or scores stay ``0.0``:

    - ``LM_EVAL_CHEGEKA_EMBEDDING_API_BASE``, ``LM_EVAL_CHEGEKA_EMBEDDING_MODEL``
    - ``LM_EVAL_CHEGEKA_JUDGE_API_BASE``, ``LM_EVAL_CHEGEKA_JUDGE_MODEL``
    - Optional: ``LM_EVAL_CHEGEKA_JUDGE_PROMPT`` (must include ``{reference}`` and ``{prediction}``)
    """
    if not doc.get("outputs"):
        return _zeros_generative_bundle()

    gold_label_set = [x.strip() for x in doc["outputs"].split(";") if x.strip()]
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

    emb_base = os.getenv("LM_EVAL_CHEGEKA_EMBEDDING_API_BASE")
    emb_model = os.getenv("LM_EVAL_CHEGEKA_EMBEDDING_MODEL")
    if emb_base and emb_model:
        from lm_eval.api.metrics_generative import compute_embedding_cosine

        out["embedding_cosine"] = max(
            compute_embedding_cosine(
                [pred],
                [r],
                api_base=emb_base,
                model=emb_model,
                use_cache=True,
            )["embedding_cosine"]
            for r in gold_label_set
        )
    else:
        out["embedding_cosine"] = 0.0

    j_base = os.getenv("LM_EVAL_CHEGEKA_JUDGE_API_BASE")
    j_model = os.getenv("LM_EVAL_CHEGEKA_JUDGE_MODEL")
    j_prompt = os.getenv(
        "LM_EVAL_CHEGEKA_JUDGE_PROMPT",
        "Оцени ответ от 1 до 10.\nЭталон: {reference}\nОтвет модели: {prediction}\nОдно число:",
    )
    if j_base and j_model:
        from lm_eval.api.metrics_generative import compute_llm_judge

        out["llm_judge"] = max(
            compute_llm_judge(
                [pred],
                [r],
                api_base=j_base,
                model=j_model,
                judge_prompt=j_prompt,
            )["llm_judge"]
            for r in gold_label_set
        )
    else:
        out["llm_judge"] = 0.0

    return out
