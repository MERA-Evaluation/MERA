from typing import Dict, List, Any
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from transformers.data.metrics import squad_metrics
import logging

eval_logger = logging.getLogger(__name__)


try:
    from sage.evaluation.scorer import Scorer
except ImportError:
    Scorer = None
    eval_logger.warning(
    "SAGE is not installed. It is required to compute metrics for the SAGE task.\n\n"
    "If you are running with --predict_only or are not evaluating this task, "
    "you can safely ignore this warning.\n\n"
    "To install SAGE and its dependencies, run the following commands:\n"
    "  pip install 'sage-spelling[errant]'\n"
    "  python -m spacy download ru_core_news_lg"
)

def doc_to_text(doc: Dict[str, Any]) -> str:
    return doc["instruction"].format(**doc["inputs"])

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    if Scorer is None:
        return {}

    scorer = Scorer()


    gold = doc["outputs"]
    source = doc["inputs"]["source"]

    pred = results[0] if results and results[0] else ""

    if not pred or not pred.strip():
        return {"errant_f1": 0.0, "em": 0.0}

    metric = scorer.score(
    [source],
    [gold],
    [pred],
    metrics=["errant"]
    )

    spell_f1 = metric.get("SPELL_F1", 0.0) / 100.0
    punct_f1 = metric.get("PUNCT_F1", 0.0) / 100.0

    combined_f1 = (spell_f1 + punct_f1) / 2


    em = squad_metrics.compute_exact(gold, pred)

    return {"spell_f1": spell_f1,
            "punct_f1": punct_f1,
            "errant_f1": combined_f1, "em": em}

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
