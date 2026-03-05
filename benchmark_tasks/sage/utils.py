from typing import Dict, List
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
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

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    if Scorer is None:
        return {}

    scorer = Scorer()


    # распределение ошибок в датасете (в процентах)
    weights = {
        "CASE_F1": 13.6,
        "YO_F1": 33.2,
        "SPELL_F1": 58.0,
        "PUNCT_F1": 43.2,
    }

    total_weight = sum(weights.values())

    source = doc["inputs"]["source"]
    reference = doc["outputs"]

    prediction = results[0]

    metric = scorer.score(
        [source],
        [reference],
        [prediction],
        metrics=["errant"]
    )

    case_f1 = metric.get("CASE_F1", 0.0) / 100.0
    yo_f1 = metric.get("YO_F1", 0.0) / 100.0
    spell_f1 = metric.get("SPELL_F1", 0.0) / 100.0
    punct_f1 = metric.get("PUNCT_F1", 0.0) / 100.0

    errant = (
        case_f1 * weights["CASE_F1"] +
        yo_f1 * weights["YO_F1"] +
        spell_f1 * weights["SPELL_F1"] +
        punct_f1 * weights["PUNCT_F1"]
    ) / total_weight

    return {
        "errant": errant,
        "CASE_F1": case_f1,
        "YO_F1": yo_f1,
        "SPELL_F1": spell_f1,
        "PUNCT_F1": punct_f1
    }

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
