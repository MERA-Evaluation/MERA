from typing import Dict, List
from sage.evaluation.scorer import Scorer

scorer = Scorer()

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:


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
    }
