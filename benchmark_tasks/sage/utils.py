from typing import Dict, List, Any
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

ZERO_SCORES = {"errant_f1": 0.0, "spell_f1": 0.0, "punct_f1": 0.0}


def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    gold = doc.get("outputs") or ""
    source = doc["inputs"]["source"]

    pred = results[0] if results and results[0] else ""

    # The public copy of the dataset ships with `outputs` blanked, so there is
    # nothing to score against; the same happens when the model answered with
    # nothing. Return the declared metrics as zeros instead of running (or
    # loading) the scorer, which would otherwise be handed an empty reference.
    if not gold.strip() or not pred.strip():
        return dict(ZERO_SCORES)

    if Scorer is None:
        return dict(ZERO_SCORES)

    scorer = Scorer()

    metric = scorer.score(
    [source],
    [gold],
    [pred],
    metrics=["errant"]
    )

    spell_f1 = metric.get("SPELL_F1", 0.0) / 100.0
    punct_f1 = metric.get("PUNCT_F1", 0.0) / 100.0

    combined_f1 = (spell_f1 + punct_f1) / 2


    return {"spell_f1": spell_f1,
            "punct_f1": punct_f1,
            "errant_f1": combined_f1}
