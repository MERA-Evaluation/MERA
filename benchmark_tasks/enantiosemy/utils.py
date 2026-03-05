from typing import Dict, List, Any
from transformers.data.metrics import squad_metrics




def doc_to_text(doc: Dict[str, Any]) -> str:
    inputs = doc["inputs"]

    return (
        "{task}\n\n{text}\n\n"
        "А) {option_a}\n"
        "Б) {option_b}\n"
        "В) {option_c}\n"
        "Г) {option_d}\n\n"
        "Ответ:"
    ).format(**inputs)


def process_results(doc: Dict, results: List[str]) -> Dict:
    if len(doc["outputs"]) > 0:
        gold_label = doc["outputs"]
        pred_label = results[0]

        em = squad_metrics.compute_exact(gold_label, pred_label)

        return {"em": em}
    return {"em": 0} 
    