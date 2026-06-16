from typing import Dict, List, Any
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, FILTER_REGISTRY
from transformers.data.metrics import squad_metrics


def doc_to_text(doc: Dict[str, Any]) -> str:

    return doc["instruction"].format(**doc["inputs"])


def process_results(doc: Dict, results: List[str]) -> Dict:
    
    if len(doc["outputs"]) > 0:
        gold_label = doc["outputs"]
        pred_label = results[0]
        if "Ответ:" in pred_label:
            pred_label = pred_label.split("Ответ:")[-1].strip()

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