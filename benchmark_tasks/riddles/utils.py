from typing import Dict, List, Any
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def doc_to_text(doc: Dict[str, Any]) -> str:
    return doc["instruction"].format(**doc["inputs"])


def normalize(text: str) -> str:
    return text.strip().lower()


def process_results(doc: Dict, results: List[str]) -> Dict:
    if len(doc["outputs"]) == 0:
        return {"em": 0}

    gold = doc["outputs"]
    pred = results[0]

    gold_variants = [
        normalize(x) for x in re.split(r"[;,]", gold) if x.strip()
    ]

    pred = normalize(pred)

    em = int(pred in gold_variants)

    return {"em": em}



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
