from transformers.data.metrics.squad_metrics import compute_exact


def process_results_template(doc: dict, results: list[str], func) -> dict:
    model_answer = results[0]
    if len(doc["outputs"]) > 0:
        return {
            "exact_match": compute_exact(doc["outputs"], model_answer),
        }
    return {
        "exact_match": 0,
    }


def process_results_group(doc: dict, results: list[str], func) -> dict:
    model_answer = results[0]
    if len(doc["outputs"]) > 0:
        return {
            "exact_match": compute_exact(doc["outputs"], model_answer),
            "group_exact_match": [compute_exact(doc["outputs"], model_answer), doc["meta"]["group_id"]],
        }
    return {
        "exact_match": 0,
        "group_exact_match": [0, 0]
    }


def aggregate_group_score(items):
    unzipped = list(zip(*items))
    scores = unzipped[0]
    groups = unzipped[1]
    dct_agg = {}
    for i in range(len(groups)):
        dct_agg.setdefault(groups[i], []).extend([scores[i]])
    metric_list = []
    for group_id in dct_agg:
        if sum(dct_agg[group_id]) == len(dct_agg[group_id]):
            metric_list.extend([1])
        else:
            metric_list.extend([0])
    return sum(metric_list) / len(metric_list)