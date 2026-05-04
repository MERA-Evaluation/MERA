import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from custom_metrics import process_results_group


def process_results(doc, results):
    return process_results_group(doc, results, func=doc_to_text)


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


def doc_to_text(doc):
    return doc["instruction"].format(**doc["inputs"])