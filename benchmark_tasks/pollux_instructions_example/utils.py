from pathlib import Path

import datasets

from benchmark_tasks.openjudge_utils import build_process_results, register_openjudge_filter


TASK_DIR = Path(__file__).resolve().parent
FILTER_NAME = register_openjudge_filter("polluxinstructionsexamplescoring", TASK_DIR)
EXAMPLE_PROMPT_IDS = {0}


def _process_doc(doc):
    return {
        "instruction": doc["instruction"],
        "inputs": "",
        "outputs": doc.get("reference_answer", "") or "",
        "meta": {
            "id": int(doc["prompt_id"]),
            "reference_answer": doc.get("reference_answer", "") or "",
            "source_dataset": "ai-forever/POLLUX-instructions",
            "source_prompt_id": int(doc["prompt_id"]),
            "task_type": doc.get("task_type", ""),
            "task_subtype": doc.get("task_subtype", ""),
            "task_subsubtype": doc.get("task_subsubtype", ""),
            "difficulty": doc.get("difficulty", ""),
            "domain": doc.get("domain", ""),
        },
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    filtered = dataset.filter(lambda doc: doc["prompt_id"] in EXAMPLE_PROMPT_IDS)
    return filtered.map(_process_doc)


process_results = build_process_results(TASK_DIR)

