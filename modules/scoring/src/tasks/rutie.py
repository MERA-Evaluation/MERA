from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from src.utils import load_json
from typing import Dict
import numpy as np


@register_task
class ruTiE(Task):
    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        return {"acc": y_true == y_pred}

    def doc_to_meta(self, doc):
        return doc["meta"]

    def doc_to_id(self, doc):
        dialog_id = self.doc_to_meta(doc)["dialog_id"]
        question_id = self.doc_to_meta(doc)["question_id"]
        return f"dialog_id:{dialog_id};question_id:{question_id}"

    def doc_to_y_true(self, doc):
        return doc["outputs"]

    def doc_to_y_pred(self, doc):
        return doc["outputs"]

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            docs = []
            for origin_doc in self.gold[doc_id]:
                doc = {
                    "outputs": str(np.random.choice(self.choices)),
                    "meta": {
                        "dialog_id": origin_doc["meta"]["dialog_id"],
                        "question_id": origin_doc["meta"]["question_id"],
                    }
                }
                docs.append(doc)
            res.append(docs)
        return {"data": {self.split: res}}

    def remove_outputs(self):
        task = load_json(self.task_conf.origin_repo_path)
        for idx in range(len(task["data"]["test"])):
            for idy in range(len(task["data"]["test"][idx])):
                task["data"]["test"][idx][idy]["outputs"] = ""
        return task
