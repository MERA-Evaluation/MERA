import random
from typing import Dict
from src.registry import register_task
from src.tasks.task import Task
import numpy as np
import re


@register_task
class USE(Task):

    def process_results(self, doc_true, doc_pred) -> Dict:
        id_task = doc_true["meta"]["id_task"]
        task_type = doc_true["meta"]["type"]
        variant = doc_true["meta"]["variant"]
        max_score = doc_true["meta"]["score"]
        answer = doc_true["outputs"]
        prediction = doc_pred["outputs"]

        score = self.get_scores(task_type, id_task, answer, prediction)

        return {
            "grade_norm": (score, variant, max_score),
            f"grade_norm.task{id_task}": (score, id_task, max_score)
        }

    @staticmethod
    def multiple_choice_score(answer: str, prediction: str, is_task16=False) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        if is_task16:
            while len(pred) < len(ans):
                pred.append("-1")
            return max(
                0,
                len(set.intersection(set(ans), set(pred))) - len(pred) + len(ans),
            )
        else:
            ans = set(ans)
            pred = set(pred)
            return int(len(set.intersection(ans, pred)) == len(ans) == len(pred))

    @staticmethod
    def matching_score(answer: str, prediction: str) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        score = 0
        if len(ans) != len(pred):
            # print('Format Error: The prediction must contain a string of 4 numbers separated by ","')
            return 0
        for idx, num in enumerate(ans):
            if num == pred[idx]:
                score += 1
        return score

    @staticmethod
    def text_score(answer: str, prediction: str) -> int:
        pred = re.sub(r"[\d+\W+]", "", prediction).lower()
        ans = answer.split(",")
        if pred in ans:
            return 1
        return 0

    def get_scores(self, task_type, id_task, answer, prediction):
        if task_type == "matching":
            score = self.matching_score(answer, prediction)
        elif task_type == "text":
            score = self.text_score(answer, prediction)
        else:
            is_task16 = False
            if id_task == "16":
                is_task16 = True
            score = self.multiple_choice_score(answer, prediction, is_task16)
        return score

    @staticmethod
    def overall_score(items):
        overall_scores = {}
        overall_max_scores = {}
        for item in items:
            score, variant, max_score = item[0], item[1], item[2]
            if variant not in overall_scores:
                overall_scores[variant] = 0
            overall_scores[variant] += score
            if variant not in overall_max_scores:
                overall_max_scores[variant] = 0
            overall_max_scores[variant] += max_score

        average_overall_score = np.mean(
            [score / overall_max_scores[variant] for variant, score in overall_scores.items()]
        )
        return average_overall_score

    @staticmethod
    def task_score(items):
        overall_scores = []
        for item in items:
            score, id_task, max_score = item[0], item[1], item[2]
            overall_scores.append(score / max_score)

        average_overall_score = np.mean(overall_scores)
        return average_overall_score

    def aggregation(self):
        res = {
            "grade_norm": self.overall_score,
        }
        for idx in range(1, 27):
            res[f"grade_norm.task{idx}"] = self.task_score
        res.pop("grade_norm.task8")
        for idx in range(5):
            res[f"grade_norm.task8_{idx}"] = self.task_score
        return res

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            origin_doc = self.gold[doc_id]
            doc = {
                "meta": {
                    "id": doc_id,
                    "id_task": origin_doc["meta"]["id_task"],
                    "variant": origin_doc["meta"]["variant"]
                }
            }
            if origin_doc["meta"]["type"] == "text":
                doc["outputs"] = str(np.random.choice(origin_doc["inputs"]["text"].split()))
            else:
                doc["outputs"] = str(random.randint(1, 4))
                if random.random() < 0.5:
                    doc["outputs"] += f",{random.randint(1, 4)}"
            res.append(doc)
        return {"data": {self.split: res}}
