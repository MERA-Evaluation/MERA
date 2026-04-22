from collections import defaultdict
from pathlib import Path
from typing import Dict

from src.enums import Errors
from src.metrics import mean
from src.tasks.task import Task

from mera_openjudge import JudgeBackendError, JudgeConfigError, JudgeParseError, OpenJudgeScorer


class OpenJudgeTask(Task):
    def __init__(self, conf):
        super().__init__(conf)
        self._scorer = None
        self.judge_config = None
        self._aggregation = None
        self._load_judge_config()
        if not getattr(self.conf, "no_load_models", False):
            self._get_scorer()

    @property
    def repo_root(self):
        return Path(__file__).resolve().parents[4]

    @property
    def judge_config_path(self):
        configured_path = getattr(self.task_conf, "judge_config_path", None)
        if configured_path:
            path = Path(str(configured_path))
            if not path.is_absolute():
                path = self.repo_root / path
            return path
        return self.repo_root / "benchmark_tasks" / self.name / "judge.yaml"

    def _load_judge_config(self):
        scorer = OpenJudgeScorer(
            judge_config=str(self.judge_config_path),
            runtime_config=self._get_runtime_config(),
            backend=_NoopBackend(),
        )
        self.judge_config = scorer.judge_config

    def _get_runtime_config(self):
        judges_conf = getattr(self.conf, "judges", None)
        if judges_conf is None:
            return {}
        return getattr(judges_conf, "pollux", {})

    def _get_scorer(self):
        if self._scorer is None:
            self._scorer = OpenJudgeScorer(
                judge_config=str(self.judge_config_path),
                runtime_config=self._get_runtime_config(),
            )
        return self._scorer

    def aggregation(self) -> Dict:
        if self._aggregation is None:
            metric_name = self.judge_config.metric_name
            self._aggregation = {
                metric_name: mean,
                **{f"judge_{criterion.key}": mean for criterion in self.judge_config.criteria},
            }
        return self._aggregation

    def average_results(self, metrics: Dict) -> float:
        return float(metrics[self.judge_config.metric_name])

    def evaluate(self, local_path):
        self.log(f"Start evaluating dataset {self.name}")
        dataset, errors = self.load_and_validate(local_path=local_path)
        results = {}
        vals = defaultdict(list)
        docs = []
        answers = []

        if not len(errors):
            for doc_id in self.gold.doc_ids():
                if doc_id not in dataset.examples:
                    self.log(f"{Errors.no_id} {self.name}")
                    errors.append({"type": str(Errors.no_id), "doc_id": doc_id})
                    continue
                answer = self.doc_to_y_pred(dataset[doc_id])
                if not isinstance(answer, str):
                    errors.append({"type": str(Errors.doc_output_type_error), "doc_id": doc_id})
                    continue
                docs.append(self.gold[doc_id])
                answers.append(answer)

        if not len(errors):
            try:
                metrics_per_doc = self._get_scorer().score_answers(docs=docs, answers=answers)
            except JudgeConfigError:
                errors.append({"type": str(Errors.judge_config_error)})
                metrics_per_doc = []
            except JudgeBackendError:
                errors.append({"type": str(Errors.judge_backend_error)})
                metrics_per_doc = []
            except JudgeParseError as exc:
                doc_id = None
                if exc.doc_index is not None and 0 <= exc.doc_index < len(docs):
                    doc_id = self.doc_to_id(docs[exc.doc_index])
                errors.append(
                    {
                        "type": str(Errors.judge_parse_error),
                        **({"doc_id": doc_id} if doc_id is not None else {}),
                    }
                )
                metrics_per_doc = []

            if not len(errors):
                for metric_dict in metrics_per_doc:
                    for metric_name, metric_value in metric_dict.items():
                        vals[metric_name].append(metric_value)
                for metric_name, items in vals.items():
                    results[metric_name] = self.aggregation()[metric_name](items)

        return results, errors

    def process_results(self, doc_true, doc_pred):
        del doc_true, doc_pred
        raise NotImplementedError("OpenJudgeTask computes metrics in batched evaluate().")


class _NoopBackend:
    def generate(self, prompts):
        return ["" for _ in prompts]
