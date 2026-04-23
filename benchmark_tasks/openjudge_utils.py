import os
from pathlib import Path

try:
    from lm_eval.api.filter import Filter
    from lm_eval.api.registry import FILTER_REGISTRY, register_filter
except ModuleNotFoundError:
    FILTER_REGISTRY = {}

    class Filter:  # type: ignore[no-redef]
        pass

    def register_filter(filter_name):  # type: ignore[no-redef]
        def decorator(cls):
            FILTER_REGISTRY[filter_name] = cls
            return cls

        return decorator

from mera_openjudge import OpenJudgeScorer, load_judge_config


def _load_runtime_config_from_env():
    return {
        "api_key": os.environ.get("MERA_JUDGE_API_KEY", ""),
        "backend": "openai_compatible",
        "base_url": os.environ.get("MERA_JUDGE_BASE_URL", ""),
        "max_new_tokens": int(os.environ.get("MERA_JUDGE_MAX_NEW_TOKENS", "128")),
        "model_name": os.environ.get("MERA_JUDGE_MODEL", "ai-forever/pollux-judge-7b"),
        "temperature": float(os.environ.get("MERA_JUDGE_TEMPERATURE", "0.1")),
        "timeout": int(os.environ.get("MERA_JUDGE_TIMEOUT", "120")),
    }


def _judge_config_path(task_dir):
    return str(Path(task_dir) / "judge.yaml")


def build_process_results(task_dir):
    judge_config = load_judge_config(_judge_config_path(task_dir))
    zero_metrics = {
        judge_config.metric_name: 0.0,
        **{f"judge_{criterion.key}": 0.0 for criterion in judge_config.criteria},
    }

    def process_results(doc, results):
        del doc
        if not results or not results[0]:
            return dict(zero_metrics)
        return dict(results[0])

    return process_results


def register_openjudge_filter(filter_name, task_dir):
    if FILTER_REGISTRY.get(filter_name, None):
        return filter_name

    judge_config_path = _judge_config_path(task_dir)

    @register_filter(filter_name)
    class OpenJudgeScoring(Filter):
        def __init__(self) -> None:
            self.judge_config_path = judge_config_path
            self.runtime_config = _load_runtime_config_from_env()
            self._scorer = None

        def _get_scorer(self):
            if self._scorer is None:
                self._scorer = OpenJudgeScorer(
                    judge_config=self.judge_config_path,
                    runtime_config=self.runtime_config,
                )
            return self._scorer

        def apply(self, resps, docs):
            answers = []
            for sample in resps:
                answer = ""
                if sample:
                    answer = str(sample[0]).strip()
                answers.append(answer)
            return [[metrics] for metrics in self._get_scorer().score_answers(docs, answers)]

    return filter_name
