from typing import Any, Mapping, Optional, Sequence

from mera_openjudge.backends.hf import HFJudgeBackend
from mera_openjudge.backends.openai_compatible import OpenAICompatibleJudgeBackend
from mera_openjudge.config import JudgeConfig, load_judge_config
from mera_openjudge.exceptions import JudgeBackendError, JudgeParseError
from mera_openjudge.parsing import parse_judge_score
from mera_openjudge.prompting import render_judge_prompt


_BACKEND_REGISTRY = {
    "hf": HFJudgeBackend,
    "openai_compatible": OpenAICompatibleJudgeBackend,
}


def _to_plain_mapping(conf_source: Any):
    if conf_source is None:
        return {}
    if isinstance(conf_source, Mapping):
        return dict(conf_source)
    if hasattr(conf_source, "items"):
        return dict(conf_source.items())
    raise JudgeBackendError("Unsupported judge runtime config.")


class OpenJudgeScorer:
    def __init__(
        self,
        judge_config: Any,
        runtime_config: Optional[Any] = None,
        backend=None,
    ):
        self.judge_config: JudgeConfig = load_judge_config(judge_config)
        self.runtime_config = _to_plain_mapping(runtime_config)
        self.backend = backend or self._build_backend(self.runtime_config)

    @staticmethod
    def _build_backend(runtime_config: Mapping[str, Any]):
        backend_name = str(runtime_config.get("backend", "hf")).strip() or "hf"
        backend_cls = _BACKEND_REGISTRY.get(backend_name)
        if backend_cls is None:
            raise JudgeBackendError(f"Unsupported judge backend: {backend_name}")
        return backend_cls(runtime_config)

    def _score_criterion(
        self,
        docs: Sequence[Mapping[str, Any]],
        answers: Sequence[str],
        criterion,
    ):
        prompts = [
            render_judge_prompt(
                doc=doc,
                answer=answer,
                criterion=criterion,
                judge_config=self.judge_config,
            )
            for doc, answer in zip(docs, answers)
        ]
        responses = self.backend.generate(prompts)
        if len(responses) != len(docs):
            raise JudgeBackendError("Judge backend returned an unexpected number of responses.")
        scores = []
        for doc_index, response_text in enumerate(responses):
            try:
                scores.append(parse_judge_score(response_text, self.judge_config.allowed_scores))
            except JudgeParseError as exc:
                raise JudgeParseError(
                    str(exc),
                    doc_index=doc_index,
                    criterion_key=criterion.key,
                    response_text=response_text,
                ) from exc
        return scores

    def score_answers(self, docs: Sequence[Mapping[str, Any]], answers: Sequence[str]):
        if len(docs) != len(answers):
            raise JudgeBackendError("Docs and answers must have the same length.")
        if not docs:
            return []

        scores_by_criterion = {}
        for criterion in self.judge_config.criteria:
            scores_by_criterion[criterion.key] = self._score_criterion(docs, answers, criterion)

        metrics = []
        for doc_index in range(len(docs)):
            doc_metrics = {}
            criterion_scores = []
            for criterion in self.judge_config.criteria:
                score = float(scores_by_criterion[criterion.key][doc_index])
                doc_metrics[f"judge_{criterion.key}"] = score
                criterion_scores.append(score)
            doc_metrics[self.judge_config.metric_name] = sum(criterion_scores) / len(criterion_scores)
            metrics.append(doc_metrics)
        return metrics
