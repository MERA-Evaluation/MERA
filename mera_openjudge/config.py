from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml

from mera_openjudge.exceptions import JudgeConfigError


@dataclass(frozen=True)
class JudgeCriterion:
    key: str
    name: str
    scale: Dict[int, str]


@dataclass(frozen=True)
class JudgeConfig:
    prompt_template: Optional[str]
    criteria: Tuple[JudgeCriterion, ...]
    reference_field: str
    metric_name: str
    allowed_scores: Tuple[int, ...]


def _to_plain_mapping(conf_source: Any) -> Dict[str, Any]:
    if isinstance(conf_source, JudgeConfig):
        return {
            "prompt_template": conf_source.prompt_template,
            "criteria": [
                {
                    "key": criterion.key,
                    "name": criterion.name,
                    "scale": criterion.scale,
                }
                for criterion in conf_source.criteria
            ],
            "reference_field": conf_source.reference_field,
            "metric_name": conf_source.metric_name,
        }
    if isinstance(conf_source, (str, Path)):
        with open(conf_source, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    if isinstance(conf_source, Mapping):
        return dict(conf_source)
    raise JudgeConfigError("Unsupported judge config source.")


def _normalize_scale(raw_scale: Mapping[Any, Any]) -> Dict[int, str]:
    if not isinstance(raw_scale, Mapping) or not raw_scale:
        raise JudgeConfigError("Each criterion must define a non-empty scale mapping.")
    normalized = {}
    for key, value in raw_scale.items():
        try:
            normalized_key = int(key)
        except (TypeError, ValueError) as exc:
            raise JudgeConfigError("Judge scale keys must be integers.") from exc
        normalized[normalized_key] = str(value).strip()
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def _get_required_str(raw_criterion: Mapping[str, Any], field_name: str) -> str:
    value = str(raw_criterion.get(field_name, "")).strip()
    if not value:
        raise JudgeConfigError(f"Criterion field '{field_name}' is required.")
    return value


def _validate_scale_keys(criteria: Iterable[JudgeCriterion]) -> Tuple[int, ...]:
    allowed_scores = None
    for criterion in criteria:
        criterion_scores = tuple(sorted(criterion.scale))
        if allowed_scores is None:
            allowed_scores = criterion_scores
            continue
        if criterion_scores != allowed_scores:
            raise JudgeConfigError(
                "All criteria in judge.yaml must use the same integer scale."
            )
    if allowed_scores is None:
        raise JudgeConfigError("At least one criterion is required.")
    return allowed_scores


def load_judge_config(conf_source: Any) -> JudgeConfig:
    raw_conf = _to_plain_mapping(conf_source)
    raw_criteria = raw_conf.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise JudgeConfigError("judge.yaml must contain a non-empty criteria list.")

    criteria = []
    seen_keys = set()
    for raw_criterion in raw_criteria:
        if not isinstance(raw_criterion, Mapping):
            raise JudgeConfigError("Each judge criterion must be a mapping.")
        key = _get_required_str(raw_criterion, "key")
        if key in seen_keys:
            raise JudgeConfigError(f"Duplicate judge criterion key: {key}")
        seen_keys.add(key)
        name = _get_required_str(raw_criterion, "name")
        scale = _normalize_scale(raw_criterion.get("scale", {}))
        criteria.append(JudgeCriterion(key=key, name=name, scale=scale))

    allowed_scores = _validate_scale_keys(criteria)
    metric_name = str(raw_conf.get("metric_name", "judge_avg")).strip() or "judge_avg"
    reference_field = (
        str(raw_conf.get("reference_field", "meta.reference_answer")).strip()
        or "meta.reference_answer"
    )
    prompt_template = raw_conf.get("prompt_template")
    if prompt_template is not None:
        prompt_template = str(prompt_template)

    return JudgeConfig(
        prompt_template=prompt_template,
        criteria=tuple(criteria),
        reference_field=reference_field,
        metric_name=metric_name,
        allowed_scores=allowed_scores,
    )


def resolve_reference_field(doc: Mapping[str, Any], dotted_path: str, default: str = "") -> str:
    value: Any = doc
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    if value is None:
        return default
    return str(value)


def format_scale(scale: Mapping[int, str]) -> str:
    return "\n".join(f"{score}: {description}" for score, description in sorted(scale.items()))
