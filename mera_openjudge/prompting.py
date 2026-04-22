from typing import Any, Dict, Mapping

from jinja2 import Environment, StrictUndefined

from mera_openjudge.config import JudgeConfig, JudgeCriterion, format_scale, resolve_reference_field


_PROMPT_ENV = Environment(undefined=StrictUndefined, autoescape=False)
_DEFAULT_PROMPT_TEMPLATE = _PROMPT_ENV.from_string(
    """
Вы оцениваете ответ модели на пользовательское задание.

Исходное задание:
{{ source_instruction }}

{% if reference_answer %}
Эталонный ответ:
{{ reference_answer }}

{% endif %}
Ответ модели:
{{ answer }}

Критерий оценки: {{ criterion_name }}
Шкала:
{{ criteria_rubrics }}

Верните ответ строго в формате:
Первая непустая строка: одно число из множества {{ allowed_scores_text }}
Далее: краткое объяснение оценки.
""".strip()
)


def render_source_instruction(doc: Mapping[str, Any]) -> str:
    instruction = str(doc.get("instruction", "")).strip()
    inputs = doc.get("inputs")
    if not instruction:
        return ""
    if inputs is None:
        return instruction
    try:
        if isinstance(inputs, Mapping):
            return instruction.format(**inputs).strip()
        return instruction.format(inputs=inputs).strip()
    except Exception:
        return instruction


def _build_context(
    doc: Mapping[str, Any],
    answer: str,
    criterion: JudgeCriterion,
    judge_config: JudgeConfig,
) -> Dict[str, Any]:
    reference_answer = resolve_reference_field(
        doc,
        judge_config.reference_field,
        default="",
    )
    return {
        "answer": answer,
        "allowed_scores": list(judge_config.allowed_scores),
        "allowed_scores_text": ", ".join(map(str, judge_config.allowed_scores)),
        "criterion": criterion,
        "criterion_key": criterion.key,
        "criterion_name": criterion.name,
        "criteria_name": criterion.name,
        "criteria_rubrics": format_scale(criterion.scale),
        "doc": doc,
        "instruction": render_source_instruction(doc),
        "reference_answer": reference_answer,
        "scale": criterion.scale,
        "source_instruction": render_source_instruction(doc),
    }


def render_judge_prompt(
    doc: Mapping[str, Any],
    answer: str,
    criterion: JudgeCriterion,
    judge_config: JudgeConfig,
) -> str:
    context = _build_context(doc=doc, answer=answer, criterion=criterion, judge_config=judge_config)
    if judge_config.prompt_template:
        return _PROMPT_ENV.from_string(judge_config.prompt_template).render(**context).strip()
    return _DEFAULT_PROMPT_TEMPLATE.render(**context).strip()
