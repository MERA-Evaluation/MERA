import os
import re
from functools import lru_cache
from string import Formatter
from typing import Any, Dict, Iterable, List, Optional, Set


METRIC_NAME = "llm_as_judge"

POLLUX_JUDGE_PROMPT = """### Задание для оценки:
{instruction}

### Эталонный ответ:
{reference_answer}

### Ответ для оценки:
{answer}

### Критерий оценки:
{criteria_name}

### Шкала оценивания по критерию:
{criteria_rubrics}
"""


def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    answer = results[0] if results else ""
    criteria = _get_criteria(doc)
    scores = judge_answer_by_criteria(
        instruction=_render_instruction(doc),
        answer=answer,
        reference_answer=str(doc.get("reference_answer", "")),
        criteria=criteria,
    )
    normalized_scores = [
        score / get_criterion_score_max(criterion)
        for score, criterion in zip(scores, criteria)
    ]
    return {METRIC_NAME: sum(normalized_scores) / len(normalized_scores)}


def _render_instruction(doc: Dict[str, Any]) -> str:
    instruction = str(doc.get("instruction", ""))
    inputs = doc.get("inputs", "")
    if isinstance(inputs, dict):
        return instruction.format(**inputs).strip()
    return instruction.format(inputs=inputs).strip()


def _get_criteria(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    criteria = doc.get("criteria")
    if criteria is None:
        criteria = doc.get("meta", {}).get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("POLLUX judge scoring requires a non-empty `criteria` list.")
    return criteria


def judge_answer_by_criteria(
    instruction: str,
    answer: str,
    reference_answer: str,
    criteria: Iterable[Dict[str, Any]],
) -> List[float]:
    return [
        judge_answer_by_criterion(
            instruction=instruction,
            answer=answer,
            reference_answer=reference_answer,
            criterion=criterion,
        )
        for criterion in criteria
    ]


def _judge_answer_with_prompt(
    instruction: str,
    answer: str,
    reference_answer: str,
    criterion: Optional[Dict[str, Any]],
    prompt_template: str,
    allowed_scores: Set[int],
) -> float:
    client = _get_openai_client()
    model = _get_required_env("POLLUX_JUDGE_MODEL")
    temperature = float(os.getenv("POLLUX_JUDGE_TEMPERATURE", "0"))
    timeout = float(os.getenv("POLLUX_JUDGE_TIMEOUT", "60"))
    max_tokens = int(os.getenv("POLLUX_JUDGE_MAX_TOKENS", "512"))
    criterion = criterion or {}
    prompt = _format_judge_prompt(
        prompt_template,
        instruction=instruction,
        answer=answer,
        reference_answer=reference_answer,
        criteria_name=str(criterion.get("criteria_name", "")),
        criteria_rubrics=str(criterion.get("rubrics", "")),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    content = response.choices[0].message.content
    score = parse_judge_score(content)
    if score not in allowed_scores:
        raise ValueError(
            f"POLLUX judge returned score {score}, expected one of "
            f"{sorted(allowed_scores)}."
        )
    return score


def judge_answer_by_criterion(
    instruction: str,
    answer: str,
    reference_answer: str,
    criterion: Dict[str, Any],
) -> float:
    allowed_scores = parse_rubric_scores(str(criterion.get("rubrics", "")))
    return _judge_answer_with_prompt(
        instruction=instruction,
        answer=answer,
        reference_answer=reference_answer,
        criterion=criterion,
        prompt_template=os.getenv("POLLUX_JUDGE_PROMPT", POLLUX_JUDGE_PROMPT),
        allowed_scores=allowed_scores,
    )


def parse_rubric_scores(rubrics: str) -> Set[int]:
    scores = set()
    for line in str(rubrics or "").splitlines():
        match = re.match(r"^[ \t]*([+-]?\d+)[ \t]*:", line)
        if match:
            scores.add(int(match.group(1)))
    if not scores:
        raise ValueError("POLLUX criterion rubrics do not contain a numeric score scale.")
    return scores


def get_criterion_score_max(criterion: Dict[str, Any]) -> int:
    scores = parse_rubric_scores(str(criterion.get("rubrics", "")))
    score_max = max(scores)
    if score_max <= 0:
        raise ValueError(
            "POLLUX criterion score scale must have a positive maximum for "
            "normalization."
        )
    return score_max


@lru_cache(maxsize=1)
def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "POLLUX LLM-as-a-Judge requires the `openai` package. "
            "Install it with `pip install openai`."
        ) from exc

    base_url = _get_required_env("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    timeout = float(os.getenv("POLLUX_JUDGE_TIMEOUT", "60"))
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} must be set for POLLUX LLM-as-a-Judge scoring.")
    return value


def _format_judge_prompt(
    template: str,
    instruction: str,
    answer: str,
    reference_answer: str = "",
    criteria_name: str = "",
    criteria_rubrics: str = "",
) -> str:
    values = {
        "instruction": instruction,
        "answer": answer,
        "reference": instruction,
        "prediction": answer,
        "reference_answer": reference_answer,
        "criteria_name": criteria_name,
        "criteria_rubrics": criteria_rubrics,
    }
    return _format_prompt_with_values(template, values)


def _format_prompt_with_values(template: str, values: Dict[str, str]) -> str:
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    unknown = field_names - set(values)
    if unknown:
        raise ValueError(
            "POLLUX_JUDGE_PROMPT contains unsupported placeholders: "
            + ", ".join(sorted(unknown))
        )
    return template.format(**values)


def parse_judge_score(content: str) -> float:
    cleaned = _strip_json_fence(str(content or "").strip())
    return _parse_plain_score(cleaned, content)


def _parse_plain_score(cleaned: str, original_content: str) -> float:
    normalized = cleaned.replace(",", ".")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return float(normalized)

    first_line = normalized.splitlines()[0].strip() if normalized.splitlines() else ""
    if re.fullmatch(r"-?\d+(?:\.\d+)?", first_line):
        return float(first_line)

    match = re.fullmatch(
        r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        first_line,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    return 0.0


def _strip_json_fence(content: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return content
