import re
from typing import Iterable

from mera_openjudge.exceptions import JudgeParseError


_INTEGER_RE = re.compile(r"-?\d+")


def _extract_allowed_scores(text: str, allowed_scores) -> Iterable[int]:
    allowed_score_set = set(allowed_scores)
    for match in _INTEGER_RE.finditer(text):
        score = int(match.group(0))
        if score in allowed_score_set:
            yield score


def parse_judge_score(text: str, allowed_scores) -> int:
    if text is None:
        raise JudgeParseError("Judge response is empty.")

    normalized_text = str(text).strip()
    if not normalized_text:
        raise JudgeParseError("Judge response is empty.")

    first_non_empty_line = ""
    for line in normalized_text.splitlines():
        stripped = line.strip()
        if stripped:
            first_non_empty_line = stripped
            break

    if first_non_empty_line:
        for score in _extract_allowed_scores(first_non_empty_line, allowed_scores):
            return score

    for score in _extract_allowed_scores(normalized_text, allowed_scores):
        return score

    raise JudgeParseError(
        "Failed to parse judge score from response.",
        response_text=normalized_text,
    )
