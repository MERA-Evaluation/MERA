"""LLM-as-a-judge scoring shared by the generative tasks.

Every task that reports ``judge_score`` sends the judge the same prompt: the
task text, the reference answer, the answer under evaluation, and one criterion
with its rubric. The prompt lives here rather than in a copy per task — that is
what makes the metric comparable across tasks, and it means the rubric is
edited in one place.

The judge answers on the rubric's own 0-2 scale. :func:`compute_judge_score`
divides that by :data:`JUDGE_MAX_SCORE`, so the number a task reports is in
[0, 1]: 2 -> 1.0 (the answer matches the reference), 1 -> 0.5 (the answer is
incomplete), 0 -> 0.0 (the answer is wrong or missing). That is the scale the
dataset cards document.

Tasks use it like this::

    from mera_judge import compute_judge_score

    judge_score = best_score(lambda c: compute_judge_score(doc, c), candidates)

See any of the task ``utils.py`` files for the import shim: lm-eval executes
those by path, so ``benchmark_tasks`` is not importable by name from them.
"""

import logging
import os

eval_logger = logging.getLogger(__name__)


CRITERIA_NAME = "Правильность ответа"

CRITERIA_RUBRICS = """0: Дан неправильный ответ или ответ отсутствует.

1: Ответ модели неполный (не на все вопросы задания получен ответ, в формулировке ответа отсутствует часть информации).

2: Ответ модели совпадает с эталонным или эквивалентен ему."""

JUDGE_PROMPT = """### Задание для оценки:
{instruction}

### Эталонный ответ:
{reference_answer}

### Ответ для оценки:
{answer}

### Критерий оценки:
{criteria_name}

### Шкала оценивания по критерию:
{criteria_rubrics}"""

# The top of the rubric: the judge's score is divided by it to give [0, 1].
JUDGE_MAX_SCORE = 2.0


def task_instruction(doc):
    """The task text as the model saw it."""
    instruction = doc.get("instruction") or ""
    try:
        return instruction.format(**(doc.get("inputs") or {}))
    except (KeyError, IndexError, ValueError):
        return instruction


def build_judge_prompt(instruction, template=None):
    """The judge prompt with everything but the two answers filled in.

    ``{reference}`` and ``{prediction}`` are left in place for
    ``compute_llm_judge``, which substitutes them itself. Braces coming from the
    task text are escaped so that substitution cannot read them as placeholders,
    and the task text is filled in last, so none of the replacements above can
    reach inside it.
    """
    template = JUDGE_PROMPT if template is None else template
    return (
        template.replace("{reference_answer}", "{reference}")
        .replace("{answer}", "{prediction}")
        .replace("{criteria_name}", CRITERIA_NAME)
        .replace("{criteria_rubrics}", CRITERIA_RUBRICS)
        .replace("{instruction}", instruction.replace("{", "{{").replace("}", "}}"))
    )


def normalize_score(raw):
    """A rubric score into [0, 1], clamped in case the judge leaves the scale."""
    return min(max(float(raw) / JUDGE_MAX_SCORE, 0.0), 1.0)


def compute_judge_score(doc, model_answer, instruction=None, reference=None):
    """The judge's score for one answer against one reference answer, in [0, 1].

    Scores 0.0 without calling anything when there is no reference answer or the
    judge is not configured (``LM_EVAL_JUDGE_API_BASE`` and
    ``LM_EVAL_JUDGE_MODEL``), and also when the judge call itself fails — a run
    goes on without the metric rather than dying on a network error.

    ``instruction`` defaults to the task text rendered from the document; pass
    it explicitly if a task shows the model something else. ``reference``
    defaults to the document's answer; a task whose examples accept several
    answers passes them one at a time, so that the judge is never asked to
    match a list. The whole prompt can be replaced through
    ``LM_EVAL_JUDGE_PROMPT``.
    """
    judge_api_base = os.getenv("LM_EVAL_JUDGE_API_BASE")
    judge_model = os.getenv("LM_EVAL_JUDGE_MODEL")

    if reference is None:
        reference = doc.get("outputs") or ""

    if not reference or not judge_api_base or not judge_model:
        return 0.0

    if instruction is None:
        instruction = task_instruction(doc)

    try:
        from lm_eval.api.metrics_generative import compute_llm_judge

        raw = compute_llm_judge(
            [model_answer],
            [reference],
            api_base=judge_api_base,
            model=judge_model,
            judge_prompt=build_judge_prompt(
                instruction, os.getenv("LM_EVAL_JUDGE_PROMPT") or None
            ),
        )["llm_judge"]
        # Inside the try: compute_llm_judge raises when the judge answers with
        # something it cannot read as a score, and that is a judge failure like
        # any other.
        return normalize_score(raw)
    except Exception:
        eval_logger.warning("%s: llm judge failed, scoring 0.0", __name__, exc_info=True)
        return 0.0
