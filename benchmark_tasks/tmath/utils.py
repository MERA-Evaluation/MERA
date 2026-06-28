import re
import logging
from typing import Any, Dict, List

eval_logger = logging.getLogger(__name__)

_MATH_VERIFY_ERROR: str | None = None

try:
    from math_verify import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse,
        verify,
    )
    from latex2sympy2_extended import NormalizationConfig
except ImportError as exc:
    parse = None
    verify = None
    _MATH_VERIFY_ERROR = (
        "math_verify is required to score T-math.\n"
        "Install with:\n"
        "  pip install math_verify latex2sympy2_extended 'antlr4-python3-runtime==4.11'"
    )
    eval_logger.warning("%s\nOriginal error: %s", _MATH_VERIFY_ERROR, exc)


def _require_math_verify() -> None:
    if parse is None or verify is None:
        raise RuntimeError(_MATH_VERIFY_ERROR or "math_verify is not available")


def _extract_prediction(results: List[Any]) -> str:
    if not results:
        return ""
    value = results[0]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return value if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def doc_to_text(doc: Dict[str, Any]) -> str:
    """Builds the request body: substitutes inputs into the instruction template.

    The instruction field already contains the full instruction with the
    {question} placeholder, and inputs is a dict {"question": "..."}.
    """
    return doc["instruction"].format(**doc["inputs"])


# ---------------------------------------------------------------------------
# Preprocessing: extraction and parsing of mathematical expressions
# ---------------------------------------------------------------------------

def _fix_decimal_comma(text: str) -> str:
    """Replaces a decimal comma between digits with a dot ('0,75' -> '0.75').

    A frequent case in Russian-language generations; without this fix
    latex2sympy parses '0,75' incorrectly (as an enumeration, not a number).
    """
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def preprocess_gold(gold: str) -> list:
    """Parses the gold answer ('7', '1/2', '0.75', '183+1839-8')
    into a list of sympy representations."""
    return parse(_fix_decimal_comma(gold), extraction_mode="first_match")


def preprocess_prediction(generation: str) -> list:
    """Extracts the final answer from the raw model generation and parses it.

    Extractor order:
      1) LatexExtractionConfig (boxed="all") -- prioritizes the contents of
         \\boxed{...}, then other LaTeX expressions;
      2) ExprExtractionConfig -- fallback to plain expressions/numbers in the
         text in case the model did not use LaTeX ('Answer: 33').
    """
    return parse(
        _fix_decimal_comma(generation),
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                )
            ),
            ExprExtractionConfig(),
        ],
        extraction_mode="first_match",
    )


# ---------------------------------------------------------------------------
# Metric: EM at the level of mathematical equivalence
# ---------------------------------------------------------------------------

def metric_exact_match(gold_parsed: list, answer_parsed: list) -> float:
    """EM based on symbolic equivalence of sympy expressions:
    1/2 == 0.5 == \\frac{1}{2}; 183+1839-8 == 2014.

    verify() is antisymmetric in its arguments -- gold is passed first.
    """
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception as exc:  # noqa: BLE001
        eval_logger.warning(
            "math_verify failed: %s, answer: %s, gold: %s",
            exc, answer_parsed, gold_parsed,
        )
        return 0.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Scoring following the methodology of the original T-math (t-tech/T-math):
    preprocessing (extraction + parsing) of both sides, then EM at the level
    of mathematical equivalence. With greedy decoding, the averaged
    exact_match is equivalent to pass@1 from the dataset card.
    """
    _require_math_verify()

    gold = doc["outputs"]
    if not gold:
        # closed test set without answers (--inference): honest zero
        return {"exact_match": 0.0}

    prediction = _extract_prediction(results)
    if not prediction.strip():
        return {"exact_match": 0.0}

    gold_parsed = preprocess_gold(gold)
    answer_parsed = preprocess_prediction(prediction)

    return {"exact_match": metric_exact_match(gold_parsed, answer_parsed)}
