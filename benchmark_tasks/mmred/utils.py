"""Utility functions for MMReD MERA benchmark tasks.

This module provides:
- Answer extraction from model outputs (handling reasoning chains)
- Custom metric processing
"""

import re
from typing import Dict, List


# ============================================================================
# Answer Extraction
# ============================================================================

def normalize_answer(answer: str, atype: str) -> str:
    """Normalize answer based on expected type.

    Args:
        answer: Raw answer string
        atype: Answer type ("person", "room", or "number")

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    answer = answer.strip().rstrip(".,!?;:")

    if atype == "number":
        # Extract digits only
        digits = re.sub(r"[^\d]", "", answer)
        return digits if digits else "0"

    # For person/room: capitalize first letter, handle common variations
    answer = answer.lower().strip()

    # Common answer normalizations
    normalizations = {
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "garden": "Garden",
        "office": "Office",
        "bedroom": "Bedroom",
        "hallway": "Hallway",
        "sandra": "Sandra",
        "mary": "Mary",
        "john": "John",
        "daniel": "Daniel",
        "michael": "Michael",
        "nobody": "Nobody",
        "no one": "Nobody",
        "none": "Nobody",
        # Russian normalizations
        "кухня": "Kitchen",
        "ванная": "Bathroom",
        "сад": "Garden",
        "офис": "Office",
        "спальня": "Bedroom",
        "коридор": "Hallway",
        "сандра": "Sandra",
        "мария": "Mary",
        "иван": "John",
        "даниил": "Daniel",
        "михаил": "Michael",
        "никто": "Nobody",
    }

    return normalizations.get(answer, answer.capitalize())


def extract_answer(text: str, atype: str = "person") -> str:
    """Extract answer from model generation, handling reasoning chains.

    Supports multiple extraction patterns for reasoning models:
    - "The answer is X"
    - "\\boxed{X}"
    - Last word/number in output

    Args:
        text: Model generated text
        atype: Expected answer type

    Returns:
        Extracted and normalized answer
    """
    if not text:
        return ""

    text = text.strip()
    answer = None

    # Pattern 1: "The answer is X" / "Answer: X" / "Result: X"
    patterns = [
        r"(?:the\s+)?(?:answer|result)\s*(?:is|:)\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
        r"(?:ответ|результат)\s*(?::|—|-)?\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1)
            break

    # Pattern 2: Boxed answer \boxed{X}
    if not answer:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            answer = match.group(1)

    # Pattern 3: Final answer marker **X** or *X*
    if not answer:
        match = re.search(r"\*\*([A-Za-zА-Яа-я0-9]+)\*\*", text)
        if match:
            answer = match.group(1)

    # Pattern 4: First word/number for short responses
    if not answer:
        if atype == "number":
            match = re.search(r"\d+", text)
            answer = match.group(0) if match else ""
        else:
            # Get first valid word (not common reasoning words)
            words = text.split()
            skip_words = {"the", "a", "an", "i", "think", "believe", "so", "therefore"}
            for word in words:
                clean = re.sub(r"[^\w]", "", word.lower())
                if clean and clean not in skip_words:
                    answer = word
                    break
            if not answer and words:
                answer = words[0]

    return normalize_answer(answer or "", atype)


def extract_answer_filter(resps, docs):
    """Filter function for lm-evaluation-harness.

    Extracts answers from model responses, using the atype from
    meta.categories if available, falling back to meta.atype.

    Args:
        resps: List of model responses (each is a list with one response)
        docs: List of document dictionaries

    Returns:
        List of extracted answer strings
    """
    extracted = []
    for resp, doc in zip(resps, docs):
        text = resp[0] if isinstance(resp, list) else resp
        meta = doc.get("meta", {})
        categories = meta.get("categories", {})
        atype = categories.get("atype", meta.get("atype", "person"))
        answer = extract_answer(text, atype)
        extracted.append([answer])  # Keep as list for pipeline
    return extracted


# ============================================================================
# Metrics
# ============================================================================

def process_results(doc: dict, results: list) -> dict:
    """Process results and compute metrics per task/length.

    Args:
        doc: Document dictionary with meta information
        results: List of model predictions

    Returns:
        Dictionary of metric name to value
    """
    meta = doc.get("meta", {})
    categories = meta.get("categories", {})
    task_type = categories.get("task_type", meta.get("task", "unknown"))
    seq_len = categories.get("seq_len", meta.get("seq_len", 0))
    atype = categories.get("atype", meta.get("atype", "person"))

    # Normalise task_type to lowercase underscore form for metric keys
    # e.g. "DC-SA-C" -> "dc_sa_c"
    task_key = task_type.lower().replace("-", "_")

    gold = normalize_answer(str(doc.get("outputs", "")), atype)
    pred = normalize_answer(str(results[0]) if results else "", atype)

    em = float(gold.lower() == pred.lower())

    return {
        "exact_match": em,
        f"em.{task_key}": em,
        f"em.{task_key}.len{seq_len}": em,
        "em.dc_aggregate": em,  # For weighted aggregate
    }


def weighted_length_aggregate(items: list) -> float:
    """Aggregate metric with exponential weight on sequence length.

    Weight schedule: 32 steps → 1×, 64 steps → 2×, 128 steps → 4×.

    Args:
        items: List of result dictionaries

    Returns:
        Weighted average score
    """
    total_weight = 0
    weighted_sum = 0

    for item in items:
        score = item.get("em.dc_aggregate", 0)
        seq_len = item.get("seq_len", 32)
        weight = 2 ** (seq_len // 32 - 1)

        weighted_sum += score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0
