"""Utility functions for MMReD MERA benchmark tasks.

This module provides:
- Answer extraction from model outputs (handling reasoning chains)
- Custom metric processing
- Image loading for multimodal evaluation
"""

import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Union

try:
    from lm_eval.api.filter import Filter
except ImportError:
    from abc import ABC, abstractmethod
    class Filter(ABC):
        def __init__(self, **kwargs): pass
        @abstractmethod
        def apply(self, resps, docs): return resps

try:
    from PIL import Image
except ImportError:
    Image = None


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
    
    # Common answer normalizations (Russian dataset is primary)
    normalizations = {
        # Russian canonical forms
        "кухня": "Кухня",
        "ванная": "Ванная",
        "сад": "Сад",
        "офис": "Офис",
        "спальня": "Спальня",
        "коридор": "Коридор",
        "сандра": "Сандра",
        "мария": "Мария",
        "иван": "Иван",
        "даниил": "Даниил",
        "михаил": "Михаил",
        "никто": "Никто",
        "никого": "Никто",
        # English fallbacks (map to Russian canonical)
        "kitchen": "Кухня",
        "bathroom": "Ванная",
        "garden": "Сад",
        "office": "Офис",
        "bedroom": "Спальня",
        "hallway": "Коридор",
        "sandra": "Сандра",
        "mary": "Мария",
        "john": "Иван",
        "daniel": "Даниил",
        "michael": "Михаил",
        "nobody": "Никто",
        "no one": "Никто",
        "none": "Никто",
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

    # Strip reasoning blocks: <think>...</think> and Gemma-4 <|channel>thought...<channel|>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Gemma-4 format: <|channel>thought...reasoning...<channel|>answer.
    # With interleaved thinking there are several thought/answer pairs —
    # the final answer follows the LAST channel close.
    gemma_matches = list(re.finditer(r"<channel\|>\s*(.+?)(?=<\|channel>|$)", text, flags=re.DOTALL))
    if gemma_matches:
        text = gemma_matches[-1].group(1)
    else:
        text = re.sub(r"<\|channel>thought.*?(?=<\|channel>|$)", "", text, flags=re.DOTALL)
    text = text.strip()

    if not text:
        return ""

    answer = None

    # Pattern 0: JSON {"answer": "value"} — most common structured output
    json_match = re.search(r'\{\s*["\']answer["\']\s*:\s*["\']?([^"\'}\s]+)["\']?\s*\}', text)
    if json_match:
        return normalize_answer(json_match.group(1), atype)

    # Also try parsing actual JSON
    json_block = re.search(r'\{[^{}]*"answer"[^{}]*\}', text)
    if json_block:
        try:
            obj = json.loads(json_block.group(0))
            if "answer" in obj:
                return normalize_answer(str(obj["answer"]), atype)
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 1: "The answer is X" / "Answer: X" / "Result: X".
    # \b — otherwise «ответы» in a reasoning chain matches and captures «ы»;
    # the separator is required (the task format is "Ответ: X");
    # the LAST match wins — CoT traces mention intermediate "ответ: ..." hypotheses.
    patterns = [
        r"(?:the\s+)?(?:answer|result)\b\s*(?:is|:)\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
        r"(?:ответ|результат)\b\s*(?::|—|–|-)\s*['\"]?([A-Za-zА-Яа-я0-9]+)['\"]?",
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1)
            break

    # Pattern 2: Boxed answer \boxed{X}
    if not answer:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            answer = match.group(1)

    # Pattern 3: Final answer marker **X** or *X* (last one — conclusions go last)
    if not answer:
        bold = list(re.finditer(r"\*\*([A-Za-zА-Яа-я0-9]+)\*\*", text))
        if bold:
            answer = bold[-1].group(1)
    
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


class extract_answer_filter(Filter):
    """Filter class for lm-evaluation-harness that extracts answers from model outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, resps: list, docs: list[dict]) -> list:
        extracted = []
        for resp, doc in zip(resps, docs):
            text = resp[0] if isinstance(resp, list) else resp
            meta = doc.get("meta", {})
            atype = meta.get("categories", {}).get("atype", meta.get("atype", "person"))
            answer = extract_answer(text, atype)
            extracted.append([answer])
        return extracted


# ============================================================================
# Metrics
# ============================================================================

def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Process results and compute metrics per task/length.
    
    Args:
        doc: Document dictionary with meta information
        results: List of model predictions
        
    Returns:
        Dictionary of metric name to value
    """
    meta = doc.get("meta", {})
    categories = meta.get("categories", {})
    task = categories.get("task_type", meta.get("task", "unknown"))
    seq_len = categories.get("seq_len", meta.get("seq_len", 0))
    atype = categories.get("atype", meta.get("atype", "person"))
    
    gold = normalize_answer(str(doc.get("outputs", "")), atype)
    pred = normalize_answer(str(results[0]) if results else "", atype)

    em = float(gold.lower() == pred.lower())

    metrics = {
        "exact_match": em,
        f"em.{task}": em,
        f"em.{task}.len{seq_len}": em,
        "em.dc_aggregate": em,  # For weighted aggregate
    }

    # Diagnostic regression metrics for numeric tasks (EM saturates to 0 on
    # long contexts and stops discriminating near-misses from random answers).
    # A non-numeric prediction gets the worst-case error so that abstaining
    # cannot improve MAE; numeric_parse_rate shows how often that happened.
    if atype == "number" and re.fullmatch(r"-?\d+", gold or ""):
        g = int(gold)
        scale = max(int(seq_len), 1)
        # Parse from the raw filtered response: normalize_answer maps
        # digit-free predictions to "0", which would mask parse failures
        raw_pred = str(results[0]) if results else ""
        if re.search(r"\d", raw_pred) and re.fullmatch(r"-?\d+", pred or ""):
            p = int(pred)
            err = abs(p - g)
            smape = 0.0 if (p == 0 and g == 0) else abs(p - g) / (abs(p) + abs(g))
            parsed = 1.0
        else:
            err = scale
            smape = 1.0
            parsed = 0.0
        metrics.update({
            "mae": float(err),
            "mae_norm": min(err / scale, 1.0),
            "smape": smape,
            "numeric_parse_rate": parsed,
            f"mae.{task}.len{seq_len}": float(err),
        })

    return metrics


def weighted_length_aggregate(items: list[dict]) -> float:
    """Aggregate metric with exponential weight on length in facts. 32 facts is base, 64 is 2x, 128 is 4x, etc.
    
    Args:
        items: List of result dictionaries
        
    Returns:
        Weighted average score
    """
    from collections import defaultdict

    # Group by task type
    task_scores = defaultdict(lambda: {"weighted_sum": 0.0, "total_weight": 0.0})

    for item in items:
        score = item.get("em.dc_aggregate", 0)
        seq_len = item.get("seq_len", 0)
        # Extract task type from metric keys
        task = "unknown"
        for k in item:
            if k.startswith("em.") and k != "em.dc_aggregate" and ".len" not in k:
                task = k.replace("em.", "")
                break

        weight = 2 ** (seq_len / 32)  # exponential: 2 for 32, 4 for 64, 16 for 128
        task_scores[task]["weighted_sum"] += score * weight
        task_scores[task]["total_weight"] += weight

    # Per-task length-weighted averages
    per_task = []
    for s in task_scores.values():
        if s["total_weight"] > 0:
            per_task.append(s["weighted_sum"] / s["total_weight"])

    if not per_task:
        return 0.0

    # Harmonic mean across task types (penalizes weakness on any task)
    eps = 1e-6
    n = len(per_task)
    return n / sum(1.0 / (v + eps) for v in per_task)


# ============================================================================
# Multimodal Support
# ============================================================================

def doc_to_image(doc: dict) -> list:
    """Load images for multimodal evaluation.
    
    Args:
        doc: Document dictionary with image paths in meta
        
    Returns:
        List of PIL Image objects
    """
    if Image is None:
        raise ImportError("PIL is required for multimodal evaluation. Install with: pip install Pillow")
    
    meta = doc.get("meta", {})
    image_paths = meta.get("images", [])
    
    if not image_paths:
        return []
    
    images = []
    for path in image_paths:
        # Handle both local paths and HuggingFace dataset paths
        if isinstance(path, str):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
        elif hasattr(path, "convert"):
            # Already a PIL Image
            images.append(path)
    
    return images


def doc_to_text_with_images(doc: dict) -> str:
    """Format document text with image placeholders.
    
    Args:
        doc: Document dictionary
        
    Returns:
        Formatted prompt text
    """
    instruction = doc.get("instruction", "")
    inputs = doc.get("inputs", {})
    
    # Format instruction with inputs
    try:
        text = instruction.format(**inputs)
    except (KeyError, ValueError):
        text = instruction
    
    return text.strip()
