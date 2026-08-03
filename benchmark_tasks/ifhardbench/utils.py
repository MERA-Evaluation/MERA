"""IFHardBench — deterministic instruction-following scoring for lm-evaluation-harness.

The benchmark scores *instruction following only* — never domain knowledge and
never the meaning of the answer. Every question ships a machine-readable list of
constraints in ``meta.constraints`` (a JSON string mirroring the human-readable
bullets that the model sees in ``inputs.constraints``). Each constraint is backed
by a pure, deterministic verifier with the uniform signature::

    verifier(response: str, params: dict, lang_code: str) -> bool

There is no LLM judge, no randomness, no I/O and no clocks, so a run is exactly
reproducible and a score can always be traced back to a specific unsatisfied
constraint.

Two metrics are emitted per question:

``sample_pass_rate``
    All-or-nothing: 1.0 iff *every* constraint of the question holds. This is the
    headline metric — real instruction following is not partial.
``constraint_pass_rate``
    Share of individually satisfied constraints; a partial-credit diagnostic that
    separates "ignored one requirement out of five" from "ignored all of them".

Because ``params`` in the dataset are already localised to the surface language
of the question at build time, the verifiers here need no constraint catalogue —
this module is fully self-contained.

Counting conventions
--------------------
Three primitives decide almost every verdict, so they are defined to match how a
person counts rather than how a tokeniser does:

* **word** — a whitespace-separated token containing at least one letter or
  digit. ``«чёрно-белый»`` is one word, ``«2025»`` is one word, a standalone dash
  is not a word. Nothing about the script matters, so an answer is not silently
  penalised for containing a number or a Latin name.
* **sentence** — text between terminators ``.``/``!``/``?``/``…`` that are
  followed by whitespace or the end of the answer. A lone ``.`` after a
  single-letter fragment does not terminate, so ``«и т.д. дальше»`` stays one
  sentence and ``«3.14»`` is not split. Trailing text with no terminator still
  counts as a sentence.
* **list item** — a line beginning with ``-``/``*``/``•`` or ``1.``/``1)``.

Word matching (required and forbidden words, positional words, acrostics) is
case-insensitive and treats ``ё`` as ``е``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

eval_logger = logging.getLogger(__name__)

DEFAULT_LANG = "ru"

#: Per-language alphabet, used where a verifier needs "a letter" specifically.
LANG: Dict[str, Dict[str, str]] = {
    "ru": {"alpha_re": r"[А-Яа-яЁё]"},
    "en": {"alpha_re": r"[A-Za-z]"},
}

_ALPHA_ANY = re.compile(r"[^\W\d_]", re.UNICODE)


# --------------------------------------------------------------------------- #
# Text primitives
# --------------------------------------------------------------------------- #

def _fold(text: str) -> str:
    """Case- and ё-insensitive folding used for every word comparison."""
    return text.lower().replace("ё", "е")


def get_words(text: str) -> List[str]:
    """Words as a person counts them: whitespace tokens carrying a letter or digit.

    Punctuation stays attached to the token (``«день,»`` is one word); tokens made
    only of punctuation (a dash used as a hyphen-minus, an ellipsis) are not words.
    """
    return [tok for tok in text.split() if any(ch.isalnum() for ch in tok)]


#: Punctuation stripped from the edges of a token before it is compared to a word.
_EDGE_PUNCT = " .,;:!?()[]{}<>*_`|/\\-" + "…«»„“—–\"'"


def word_core(token: str) -> str:
    """A token reduced to the form used for comparisons: no outer punctuation."""
    return _fold(token).strip(_EDGE_PUNCT)


def count_words(text: str) -> int:
    return len(get_words(text))


#: Terminator runs that are followed by whitespace or the end of the text.
_TERMINATOR_RE = re.compile(r"[.!?…]+(?=\s|$)")


def split_sentences(text: str) -> List[str]:
    """Split into sentences on ``.!?…`` that actually end a sentence.

    A single ``.`` is ignored when the letters immediately before it form a
    one-letter fragment — that is an abbreviation (``т.е.``, ``и т.д.``) or an
    initial, not a sentence boundary. Runs such as ``...`` or ``?!`` always
    terminate. Text after the last terminator is a sentence of its own.
    """
    pieces: List[str] = []
    start = 0
    for m in _TERMINATOR_RE.finditer(text):
        run = m.group(0)
        if run == ".":
            before = text[:m.start()]
            letters = 0
            for ch in reversed(before):
                if _ALPHA_ANY.match(ch):
                    letters += 1
                else:
                    break
            if letters < 2:
                continue
        pieces.append(text[start:m.end()])
        start = m.end()
    if text[start:].strip():
        pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Paragraphs separated by one or more blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


_BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s+(\S.*)$")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+(\S.*)$")


def split_list_items(text: str, kind: str = "any") -> List[str]:
    """Payloads of the list lines of ``text`` (the marker itself is stripped)."""
    out: List[str] = []
    for line in text.splitlines():
        if kind in ("any", "bullet"):
            m = _BULLET_LINE_RE.match(line)
            if m:
                out.append(m.group(1).strip())
                continue
        if kind in ("any", "numbered"):
            m = _NUMBERED_LINE_RE.match(line)
            if m:
                out.append(m.group(1).strip())
    return out


def first_letter(text: str) -> Optional[str]:
    """First alphabetic character of ``text``, folded; ``None`` if there is none."""
    m = _ALPHA_ANY.search(text)
    return _fold(m.group(0)) if m else None


def strip_final_punctuation(text: str) -> str:
    return text.strip().rstrip(".!?…\"'»)]}»„“ ").strip()


# --------------------------------------------------------------------------- #
# Count specifications
# --------------------------------------------------------------------------- #

def apply_op(actual: int, op: str, value: int) -> bool:
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "le":
        return actual <= value
    if op == "lt":
        return actual < value
    if op == "ge":
        return actual >= value
    if op == "gt":
        return actual > value
    raise ValueError(f"unknown op {op!r}")


def check_count(actual: int, params: dict) -> bool:
    """Evaluate a count against ``{op, value}`` or ``{op: between, min, max}``."""
    op = params.get("op", "eq")
    if op == "between":
        return int(params["min"]) <= actual <= int(params["max"])
    return apply_op(actual, op, int(params["value"]))


def _sub_spec(params: dict) -> dict:
    """The nested count spec that per-unit constraints carry under ``each``."""
    return params["each"] if "each" in params else params


# --------------------------------------------------------------------------- #
# STRUCTURE family — how much text, and how it is divided
# --------------------------------------------------------------------------- #

def v_word_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return check_count(count_words(response), params)


def v_sentence_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return check_count(len(split_sentences(response)), params)


def v_paragraph_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return check_count(len(split_paragraphs(response)), params)


def v_char_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Characters including spaces and punctuation, outer whitespace stripped."""
    return check_count(len(response.strip()), params)


def v_sentence_words(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every sentence individually satisfies the nested count spec."""
    sentences = split_sentences(response)
    if not sentences:
        return False
    spec = _sub_spec(params)
    return all(check_count(count_words(s), spec) for s in sentences)


def v_item_words(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every list item individually satisfies the nested count spec."""
    items = split_list_items(response, params.get("kind", "any"))
    if not items:
        return False
    spec = _sub_spec(params)
    return all(check_count(count_words(it), spec) for it in items)


def v_ascending_sentences(response: str, params: dict,
                          lang_code: str = DEFAULT_LANG) -> bool:
    """Each sentence is strictly longer than the one before it (in words).

    ``direction: "desc"`` reverses it. Two sentences are the minimum for the
    requirement to mean anything.
    """
    counts = [count_words(s) for s in split_sentences(response)]
    if len(counts) < 2:
        return False
    if params.get("direction") == "desc":
        return all(a > b for a, b in zip(counts, counts[1:]))
    return all(a < b for a, b in zip(counts, counts[1:]))


def v_no_repeated_word(response: str, params: dict,
                       lang_code: str = DEFAULT_LANG) -> bool:
    """No word of at least ``min_length`` letters occurs twice.

    Short words are exempt because banning a second ``и`` or ``на`` would be a
    requirement about Russian syntax rather than about following an instruction.
    """
    words = get_words(response)
    if not words:
        return False
    min_length = int(params.get("min_length", 4))
    seen = set()
    for token in words:
        core = word_core(token)
        if len(core) < min_length:
            continue
        if core in seen:
            return False
        seen.add(core)
    return True


def v_paragraph_sentences(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every paragraph individually satisfies the nested sentence-count spec."""
    paragraphs = split_paragraphs(response)
    if not paragraphs:
        return False
    spec = _sub_spec(params)
    return all(check_count(len(split_sentences(p)), spec) for p in paragraphs)


# --------------------------------------------------------------------------- #
# LEXICAL family — required and forbidden words, position, acrostic
# --------------------------------------------------------------------------- #

def _stems(params: dict, single_key: str, plural_key: str) -> List[str]:
    if plural_key in params:
        return [_fold(s) for s in params[plural_key]]
    return [_fold(params[single_key])]


def _has_stem(response: str, stem: str) -> bool:
    return any(word_core(tok).startswith(stem) for tok in get_words(response))


def v_include_keyword(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A word built on the given stem occurs — i.e. the word in any inflected form.

    ``params`` carries ``stem`` (prefix match, "the word in any form") or the
    legacy ``word`` (exact form).
    """
    if "stem" in params:
        return _has_stem(response, _fold(params["stem"]))
    target = _fold(params["word"])
    return any(word_core(tok) == target for tok in get_words(response))


def v_include_all(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every listed word occurs, each in any inflected form."""
    if "stems" in params:
        return all(_has_stem(response, s) for s in _stems(params, "stem", "stems"))
    targets = {_fold(w) for w in params["words"]}
    present = {word_core(tok) for tok in get_words(response)}
    return targets <= present


def v_exclude_word(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """None of the listed words occurs, in any inflected or derived form."""
    if "stems" in params:
        return not any(_has_stem(response, s) for s in _stems(params, "stem", "stems"))
    banned = {_fold(w) for w in params["words"]}
    return not any(word_core(tok) in banned for tok in get_words(response))


def v_starts_with(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return _fold(response.strip()).startswith(_fold(params["phrase"].strip()))


def v_ends_with(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return _fold(strip_final_punctuation(response)).endswith(
        _fold(strip_final_punctuation(params["phrase"]))
    )


def v_nth_word(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The word at position ``n`` (1-based) is the given word, in that exact form."""
    n = int(params["n"])
    words = get_words(response)
    if len(words) < n or n < 1:
        return False
    return word_core(words[n - 1]) == _fold(params["word"]).strip()


def v_acrostic(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """First letters of the list items spell the given word, in order."""
    target = _fold(params["word"])
    items = split_list_items(response, params.get("kind", "any"))
    if len(items) != len(target):
        return False
    initials = [first_letter(it) for it in items]
    return all(a is not None and a == b for a, b in zip(initials, target))


# --------------------------------------------------------------------------- #
# STYLE family — letters, case, punctuation
# --------------------------------------------------------------------------- #

def v_forbid_letter(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A lipogram: the given letter occurs nowhere, in either case."""
    return params["letter"].lower() not in response.lower()


def v_no_commas(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return "," not in response


def v_no_caps(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    return response == response.lower()


def v_end_punctuation(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    s = response.strip()
    return bool(s) and s[-1] == params["char"]


def v_sentence_initials(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every sentence starts with a different letter."""
    sentences = split_sentences(response)
    if len(sentences) < 2:
        return False
    initials = [first_letter(s) for s in sentences]
    if any(x is None for x in initials):
        return False
    return len(set(initials)) == len(initials)


# --------------------------------------------------------------------------- #
# FORMAT family — lists, JSON, tables
# --------------------------------------------------------------------------- #

def _list_only(response: str, matcher: re.Pattern) -> bool:
    """Every non-empty line of the response is a list line of the wanted kind."""
    lines = [ln for ln in response.splitlines() if ln.strip()]
    return bool(lines) and all(matcher.match(ln) for ln in lines)


def v_bullet_list(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A bulleted list of the required length.

    With ``strict`` (the shipped default) the whole answer must be the list —
    no lead-in line, no closing remark. The bullets that carry this constraint
    say so in words.
    """
    items = split_list_items(response, "bullet")
    if not check_count(len(items), params):
        return False
    if params.get("strict", True):
        return _list_only(response, _BULLET_LINE_RE)
    return True


def v_numbered_list(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    items = split_list_items(response, "numbered")
    if not check_count(len(items), params):
        return False
    if params.get("strict", True):
        return _list_only(response, _NUMBERED_LINE_RE)
    return True


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def v_json_object(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The whole answer is one JSON object with exactly the named keys.

    ``keys``      — the exact key set (no extras, none missing).
    ``max_words`` — optional cap on the word count of every string value.
    ``no_commas_in_values`` — optional ban on commas inside the values.
    """
    s = response.strip()
    m = _FENCE_RE.fullmatch(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    if set(obj) != set(params["keys"]):
        return False
    max_words = params.get("max_words")
    no_commas = params.get("no_commas_in_values", False)
    for value in obj.values():
        if not isinstance(value, str) or not value.strip():
            return False
        if max_words is not None and count_words(value) > int(max_words):
            return False
        if no_commas and "," in value:
            return False
    return True


_TABLE_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")


def v_markdown_table(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The whole answer is one markdown table: header, separator, ``rows`` data rows.

    Every row must be pipe-delimited and carry exactly ``cols`` non-empty cells.
    """
    lines = split_lines(response)
    if len(lines) != int(params["rows"]) + 2:
        return False
    cols = int(params["cols"])
    parsed = []
    for ln in lines:
        if not (ln.startswith("|") and ln.endswith("|")):
            return False
        cells = [c.strip() for c in ln[1:-1].split("|")]
        if len(cells) != cols:
            return False
        parsed.append(cells)
    if not all(_TABLE_SEP_CELL_RE.match(c) for c in parsed[1]):
        return False
    for row in parsed[:1] + parsed[2:]:
        if not all(c for c in row):
            return False
    return True


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

VERIFIERS: Dict[str, Callable[..., bool]] = {
    # structure — how much text and how it is divided
    "structure:word_count": v_word_count,
    "structure:sentence_count": v_sentence_count,
    "structure:paragraph_count": v_paragraph_count,
    "structure:char_count": v_char_count,
    "structure:sentence_words": v_sentence_words,
    "structure:item_words": v_item_words,
    "structure:paragraph_sentences": v_paragraph_sentences,
    "structure:ascending_sentences": v_ascending_sentences,
    "lexical:no_repeated_word": v_no_repeated_word,
    # lexical — required and forbidden words, position, acrostic
    "lexical:include_keyword": v_include_keyword,
    "lexical:include_all": v_include_all,
    "lexical:exclude_word": v_exclude_word,
    "lexical:starts_with": v_starts_with,
    "lexical:ends_with": v_ends_with,
    "lexical:nth_word": v_nth_word,
    "lexical:acrostic": v_acrostic,
    # style — letters, case, punctuation
    "style:forbid_letter": v_forbid_letter,
    "style:no_commas": v_no_commas,
    "style:no_caps": v_no_caps,
    "style:end_punctuation": v_end_punctuation,
    "style:sentence_initials": v_sentence_initials,
    # format — lists, JSON, tables
    "format:bullet_list": v_bullet_list,
    "format:numbered_list": v_numbered_list,
    "format:json_object": v_json_object,
    "format:markdown_table": v_markdown_table,
}


def verify(category: str, response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Dispatch to the registered verifier for ``category``."""
    fn = VERIFIERS.get(category)
    if fn is None:
        raise KeyError(f"no verifier for constraint category {category!r}")
    return fn(response, params, lang_code)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

_PROMPTS_CACHE: Optional[List[str]] = None
_META_CANDIDATES = (
    "../../datasets/IFHardBench/dataset_meta.json",
    "../../../datasets/IFHardBench/dataset_meta.json",
)


def _load_prompts() -> List[str]:
    """Prompt templates from ``dataset_meta.json`` (repo-local runs only).

    On the Hugging Face copy of the dataset ``instruction`` already holds the
    prompt text, so this is never called. It is a convenience for running the
    task straight off the JSON files in ``datasets/IFHardBench/``, where
    ``instruction`` is still an index into ``dataset_meta.json["prompts"]``.
    """
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    for rel in _META_CANDIDATES:
        path = os.path.normpath(os.path.join(here, rel))
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                _PROMPTS_CACHE = json.load(fh)["prompts"]
            return _PROMPTS_CACHE
    raise FileNotFoundError(
        "instruction is an index but dataset_meta.json was not found; "
        "expected one of: " + ", ".join(_META_CANDIDATES)
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Build the request body: substitute ``inputs`` into the instruction template.

    ``inputs.context`` is an empty string for the short half of the dataset; those
    questions use prompt templates that carry no ``{context}`` placeholder, so the
    empty value is never rendered.
    """
    instruction = doc["instruction"]
    if isinstance(instruction, int):  # local JSON files store a prompt index
        instruction = _load_prompts()[instruction]
    return instruction.format(**doc["inputs"])


def _resolve_instruction(doc: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(doc.get("instruction"), int):
        doc["instruction"] = _load_prompts()[doc["instruction"]]
    return doc


def process_docs(dataset):
    """Replace prompt indices with prompt text (local ``datasets/`` JSON only).

    Needed because the MERA-format JSON files store ``instruction`` as an index
    into ``dataset_meta.json["prompts"]`` while the Hugging Face copy stores the
    prompt itself. Resolving up front — rather than inside ``doc_to_text`` — keeps
    the few-shot path working too: :class:`FewshotSampler` swaps ``doc_to_text``
    for the Jinja ``query`` template, which calls ``instruction.format`` directly.
    """
    return dataset.map(_resolve_instruction)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def normalize_generation(generation: str) -> str:
    """Strip reasoning traces, then outer whitespace — nothing else.

    Reasoning blocks are scaffolding, not part of the answer, so leaving them in
    would fail every constraint for a reasoning model regardless of how well it
    followed the instruction. Beyond that the response is scored verbatim: any
    further "lenient extraction" (peeling off a preamble, unwrapping code fences)
    would score a response the instruction did not ask for.
    """
    text = _THINK_BLOCK_RE.sub("", generation or "")
    text = _THINK_TAG_RE.sub("", text)
    return text.strip()


def _extract_prediction(results: List[Any]) -> str:
    if not results:
        return ""
    value = results[0]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return value if isinstance(value, str) else str(value)


def get_constraints(doc: Dict[str, Any]) -> List[dict]:
    """Machine-readable constraints of a question (JSON string in ``meta``)."""
    raw = doc["meta"]["constraints"]
    return json.loads(raw) if isinstance(raw, str) else raw


def score_response(doc: Dict[str, Any], response: str) -> List[dict]:
    """One ``{category, family, pass}`` record per constraint of the question."""
    lang = (doc["meta"].get("categories") or {}).get("language", DEFAULT_LANG)
    out: List[dict] = []
    for c in get_constraints(doc):
        try:
            ok = bool(verify(c["category"], response, c["params"], lang))
        except Exception as exc:  # noqa: BLE001 — a broken verifier must not abort a run
            eval_logger.warning(
                "verifier %s failed on sample %s: %s",
                c["category"], doc["meta"].get("id"), exc,
            )
            ok = False
        out.append({"category": c["category"], "family": c["family"], "pass": ok})
    return out


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Score one generation against the question's own constraint stack.

    An empty response never passes: prohibition-style constraints (forbidden
    word, no commas, upper-bounded counts, ...) are vacuously satisfied by the
    empty string, so without this guard a model that answered nothing would
    "pass" every question whose stack happens to contain only such constraints.
    """
    response = normalize_generation(_extract_prediction(results))
    checks = score_response(doc, response)
    n_total = len(checks)
    if n_total == 0:
        eval_logger.warning("sample %s has no constraints", doc["meta"].get("id"))
        return {"sample_pass_rate": 0.0, "constraint_pass_rate": 0.0}

    n_pass = sum(1 for c in checks if c["pass"])
    is_empty = not response.strip()
    return {
        "sample_pass_rate": float(n_pass == n_total and not is_empty),
        "constraint_pass_rate": (0.0 if is_empty else n_pass / n_total),
    }
