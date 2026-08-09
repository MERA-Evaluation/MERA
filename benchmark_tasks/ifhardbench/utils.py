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

Three metrics are emitted per question:

``sample_pass_rate``
    All-or-nothing: 1.0 iff *every* constraint of the question holds. The
    headline metric — real instruction following is not partial. A mean over
    rows, so it answers "how often is the whole answer right".
``balance_score``
    The second headline, and the one that asks a different question: "is there a
    kind of instruction this model cannot do at all". Per-requirement-type pass
    rates, combined as a geometric mean with equal weight per type and floored
    at :data:`BALANCE_FLOOR`. Both other metrics average over occurrences, so a
    rare requirement the model always fails costs them almost nothing; here it
    costs about seven per cent of the score. See :func:`agg_balance_score`.
``constraint_pass_rate``
    Share of individually satisfied constraints; a partial-credit diagnostic that
    separates "ignored one requirement out of five" from "ignored all of them".
    **Declared over non-empty responses only**: an empty response — a reasoning
    model that spent its whole token budget on the trace — already fails
    ``sample_pass_rate`` outright, and zeroing five constraint verdicts for it
    would make this diagnostic measure token budgeting instead of what it exists
    to measure, the comprehensibility of the requirements. Per-sample the metric
    is ``None`` for an empty response; the aggregation
    (:func:`agg_constraint_pass_rate`) averages the rest. Because the two
    metrics treat empty responses differently, the share of empty responses
    belongs next to them wherever they are quoted.

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

import csv
import io
import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
try:
    from lm_eval.api.registry import register_filter, FILTER_REGISTRY
    from lm_eval.api.filter import Filter
except ImportError:  # pragma: no cover — see the note below
    # This module is the single definition of what every requirement means, and
    # it is imported in two very different places: by lm-eval when the benchmark
    # runs, and by plain Python when the dataset is built or checked (the
    # generator scores its own witnesses with it, and the upload notebook
    # re-scores every one of them before pushing to the Hub). Only the response
    # filter at the bottom of this file needs lm-eval; the verifiers and the
    # metrics do not. A hard import would make the build depend on the harness
    # for a class the build never uses.
    register_filter = FILTER_REGISTRY = Filter = None

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

    ``params`` carries ``stem`` (prefix match, "the word in any form"), the
    legacy ``word`` (exact form), or ``any_stems``: a list of spellings any one
    of which satisfies the requirement. The last one exists for the
    transliteration types, where «вай-фай» and «вайфай» are the same answer and
    the wording says so.
    """
    if "any_stems" in params:
        return any(_has_stem(response, _fold(s)) for s in params["any_stems"])
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


def _parse_json_payload(response: str):
    """The response as parsed JSON, or ``None``. A ``` fence is tolerated."""
    s = response.strip()
    m = _FENCE_RE.fullmatch(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def v_json_list_value(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A JSON object where one named value is itself a markdown bullet list.

    ``keys``       — the exact key set;
    ``list_key``   — the key whose string value must be a bullet list of
                     exactly ``list_items`` items — real newlines escaped as
                     ``\\n`` inside the JSON string, every line a bullet;
    ``max_words``  — cap on the word count of every *other* value.
    """
    obj = _parse_json_payload(response)
    if not isinstance(obj, dict) or set(obj) != set(params["keys"]):
        return False
    list_key = params["list_key"]
    for key, value in obj.items():
        if not isinstance(value, str) or not value.strip():
            return False
        if key == list_key:
            lines = [ln for ln in value.splitlines() if ln.strip()]
            items = split_list_items(value, "bullet")
            if len(items) != int(params["list_items"]) or len(lines) != len(items):
                return False
        elif count_words(value) > int(params["max_words"]):
            return False
    return True


def v_json_array(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A JSON array of exactly ``items`` objects ``{"пункт": str, "слов": int}``
    where every ``слов`` equals the actual word count of its ``пункт``.

    The count is self-verifying: the model has to commit to a number the
    scorer will recount.
    """
    arr = _parse_json_payload(response)
    if not isinstance(arr, list) or len(arr) != int(params["items"]):
        return False
    for entry in arr:
        if not isinstance(entry, dict) or set(entry) != {"пункт", "слов"}:
            return False
        text, stated = entry["пункт"], entry["слов"]
        if not isinstance(text, str) or not text.strip():
            return False
        if isinstance(stated, bool) or not isinstance(stated, int):
            return False
        if stated != count_words(text):
            return False
    return True


def v_csv_line(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """One CSV record: exactly ``fields`` comma-separated fields, all non-empty,
    exactly one of which contains a comma of its own — so that field has to be
    double-quoted for the record to parse.
    """
    s = response.strip()
    if "\n" in s or '"' not in s:
        return False
    try:
        rows = list(csv.reader(io.StringIO(s), skipinitialspace=True))
    except Exception:
        return False
    if len(rows) != 1:
        return False
    row = rows[0]
    if len(row) != int(params["fields"]):
        return False
    if not all(cell.strip() for cell in row):
        return False
    return sum("," in cell for cell in row) == 1


def v_two_parts(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A bullet list, a blank line, then exactly one closing sentence.

    ``items``          — length of the list;
    ``sentence_words`` — exact word count of the closing sentence.
    """
    parts = re.split(r"\n\s*\n", response.strip(), maxsplit=1)
    if len(parts) != 2:
        return False
    head, tail = parts
    head_lines = [ln for ln in head.splitlines() if ln.strip()]
    items = split_list_items(head, "bullet")
    if len(items) != int(params["items"]) or len(head_lines) != len(items):
        return False
    tail = tail.strip()
    if not tail or split_list_items(tail, "any"):
        return False
    if len(split_sentences(tail)) != 1:
        return False
    return count_words(tail) == int(params["sentence_words"])


def v_items_total_words(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The whole answer holds exactly ``factor`` times as many words as it has
    list items — the two counters are tied to each other, not to a constant.
    """
    items = split_list_items(response, params.get("kind", "bullet"))
    if not items:
        return False
    return count_words(response) == int(params["factor"]) * len(items)


def v_item_words_ladder(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """List items form a word-count ladder: ``start`` words in the first item,
    one more in every next (``direction: desc`` reverses it).
    """
    items = split_list_items(response, params.get("kind", "any"))
    if len(items) < 2:
        return False
    counts = [count_words(it) for it in items]
    start = int(params["start"])
    expected = [start + i for i in range(len(counts))]
    if params.get("direction") == "desc":
        expected.reverse()
    return counts == expected


def v_first_line_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The first line is a bare number equal to the word count of everything
    after it. The number has to be committed before the text is written.
    """
    head, sep, rest = response.partition("\n")
    if not sep or not rest.strip():
        return False
    head = head.strip()
    if not head.isdigit():
        return False
    return int(head) == count_words(rest)


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
# Exact counters, cross-references, chains, mirrors
# --------------------------------------------------------------------------- #

#: Russian alphabet order for the "items in alphabetical order" check. ``ё`` is
#: absent because :func:`first_letter` folds it into ``е`` before comparison.
_RU_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"


def v_forbid_letters(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """A double lipogram: none of the listed letters occurs, in either case."""
    low = response.lower()
    return all(letter.lower() not in low for letter in params["letters"])


def v_comma_count(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The whole answer carries exactly the stated number of commas."""
    return check_count(response.count(","), params)


def _count_letter_words(response: str, letter: str) -> int:
    """How many words start with the given letter (case- and ё-insensitive)."""
    target = _fold(letter)
    n = 0
    for token in get_words(response):
        core = word_core(token)
        if core and core[0] == target:
            n += 1
    return n


def v_letter_start_count(response: str, params: dict,
                         lang_code: str = DEFAULT_LANG) -> bool:
    """Exactly N words of the answer begin with the given letter."""
    return check_count(_count_letter_words(response, params["letter"]), params)


def v_word_exact_count(response: str, params: dict,
                       lang_code: str = DEFAULT_LANG) -> bool:
    """The word occurs exactly ``count`` times, all inflected forms summed.

    The stem is matched as a prefix, the same way the include-family matches —
    «чай», «чая» and «чаем» are one word in three forms and count three times.
    """
    stem = _fold(params["stem"])
    hits = sum(1 for tok in get_words(response) if word_core(tok).startswith(stem))
    return hits == int(params["count"])


def v_ring_word(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The answer ends on the same word it starts with (the ring device)."""
    words = get_words(response)
    if len(words) < 2:
        return False
    first, last = word_core(words[0]), word_core(words[-1])
    return bool(first) and first == last


def v_paragraph_echo(response: str, params: dict,
                     lang_code: str = DEFAULT_LANG) -> bool:
    """The last word of the first paragraph occurs again in the last paragraph."""
    paragraphs = split_paragraphs(response)
    if len(paragraphs) < 2:
        return False
    first_words = get_words(paragraphs[0])
    if not first_words:
        return False
    target = word_core(first_words[-1])
    if not target:
        return False
    return any(word_core(tok) == target for tok in get_words(paragraphs[-1]))


def v_item_chain(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Neighbouring list items are letter-linked: the last word of every item
    starts with the same letter as the first word of the item after it.
    """
    items = split_list_items(response, params.get("kind", "any"))
    if len(items) < 2:
        return False
    for cur, nxt in zip(items, items[1:]):
        cur_words, nxt_words = get_words(cur), get_words(nxt)
        if not cur_words or not nxt_words:
            return False
        a = first_letter(cur_words[-1])
        b = first_letter(nxt_words[0])
        if a is None or b is None or a != b:
            return False
    return True


def v_nth_word_from_end(response: str, params: dict,
                        lang_code: str = DEFAULT_LANG) -> bool:
    """The word at position ``n`` counted from the end is the given word."""
    n = int(params["n"])
    words = get_words(response)
    if len(words) < n or n < 1:
        return False
    return word_core(words[-n]) == _fold(params["word"]).strip()


_CHECKSUM_RE = re.compile(r"\((\d+)\)\s*$")


def v_checksum(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The answer ends with ``(N)`` where N restates a count over the answer.

    ``kind: "commas"``        — N is the number of commas;
    ``kind: "letter_words"``  — N is the number of words starting with
                                ``params["letter"]``;
    ``kind: "sentences"``     — N is the number of sentences, the marker excluded;
    ``kind: "words"``         — N is the number of words, the marker excluded.

    The first two counts are not disturbed by the ``(N)`` marker itself (it
    carries no commas and no letters), so they read the whole answer. The latter
    two *are* disturbed by it — ``(N)`` is a word, and an unterminated
    tail is a sentence — so they read the answer with the marker cut off, which
    is exactly what their wording promises ("саму пометку не считай").
    """
    text = response.strip()
    m = _CHECKSUM_RE.search(text)
    if not m:
        return False
    stated = int(m.group(1))
    kind = params["kind"]
    if kind == "commas":
        return stated == response.count(",")
    if kind == "letter_words":
        return stated == _count_letter_words(response, params["letter"])
    body = _CHECKSUM_RE.sub("", text).strip()
    if kind == "sentences":
        return stated == len(split_sentences(body))
    if kind == "words":
        return stated == count_words(body)
    raise ValueError(f"unknown checksum kind {kind!r}")


def v_word_char_count(response: str, params: dict,
                      lang_code: str = DEFAULT_LANG) -> bool:
    """Two exact totals at once: N words and M characters (spaces counted)."""
    return (count_words(response) == int(params["words"])
            and len(response.strip()) == int(params["chars"]))


def v_order_words(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Both words occur (any form), and the first occurrence of A precedes the
    first occurrence of B."""
    stem_a, stem_b = _fold(params["stem_a"]), _fold(params["stem_b"])
    pos_a = pos_b = None
    for i, tok in enumerate(get_words(response)):
        core = word_core(tok)
        if pos_a is None and core.startswith(stem_a):
            pos_a = i
        if pos_b is None and core.startswith(stem_b):
            pos_b = i
    return pos_a is not None and pos_b is not None and pos_a < pos_b


def v_alpha_items(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """List items are ordered by strictly ascending first letters (Russian
    alphabet order, ё folded into е, no two items sharing an initial)."""
    items = split_list_items(response, params.get("kind", "any"))
    if len(items) < 2:
        return False
    ranks = []
    for it in items:
        letter = first_letter(it)
        if letter is None or letter not in _RU_ALPHABET:
            return False
        ranks.append(_RU_ALPHABET.index(letter))
    return all(a < b for a, b in zip(ranks, ranks[1:]))


def v_mirror_sentences(response: str, params: dict,
                       lang_code: str = DEFAULT_LANG) -> bool:
    """The first and the last sentence are exactly the same length in words."""
    sentences = split_sentences(response)
    if len(sentences) < 2:
        return False
    return count_words(sentences[0]) == count_words(sentences[-1])


def v_sentence_acrostic(response: str, params: dict,
                        lang_code: str = DEFAULT_LANG) -> bool:
    """First letters of the sentences, in order, spell the given word."""
    target = _fold(params["word"])
    sentences = split_sentences(response)
    if len(sentences) != len(target):
        return False
    initials = [first_letter(s) for s in sentences]
    return all(a is not None and a == b for a, b in zip(initials, target))


def v_anaphora(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every paragraph opens with one and the same word (the anaphora device)."""
    paragraphs = split_paragraphs(response)
    if len(paragraphs) < 2:
        return False
    firsts = []
    for p in paragraphs:
        words = get_words(p)
        if not words:
            return False
        core = word_core(words[0])
        if not core:
            return False
        firsts.append(core)
    return len(set(firsts)) == 1


# --------------------------------------------------------------------------- #
# The everyday requirements
#
# What this block has in common: none of it is a puzzle. "Plain text, no
# markdown", "wrap at sixty columns", "no digits", "don't start with
# «Конечно»" are things people actually write in a system prompt, and all of
# them are decidable by looking at the answer. Each type is a *family*: the
# same verifier serves an easy step and a hard one, and which step ships is
# decided by measurement (see ifhb11/calibrate.py), not by taste.
# --------------------------------------------------------------------------- #

#: A line that opens a markdown block: heading, quote, bullet or numbered item.
#: The Russian em dash «—» is deliberately absent: it opens direct speech, not a
#: list, and treating it as markup would fail ordinary Russian prose.
_MD_LINE_RE = re.compile(r"^\s*(#|>|[-+•]\s|\d+[.)]\s)")


def v_no_markdown(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Plain text: no markup characters and no line that opens a markdown block.

    The asterisk and the backtick are banned outright rather than matched in
    pairs. Pair matching is fragile (one stray ``*`` reads as either emphasis or
    a bullet depending on what follows), and "no asterisks at all" is both
    simpler to verify and simpler to *say*, which is what the wording does.
    """
    if "*" in response or "`" in response or "__" in response:
        return False
    return not any(_MD_LINE_RE.match(line) for line in response.splitlines())


#: A bold span: ``**...**`` with no asterisk or newline inside.
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def _bold_core(token: str) -> Optional[str]:
    """The text of a fully bold word token, or ``None`` if it is not one."""
    stripped = token.strip(_EDGE_PUNCT.replace("*", ""))
    m = re.fullmatch(r"\*\*([^*]+)\*\*", stripped)
    return m.group(1) if m else None


def v_bold_words(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Exactly ``count`` bold spans, plus whatever the harder steps add.

    ``words_each``  — every span holds exactly that many words (one, normally);
    ``position``    — the word at that 1-based position of the answer is the
                      bold one (the family's positional step);
    ``stems``       — the bold spans are exactly the words the stack requires
                      elsewhere, one span per stem (the cross-reference step).

    ``response.count("**") == 2 * count`` is checked separately from the span
    count: three well-formed spans plus one dangling ``**`` would otherwise pass.
    """
    count = int(params["count"])
    spans = _BOLD_RE.findall(response)
    if len(spans) != count or response.count("**") != 2 * count:
        return False
    if "words_each" in params:
        each = int(params["words_each"])
        if not all(count_words(s) == each for s in spans):
            return False
    if "position" in params:
        n = int(params["position"])
        words = get_words(response)
        if len(words) < n or n < 1:
            return False
        if _bold_core(words[n - 1]) is None:
            return False
    if "stems" in params:
        stems = [_fold(s) for s in params["stems"]]
        cores = [word_core(s) for s in spans]
        if len(cores) != len(stems):
            return False
        left = list(stems)
        for core in cores:
            hit = next((s for s in left if core.startswith(s)), None)
            if hit is None:
                return False
            left.remove(hit)
    return True


def v_line_length(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """No line longer than ``max`` characters; optionally, exactly ``count`` lines.

    Trailing spaces are not counted — they are invisible, and failing an answer
    for one would be a requirement about whitespace rather than about width.
    """
    lines = response.strip().splitlines()
    if not lines:
        return False
    if max(len(line.rstrip()) for line in lines) > int(params["max"]):
        return False
    if "count" in params:
        return len([ln for ln in lines if ln.strip()]) == int(params["count"])
    return True


def v_word_cap(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The word occurs at most ``cap`` times, all inflected forms summed.

    The directed, honest version of the retired ``no_repeated_word``: instead of
    banning repetition in general (which measured as a requirement about Russian
    syntax), it caps the one word the topic leans on.
    """
    stem = _fold(params["stem"])
    hits = sum(1 for tok in get_words(response) if word_core(tok).startswith(stem))
    return hits <= int(params["cap"])


def v_exclude_phrases(response: str, params: dict,
                      lang_code: str = DEFAULT_LANG) -> bool:
    """None of the listed phrases appears — anywhere, or only at the opening.

    ``mode: "start"`` checks the beginning of the answer (the anti-«Конечно!»
    rule), ``mode: "anywhere"`` searches the whole text. Matching is folded, so
    case and ё never decide a verdict.
    """
    text = _fold(response.strip())
    phrases = [_fold(p) for p in params["phrases"]]
    if params.get("mode") == "start":
        head = text.lstrip("«\"'“„—– \t\n")
        return not any(head.startswith(p) for p in phrases)
    return not any(p in text for p in phrases)


def v_max_word_length(response: str, params: dict,
                      lang_code: str = DEFAULT_LANG) -> bool:
    """No word longer than ``max`` letters; a hyphen is not a letter.

    Counting letters rather than characters is what the wording promises and
    what makes the rule checkable by hand: «чёрно-белый» is ten letters, not
    eleven, and a digit group is no letters at all.
    """
    words = get_words(response)
    if not words:
        return False
    limit = int(params["max"])
    for token in words:
        core = word_core(token)
        if sum(1 for ch in core if ch.isalpha()) > limit:
            return False
    return True


_TERMINATOR_TAIL_RE = re.compile(r"[.!?…]+$")


def _sentences_ending_with(response: str, char: str) -> int:
    """Sentences whose terminating punctuation contains ``char``.

    A sentence is what :func:`split_sentences` says it is, so «Правда?!» counts
    as both a question and an exclamation — which is what the wording of both
    types says ("предложений, заканчивающихся вопросительным знаком").
    """
    n = 0
    for sentence in split_sentences(response):
        tail = sentence.strip().rstrip("\"'»)]}“„ ")
        m = _TERMINATOR_TAIL_RE.search(tail)
        if m and char in m.group(0):
            n += 1
    return n


def v_question_count(response: str, params: dict,
                     lang_code: str = DEFAULT_LANG) -> bool:
    return check_count(_sentences_ending_with(response, "?"), params)


def v_exclamation_count(response: str, params: dict,
                        lang_code: str = DEFAULT_LANG) -> bool:
    return check_count(_sentences_ending_with(response, "!"), params)


def _linked_ban(response: str, params: dict, banned: bool) -> bool:
    """A ban plus, when the type ships as a pairing, the word that must replace
    what was banned.

    Both bans are nearly free on their own (measured: 0.64 and 0.88 of
    unconstrained answers already satisfy them). They ship joined to an
    ``any_stems`` requirement — say the number from the context, but in words;
    name the term from the context, but in Cyrillic — and that pairing is what
    the verifier checks.
    """
    if banned:
        return False
    if "any_words" in params:
        # Exact word forms, not prefixes. The include-family matches stems, and
        # for a numeral that is too loose: «два» is a prefix of «двадцать», so a
        # prefix match would accept an answer that named a different number.
        wanted = {_fold(w) for w in params["any_words"]}
        return any(word_core(tok) in wanted for tok in get_words(response))
    if "any_stems" in params:
        return any(_has_stem(response, _fold(s)) for s in params["any_stems"])
    return True


def v_no_digits(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Not a single digit anywhere — every number spelled out in words."""
    return _linked_ban(response, params, any(ch.isdigit() for ch in response))


_LATIN_RE = re.compile(r"[A-Za-z]")


def v_no_latin(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Not a single Latin letter — a term from the context must be russified."""
    return _linked_ban(response, params, bool(_LATIN_RE.search(response)))


def v_quote_task(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The first line repeats the task verbatim — no folding, "дословно" means it."""
    lines = response.strip().splitlines()
    if not lines:
        return False
    return lines[0].strip() == params["text"].strip()


_HASHTAG_LINE_RE = re.compile(r"^#[а-яёА-ЯЁ]+(?: #[а-яёА-ЯЁ]+)*$")


def v_hashtags(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """The last line is exactly ``count`` Cyrillic hashtags separated by spaces.

    ``alpha: true`` additionally requires them in alphabetical order, and
    ``echo_first: true`` that the last hashtag repeats the answer's first word —
    the family's ordering and cross-reference steps.
    """
    lines = [ln for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1].strip()
    if not _HASHTAG_LINE_RE.match(last):
        return False
    tags = [t[1:] for t in last.split()]
    if len(tags) != int(params["count"]):
        return False
    folded = [_fold(t) for t in tags]
    if params.get("alpha") and folded != sorted(folded):
        return False
    if params.get("echo_first"):
        body = "\n".join(lines[:-1])
        words = get_words(body)
        if not words or word_core(words[0]) != folded[-1]:
            return False
    return True


def v_anadiplosis(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every sentence opens with the word the previous sentence closed on."""
    sentences = split_sentences(response)
    if len(sentences) < 2:
        return False
    for cur, nxt in zip(sentences, sentences[1:]):
        cur_words, nxt_words = get_words(cur), get_words(nxt)
        if not cur_words or not nxt_words:
            return False
        if word_core(cur_words[-1]) != word_core(nxt_words[0]):
            return False
    return True


def v_epiphora(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Every list item (or paragraph) ends on one and the same word."""
    unit = params.get("unit", "items")
    if unit == "items":
        pieces = split_list_items(response, params.get("kind", "any"))
    else:
        pieces = split_paragraphs(response)
    if len(pieces) < 2:
        return False
    lasts = []
    for piece in pieces:
        words = get_words(piece)
        if not words:
            return False
        core = word_core(words[-1])
        if not core:
            return False
        lasts.append(core)
    return len(set(lasts)) == 1


# --------------------------------------------------------------------------- #
# Conditional requirements
#
# A rule that has to be *evaluated* before it can be followed. The generator
# owns the context, so which branch holds is decided at generation time and
# shipped in the parameters; the model sees only the wording, which always
# states both branches. What the verifier does is therefore ordinary — it is
# the question that is new, not the check.
# --------------------------------------------------------------------------- #

def v_conditional_include(response: str, params: dict,
                          lang_code: str = DEFAULT_LANG) -> bool:
    """Use the word if the condition holds; if it does not, the word is banned."""
    if params["holds"]:
        return _has_stem(response, _fold(params["stem"]))
    return not _has_stem(response, _fold(params["stem"]))


def v_conditional_format(response: str, params: dict,
                         lang_code: str = DEFAULT_LANG) -> bool:
    """Whichever of the two shapes the condition selected must be the one used.

    Both branches are real requirements — a list of N items, or prose of M
    sentences — so unlike :func:`v_conditional_include` neither branch is free.
    """
    branch = params["then"] if params["holds"] else params["otherwise"]
    kind = branch["shape"]
    if kind == "bullet_list":
        return v_bullet_list(response, branch)
    if kind == "numbered_list":
        return v_numbered_list(response, branch)
    if kind == "sentences":
        return v_sentence_count(response, branch)
    if kind == "paragraphs":
        return v_paragraph_count(response, branch)
    raise ValueError(f"unknown conditional shape {kind!r}")


def v_override(response: str, params: dict, lang_code: str = DEFAULT_LANG) -> bool:
    """Only the rule the meta-rule elects is checked; the loser is not.

    The pair is deliberately contradictory, so verifying both would make the
    question unpassable. What is measured is whether the model noticed which of
    the two wins — an answer that "splits the difference" fails the winner.
    """
    winner = params["winner"]
    return verify(winner["category"], response, winner["params"], lang_code)


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
    "structure:items_total_words": v_items_total_words,
    "structure:item_words_ladder": v_item_words_ladder,
    "structure:first_line_count": v_first_line_count,
    # format — lists, JSON, tables
    "format:bullet_list": v_bullet_list,
    "format:numbered_list": v_numbered_list,
    "format:json_object": v_json_object,
    "format:json_list_value": v_json_list_value,
    "format:json_array": v_json_array,
    "format:csv_line": v_csv_line,
    "format:two_parts": v_two_parts,
    "format:markdown_table": v_markdown_table,
    # Exact counters
    "style:forbid_letters": v_forbid_letters,
    "style:comma_count": v_comma_count,
    "style:letter_start_count": v_letter_start_count,
    "lexical:word_exact_count": v_word_exact_count,
    "structure:checksum": v_checksum,
    "structure:word_char_count": v_word_char_count,
    # Cross-references, chains and mirrors
    "lexical:ring_word": v_ring_word,
    "lexical:paragraph_echo": v_paragraph_echo,
    "lexical:item_chain": v_item_chain,
    "lexical:nth_word_from_end": v_nth_word_from_end,
    "lexical:order_words": v_order_words,
    "lexical:alpha_items": v_alpha_items,
    "structure:mirror_sentences": v_mirror_sentences,
    "lexical:sentence_acrostic": v_sentence_acrostic,
    "style:anaphora": v_anaphora,
    # The everyday requirements
    "style:no_markdown": v_no_markdown,
    "format:bold_words": v_bold_words,
    "format:line_length": v_line_length,
    "lexical:word_cap": v_word_cap,
    "lexical:exclude_phrases": v_exclude_phrases,
    "style:max_word_length": v_max_word_length,
    "style:question_count": v_question_count,
    "style:exclamation_count": v_exclamation_count,
    "style:no_digits": v_no_digits,
    "style:no_latin": v_no_latin,
    "lexical:quote_task": v_quote_task,
    "format:hashtags": v_hashtags,
    "lexical:anadiplosis": v_anadiplosis,
    "lexical:epiphora": v_epiphora,
    # Conditional requirements
    "logic:conditional_include": v_conditional_include,
    "logic:conditional_format": v_conditional_format,
    "logic:override": v_override,
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
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def normalize_generation(generation: str) -> str:
    """Strip reasoning traces, then outer whitespace — nothing else.

    Reasoning blocks are scaffolding, not part of the answer, so leaving them in
    would fail every constraint for a reasoning model regardless of how well it
    followed the instruction. Beyond that the response is scored verbatim: any
    further "lenient extraction" (peeling off a preamble, unwrapping code fences)
    would score a response the instruction did not ask for.

    The trace is delimited by its **closing** tag, not by a balanced pair:
    everything up to and including the first ``</think>`` is dropped. Matching
    ``<think>...</think>`` alone is not enough in practice, because the opening
    tag frequently never appears in the generation —

    * chat templates for reasoning models routinely pre-fill ``<think>`` at the
      end of the prompt, so the model emits only ``trace</think>answer`` and a
      balanced-pair rule would score the whole trace;
    * a server that re-assembles a separated trace back into the text, or a
      model that leaks a token before opening the block, yields
      ``token<think>trace</think>answer``, where a balanced-pair rule leaves the
      leaked token glued to the answer and corrupts every exact counter.

    Any further balanced blocks are removed afterwards, so a model that
    interleaves several traces keeps the answer between them.

    An **unterminated** trace has no ``</think>`` and is deliberately left
    alone: the model was cut off inside its own reasoning, and that text is the
    answer it actually produced. It fails — the answer is not absent, it is
    wrong.
    """
    text = generation or ""
    close = _THINK_CLOSE_RE.search(text)
    if close:
        text = text[close.end():]
    text = _THINK_BLOCK_RE.sub("", text)
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
    """Machine-readable constraints of a question (JSON string in ``meta``).

    This field is the whole scoring key: no other part of the sample says what
    the answer has to satisfy. A build that ships it blank cannot be scored at
    all, so an empty value fails loudly here rather than being read as "this
    question has no requirements" — which would silently score every answer as
    a pass-by-vacuity or a zero, depending on the metric.
    """
    raw = doc["meta"]["constraints"]
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError(
                "meta.constraints is empty for sample %r: this build carries no "
                "scoring key and cannot be evaluated. Re-upload the split with "
                "meta.constraints populated (it is required for scoring; only "
                "`outputs` may be blanked for a private dataset)."
                % (doc.get("meta", {}).get("id"),)
            )
        return json.loads(raw)
    return raw


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


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Score one generation against the question's own constraint stack.

    An empty response never passes: prohibition-style constraints (forbidden
    word, no commas, upper-bounded counts, ...) are vacuously satisfied by the
    empty string, so without this guard a model that answered nothing would
    "pass" every question whose stack happens to contain only such constraints.

    ``constraint_pass_rate`` is ``None`` for an empty response — the metric is
    declared over non-empty answers (see the module docstring), and
    :func:`agg_constraint_pass_rate` skips the ``None`` values when averaging.
    ``sample_pass_rate`` counts the empty response as a plain failure.

    ``balance_score`` needs the verdicts themselves, not a share, because it is
    aggregated per requirement type across the whole run — so the per-sample
    value is the list of ``(category, pass)`` pairs. See
    :func:`agg_balance_score`.
    """
    response = normalize_generation(_extract_prediction(results))
    checks = score_response(doc, response)
    n_total = len(checks)
    if n_total == 0:
        eval_logger.warning("sample %s has no constraints", doc["meta"].get("id"))
        return {"sample_pass_rate": 0.0, "constraint_pass_rate": 0.0,
                "balance_score": None}

    n_pass = sum(1 for c in checks if c["pass"])
    is_empty = not response.strip()
    return {
        "sample_pass_rate": float(n_pass == n_total and not is_empty),
        "constraint_pass_rate": (None if is_empty else n_pass / n_total),
        "balance_score": (None if is_empty else
                          [[c["category"], bool(c["pass"])] for c in checks]),
    }


def agg_constraint_pass_rate(values: List[Optional[float]]) -> float:
    """Mean of the per-sample constraint shares over non-empty responses.

    Empty responses contribute ``None`` (see :func:`process_results`) and are
    excluded: they already fail ``sample_pass_rate``, and letting them zero
    five constraint verdicts at once would turn this diagnostic of requirement
    comprehensibility into a proxy for token budgeting. If every response is
    empty the metric is 0.0 rather than undefined.
    """
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


#: A per-type rate never enters the geometric mean below this value. A type the
#: model never once gets right must cost a visible penalty, not annihilate the
#: score: with 61 types, a floored factor multiplies ``balance_score`` by
#: ``0.01 ** (1/61) ≈ 0.93``, so each dead type costs about seven per cent.
BALANCE_FLOOR = 0.01

#: A type needs at least this many verdicts in the run before it may move the
#: score. Two types ship below the dataset's own reporting floor (the bank owes
#: them material), and one generation deciding a whole factor of a headline
#: metric would be noise, not measurement. Excluded types are logged, not
#: silently dropped.
BALANCE_MIN_SUPPORT = 5


def balance_components(values: Sequence[Optional[Sequence[Sequence[Any]]]]
                       ) -> Tuple[Dict[str, Tuple[int, int]], List[str]]:
    """``({category: (passed, seen)}, categories below the support floor)``."""
    tally: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for value in values:
        if not value:                      # empty response, or no constraints
            continue
        for category, ok in value:
            tally[category][0] += bool(ok)
            tally[category][1] += 1
    counted = {c: (p, n) for c, (p, n) in tally.items() if n >= BALANCE_MIN_SUPPORT}
    thin = sorted(c for c, (_, n) in tally.items() if n < BALANCE_MIN_SUPPORT)
    return counted, thin


def agg_balance_score(values: List[Optional[List[List[Any]]]]) -> float:
    """Geometric mean of the per-requirement-type pass rates.

    ``constraint_pass_rate`` is a mean over *verdicts*, so it is weighted by how
    often each requirement type occurs: a model can score well on it while being
    unable to do a whole kind of instruction, as long as that kind is rare.
    ``balance_score`` averages within each type first and then combines the types
    with equal weight, so every kind of instruction counts the same however many
    questions use it.

    The combination is geometric, which is what makes it a balance measure: the
    same total number of failures costs more when concentrated in one type than
    when spread evenly, so a model cannot buy a headline by being excellent at
    the common types and hopeless at one. A rate of zero would zero the whole
    product, so rates are floored at :data:`BALANCE_FLOOR` — a blind spot is a
    penalty, not an annihilation. Types with fewer than
    :data:`BALANCE_MIN_SUPPORT` verdicts in the run stay out (they are returned
    by :func:`balance_components` for reporting).

    **Read it against ``constraint_pass_rate``, never against
    ``sample_pass_rate``.** Both this metric and ``constraint_pass_rate`` are
    built from per-requirement verdicts and live on that scale;
    ``sample_pass_rate`` is a product over the whole stack and is structurally
    far lower (at the shipped mean stack of 4.43 and a per-requirement rate of
    0.83 it lands near 0.83⁴·⁴³ ≈ 0.44). A ``balance_score`` above
    ``sample_pass_rate`` is the normal case, not a sign that the geometric mean
    failed to bite.

    Two things separate it from ``constraint_pass_rate`` and they pull in
    opposite directions: equal weight per type instead of weight by frequency
    (which can move the number either way — up, when the model's frequent types
    are its hard ones), and a geometric mean instead of an arithmetic one (always
    down). Reporting the equal-weight *arithmetic* mean alongside separates the
    two: the gap between it and this metric is the concentration penalty alone,
    and it widens for a model whose failures sit in a handful of types.

    Empty responses are excluded, exactly as in :func:`agg_constraint_pass_rate`
    and for the same reason; they already fail ``sample_pass_rate``.
    """
    counted, _thin = balance_components(values)
    if not counted:
        return 0.0
    logs = [math.log(max(p / n, BALANCE_FLOOR)) for p, n in counted.values()]
    return math.exp(sum(logs) / len(logs))


if Filter is not None and "remove_whitespace_and_nones" not in FILTER_REGISTRY:
    @register_filter("remove_whitespace_and_nones")
    class RemoveWhitespaceAndNones(Filter):

        def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
            def filter_set(inst):
                filtered_resp = []
                for resp in inst:
                    if not resp:
                        resp = ""
                    else:
                        resp = resp.lstrip()
                    filtered_resp.append(resp)
                return filtered_resp

            filtered_resps = [filter_set(resp) for resp in resps]

            return filtered_resps
