"""SOBHard — deterministic structured-output scoring for lm-evaluation-harness.

The benchmark scores *structured output under explicit constraints*: the model is
given a document in one machine format and must emit a document in another,
packaged exactly as the prompt demands. Nothing here is judged by an LLM and
nothing depends on wall-clock, network or randomness, so a run is exactly
reproducible and every point of the score traces back to a named constraint.

The prompt makes about a dozen separate promises to the model — one fenced block,
a specific fence tag, no prose outside it, key order not graded, numbers within
1e-6 relative tolerance, strings byte-exact, plus the semantics of the task family
itself. This module checks each of them separately::

    check(ctx) -> (passed: bool, detail: str)

Layers
------
``pack.*``    packaging  — exactly one fenced block, correct tag, nothing outside
``syn.*``     syntax     — the body parses as the target format
``val.*``     values     — structure, numbers within tolerance, strings byte-exact
``<family>.*`` semantics — what a plain comparison with the reference cannot see

Metrics
-------
``sample_pass_rate``
    All-or-nothing: 1.0 iff *every* applicable constraint holds. Headline metric.
``constraint_pass_rate``
    Share of individually satisfied constraints — partial credit, and the
    diagnostic that separates "wrong packaging" from "wrong content".
``format_pass_rate`` / ``content_pass_rate`` / ``task_pass_rate``
    The three layers scored separately, so a model that solves the task but
    cannot follow the packaging rules is distinguishable from one that cannot
    solve it at all.

Self-containment
----------------
This module imports nothing from the SOBHard generator. Checks that need the
*source* document in the pipeline's canonical shape (which requires parsers for
twenty exotic input formats) are precomputed at build time and shipped inside
``meta.checks``; here they are plain JSON comparisons. Only the five *target*
formats are parsed at scoring time, and those parsers are byte-compatible ports
of the generator's own.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import math
import re
from configparser import ConfigParser, Error as CFGError, MissingSectionHeaderError
from typing import Any, Callable, Dict, List, Tuple

eval_logger = logging.getLogger(__name__)

RTOL = 1e-6

#: Formats with no type system: every value is a string after parsing, so the
#: promised 1e-6 numeric tolerance only applies with coercion.
TYPELESS_FORMATS = {"ini", "csv", "tsv"}

#: Keywords a schema may contain, per the inference rules its question states.
#: ``$schema`` is tolerated everywhere: the question asks for a schema "in the
#: draft-07 style", and declaring the dialect is the standard way to answer
#: that. It is a declaration, not a constraint, so it does not violate "no extra
#: constraints" — and 17 of 41 answered schema questions in the first real run
#: were failed for it alone. It is also stripped before the content comparison
#: (see Ctx), otherwise tolerating it here would change nothing.
SCHEMA_ALLOWED_KEYWORDS: Dict[str, set] = {
    "minimal": {"type", "properties", "required", "items", "$schema"},
    "union": {"type", "properties", "required", "items", "$schema"},
    "bounds": {"type", "properties", "required", "items", "minItems",
               "maxItems", "$schema"},
    "enum": {"type", "properties", "required", "items", "enum", "$schema"},
    "oneOf": {"type", "properties", "required", "items", "oneOf", "$schema"},
    "full": {"type", "properties", "required", "items", "minItems", "maxItems",
             "enum", "$schema"},
}

#: Fence tag per target format — mirrors the generator's FORMAT_TAGS.
FORMAT_TAGS: Dict[str, str] = {
    "json": "json", "jsonl": "jsonl", "yaml": "yaml",
    "toml": "toml", "ini": "ini",
}

#: Reasoning traces are scaffolding, not the answer.
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


class FormatError(ValueError):
    """A document did not parse as its declared format."""


class MissingParser(RuntimeError):
    """No parser is installed for a format the dataset uses.

    Raised rather than reported as a parse failure, and deliberately fatal: a
    missing parser would otherwise score every question in that format as zero
    and blame the model for a property of the environment. A run that cannot
    score 10% of the set must stop, not quietly report a lower number.
    """


# --------------------------------------------------------------------------- #
# Target-format parsers (ports of the generator's, same semantics)
# --------------------------------------------------------------------------- #

# CommonMark allows three OR MORE backticks, and tildes as an alternative. Models
# reach for four backticks precisely when the document itself contains three, so a
# three-only pattern fails on exactly the documents most likely to need it.
_FENCE_STRIP_RE = re.compile(
    r"^(?P<f>`{3,}|~{3,})[A-Za-z0-9_+-]*[ \t]*\n(?P<body>.*?)\n?(?P=f)`*~*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _FENCE_STRIP_RE.match(text.strip())
    return m.group("body") if m else text.strip()


def _parse_json(text: str) -> Any:
    s = _strip_code_fence(text)
    if not s:
        raise FormatError("empty input")
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise FormatError(f"invalid JSON: {e}") from e


def _parse_jsonl(text: str) -> List[Any]:
    s = _strip_code_fence(text)
    if not s:
        return []
    items: List[Any] = []
    for i, line in enumerate(s.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise FormatError(f"invalid JSON on line {i}: {e}") from e
    return items


def _parse_yaml(text: str) -> Any:
    import yaml  # PyYAML is already an lm-eval dependency

    s = _strip_code_fence(text)
    if not s:
        raise FormatError("empty input")
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError as e:
        raise FormatError(f"invalid YAML: {e}") from e


def _parse_toml(text: str) -> Any:
    try:
        import tomllib                       # Python >= 3.11
    except ImportError:                      # pragma: no cover
        try:
            import tomli as tomllib          # type: ignore[no-redef]
        except ImportError as e:             # pragma: no cover
            raise MissingParser(
                "SOBHard needs a TOML parser for the 50 questions whose target "
                "format is TOML. Python 3.11+ ships tomllib; on 3.10 and older "
                "install the backport:\n\n    pip install tomli\n"
            ) from e
    s = _strip_code_fence(text)
    if not s:
        raise FormatError("empty input")
    try:
        return tomllib.loads(s)
    except Exception as e:
        raise FormatError(f"invalid TOML: {e}") from e


def _parse_ini(text: str) -> Dict[str, Any]:
    s = _strip_code_fence(text)
    if not s:
        raise FormatError("empty input")
    cp = ConfigParser(interpolation=None, delimiters=("=", ":"),
                      strict=False, default_section="__default__")
    cp.optionxform = str                     # preserve key case
    try:
        cp.read_string(s)
    except (MissingSectionHeaderError, CFGError) as e:
        raise FormatError(f"invalid INI: {e}") from e
    out: Dict[str, Any] = {sec: dict(cp.items(sec)) for sec in cp.sections()}
    if cp.defaults():
        out["__default__"] = dict(cp.defaults())
    return out


_PARSERS: Dict[str, Callable[[str], Any]] = {
    "json": _parse_json, "jsonl": _parse_jsonl, "yaml": _parse_yaml,
    "toml": _parse_toml, "ini": _parse_ini,
}


def parse_as(fmt: str, text: str) -> Tuple[Any, str | None]:
    """Parse ``text`` as ``fmt``. Returns ``(object, error)``.

    Never raises for a malformed *document* — that is a legitimate failure of the
    answer and becomes ``syn.parses_target``. A missing *parser* propagates: it
    is a property of the environment, and silently attributing it to the model
    would put a false ceiling on the score.
    """
    fn = _PARSERS.get(fmt)
    if fn is None:
        return None, f"no parser for target format {fmt!r}"
    try:
        return fn(text), None
    except MissingParser:
        raise
    except Exception as e:                   # noqa: BLE001 — any failure is a parse failure
        return None, f"{type(e).__name__}: {e}"


def check_dependencies(formats: Any = ("json", "jsonl", "yaml", "toml", "ini")) -> None:
    """Fail fast if a format used by the dataset has no parser installed."""
    missing = []
    probe = {"json": "{}", "jsonl": "{}", "yaml": "a: 1", "toml": "a = 1", "ini": "[s]\na = 1"}
    for fmt in formats:
        try:
            _PARSERS[fmt](probe[fmt])
        except MissingParser as e:
            missing.append(f"{fmt}: {e}")
        except Exception:                    # noqa: BLE001 — probe content is not the point
            pass
    if missing:
        raise MissingParser("\n".join(missing))


# --------------------------------------------------------------------------- #
# Markdown fence handling
# --------------------------------------------------------------------------- #

# Three or more backticks, or three or more tildes, plus an optional info string.
_FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")


def extract_blocks(text: str) -> Tuple[List[Tuple[str, str]], str, bool]:
    """Split a response into fenced blocks, the text outside them, and whether
    an opening fence was left unclosed.

    Line-based on purpose: a fence is a line consisting only of ``` plus an
    optional tag. More predictable than a whole-text regex, and it does not trip
    over triple backticks inside a string value of the document itself.
    """
    lines = (text or "").split("\n")
    fences = [(i, m.group(1)) for i, ln in enumerate(lines)
              if (m := _FENCE_LINE.match(ln))]
    blocks: List[Tuple[str, str]] = []
    covered: set[int] = set()
    i = 0
    while i + 1 < len(fences):
        o, tag = fences[i]
        c, _ = fences[i + 1]
        blocks.append((tag, "\n".join(lines[o + 1:c])))
        covered.update(range(o, c + 1))
        i += 2
    unclosed = (len(fences) - i) == 1
    if unclosed:
        covered.add(fences[i][0])
    outside = "\n".join(ln for k, ln in enumerate(lines) if k not in covered)
    return blocks, outside.strip(), unclosed


# --------------------------------------------------------------------------- #
# Structural comparison
# --------------------------------------------------------------------------- #

def _as_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def _num_close(a: float, b: float, rtol: float = RTOL) -> bool:
    if a == b:
        return True
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def diff_values(pred: Any, gold: Any, *, coerce_numeric: bool = False,
                rtol: float = RTOL, path: str = "$") -> List[Tuple[str, str, str]]:
    """Compare two structures, splitting mismatches into categories.

    Categories: ``structure`` (key set, list length, type), ``numeric`` (numbers
    outside tolerance), ``string`` (strings not byte-equal). Dict key order is
    ignored — as the prompt promises; list element order is not.
    """
    out: List[Tuple[str, str, str]] = []

    if isinstance(pred, dict) and isinstance(gold, dict):
        if set(pred) != set(gold):
            miss = sorted(set(gold) - set(pred))
            extra = sorted(set(pred) - set(gold))
            out.append(("structure", path, f"missing={miss[:5]} extra={extra[:5]}"))
            return out
        for k in gold:
            out += diff_values(pred[k], gold[k], coerce_numeric=coerce_numeric,
                               rtol=rtol, path=f"{path}.{k}")
        return out

    if isinstance(pred, list) and isinstance(gold, list):
        if len(pred) != len(gold):
            out.append(("structure", path, f"len {len(pred)} != {len(gold)}"))
            return out
        for i, (p, g) in enumerate(zip(pred, gold)):
            out += diff_values(p, g, coerce_numeric=coerce_numeric,
                               rtol=rtol, path=f"{path}[{i}]")
        return out

    if isinstance(pred, (dict, list)) != isinstance(gold, (dict, list)):
        out.append(("structure", path,
                    f"type {type(pred).__name__} != {type(gold).__name__}"))
        return out

    # Python makes True == 1 and False == 0, so a plain != would accept `1` where
    # the reference has `true`. In every target format those are different values,
    # and a conversion that swaps them is not lossless — require the same type.
    if isinstance(pred, bool) or isinstance(gold, bool):
        if not (isinstance(pred, bool) and isinstance(gold, bool)) or pred != gold:
            out.append(("structure", path, f"{pred!r} != {gold!r}"))
        return out
    if pred is None or gold is None:
        if pred is not gold:
            out.append(("structure", path, f"{pred!r} != {gold!r}"))
        return out

    p_num, g_num = _as_number(pred), _as_number(gold)
    native = isinstance(pred, (int, float)) and isinstance(gold, (int, float))
    if native or (coerce_numeric and p_num is not None and g_num is not None):
        if not _num_close(p_num, g_num, rtol):
            out.append(("numeric", path, f"{pred!r} != {gold!r}"))
        return out

    if isinstance(pred, str) and isinstance(gold, str):
        if pred != gold:
            out.append(("string", path, f"{pred[:40]!r} != {gold[:40]!r}"))
        return out

    if pred != gold:
        out.append(("structure", path, f"{pred!r} != {gold!r}"))
    return out


def list_order_violations(pred: Any, gold: Any, path: str = "$") -> List[str]:
    """Paths of lists whose members match as a multiset but not as a sequence.

    Isolates a reordering from a substitution: a changed value is reported by
    ``transform.values_unchanged`` instead.
    """
    def key(o: Any) -> str:
        return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)

    out: List[str] = []
    if isinstance(pred, dict) and isinstance(gold, dict) and set(pred) == set(gold):
        for k in gold:
            out += list_order_violations(pred[k], gold[k], f"{path}.{k}")
    elif isinstance(pred, list) and isinstance(gold, list) and len(pred) == len(gold):
        sp, sg = [key(x) for x in pred], [key(x) for x in gold]
        if sp != sg and sorted(sp) == sorted(sg):
            return [path]
        for i, (p, g) in enumerate(zip(pred, gold)):
            out += list_order_violations(p, g, f"{path}[{i}]")
    return out


def keys_sorted_recursively(obj: Any) -> Tuple[bool, str]:
    """Every object's keys appear in lexicographic order.

    Reads insertion order, which ``json.loads`` preserves, so the check sees the
    order the model actually emitted.
    """
    stack: List[Tuple[Any, str]] = [(obj, "$")]
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            ks = list(cur.keys())
            if ks != sorted(ks, key=str):
                bad = next((k for a, k in zip(ks, sorted(ks, key=str)) if a != k), "?")
                return False, f"{path}: order breaks at {bad!r}"
            for k, v in cur.items():
                stack.append((v, f"{path}.{k}"))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                stack.append((v, f"{path}[{i}]"))
    return True, ""


def leaf_multiset(obj: Any) -> List[str]:
    out: List[str] = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        else:
            out.append(json.dumps(cur, ensure_ascii=False, sort_keys=True))
    return sorted(out)


def schema_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def infer_schema(obj: Any, kind: str) -> Any:
    """The inference rules the schema_gen question states, one branch per kind.

    Five kinds, and three of them cannot be produced while reading the document
    from left to right: ``union`` needs every element of an array before it can
    name the array's properties, ``bounds`` needs its length, ``enum`` needs its
    distinct values and then a sort.
    """
    t = schema_type(obj)
    if t == "object":
        return {"type": "object",
                "properties": {k: infer_schema(v, kind) for k, v in obj.items()},
                "required": sorted(obj.keys(), key=str)}
    if t == "array":
        node: Dict[str, Any] = {"type": "array"}
        if obj:
            node["items"] = _infer_items(obj, kind)
        if kind in ("bounds", "full"):
            node["minItems"] = len(obj)
            node["maxItems"] = len(obj)
        return node
    return {"type": t}


def _infer_items(arr: List[Any], kind: str) -> Any:
    if kind in ("union", "full"):
        if all(isinstance(x, dict) for x in arr):
            first: Dict[str, Any] = {}
            for el in arr:
                for k, v in el.items():
                    if k not in first:
                        first[k] = v
            required = sorted(set.intersection(*[set(e) for e in arr]), key=str)
            return {"type": "object",
                    "properties": {k: infer_schema(first[k], kind)
                                   for k in sorted(first, key=str)},
                    "required": required}
        node = infer_schema(arr[0], kind)
        if kind == "full" and all(isinstance(x, str) for x in arr):
            node = dict(node)
            node["enum"] = sorted(set(arr))
        return node
    if kind == "oneOf":
        distinct: List[Any] = []
        for el in arr:
            node = infer_schema(el, kind)
            if node not in distinct:
                distinct.append(node)
        if len(distinct) == 1:
            return distinct[0]
        # The canonical JSON the question describes: keys in order, no spaces.
        distinct.sort(key=lambda n: json.dumps(n, ensure_ascii=False,
                                               sort_keys=True,
                                               separators=(",", ":")))
        return {"oneOf": distinct}
    node = infer_schema(arr[0], kind)
    if kind == "enum" and all(isinstance(x, str) for x in arr):
        node = dict(node)
        node["enum"] = sorted(set(arr))
    return node


def collect_schema_keywords(node: Any) -> set:
    """Keyword names used anywhere in a schema.

    Property names and the members of ``required`` and ``enum`` are data, not
    keywords, and are not collected — a document with a property called
    ``minItems`` must not be failed for naming it.
    """
    found: set = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                found.add(k)
                if k == "properties" and isinstance(v, dict):
                    stack.extend(v.values())
                elif k in ("required", "enum"):
                    continue
                else:
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


# --------------------------------------------------------------------------- #
# Transform rules: the property each one promises
# --------------------------------------------------------------------------- #
#
# A rule is checked by what it claims about the answer, not by re-running it.
# "Every object's keys are in order" or "no key contains a dot any more" are
# statements a correct answer satisfies however it was produced, and they catch
# the one failure a comparison with the reference cannot: an answer that copied
# the input and happens to differ from gold for an unrelated reason still fails
# the property, and the reason is named.

_DIGIT_STRING = re.compile(r"^-?[0-9]+$")
_PATH_KEY = re.compile(r"^[^.\[\]]+(\.[^.\[\]]+|\[\d+\])*$")


def _records_in(pred: Any) -> List[dict] | None:
    """The record list of an answer, whether or not it has been grouped."""
    if isinstance(pred, list) and all(isinstance(x, dict) for x in pred):
        return pred
    if isinstance(pred, dict) and pred and all(
            isinstance(v, list) and all(isinstance(x, dict) for x in v)
            for v in pred.values()):
        return [r for v in pred.values() for r in v]
    return None


def _numeric(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _p_sort_keys(pred: Any, p: dict) -> Tuple[bool, str]:
    return keys_sorted_recursively(pred)


def _p_flatten(pred: Any, p: dict) -> Tuple[bool, str]:
    if not isinstance(pred, dict):
        return False, f"the answer is {type(pred).__name__}, not an object"
    nested = [k for k, v in pred.items() if isinstance(v, (dict, list))]
    if nested:
        return False, f"still nested at {nested[:3]}"
    bad = [k for k in pred if not _PATH_KEY.match(str(k))]
    return not bad, f"keys that are not paths: {bad[:3]}"


def _p_unflatten(pred: Any, p: dict) -> Tuple[bool, str]:
    bad: List[str] = []
    stack = [pred]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if "." in str(k) or "[" in str(k):
                    bad.append(str(k))
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return not bad, f"keys still holding a path: {bad[:3]}"


def _p_camel_keys(pred: Any, p: dict) -> Tuple[bool, str]:
    bad: List[str] = []
    stack = [pred]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if "_" in str(k):
                    bad.append(str(k))
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return not bad, f"keys still in snake_case: {bad[:3]}"


def _p_filter(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    field, op, target = p["field"], p["op"], p["value"]
    if not any(field in r for r in records):
        # A later rule in the chain dropped the field the predicate names, so
        # the promise is no longer observable in the answer. Not a failure.
        return True, "the filtered field is not in the answer any more"
    for r in records:
        v = r.get(field)
        if op == "gt":
            ok = _numeric(v) is not None and _numeric(v) > target
        elif op == "lt":
            ok = _numeric(v) is not None and _numeric(v) < target
        elif op == "eq":
            ok = v == target
        else:
            ok = v != target
        if not ok:
            return False, f"a record survived the filter with {field}={v!r}"
    return True, ""


def _p_sort_records(pred: Any, p: dict) -> Tuple[bool, str]:
    lists: List[list] = []
    if isinstance(pred, list):
        lists = [pred]
    elif isinstance(pred, dict):
        lists = [v for v in pred.values() if isinstance(v, list)]
    if not lists:
        return False, f"the answer holds no list to be ordered ({type(pred).__name__})"
    field, numeric = p["field"], p["numeric"]
    for seq in lists:
        keys = []
        for r in seq:
            if not isinstance(r, dict) or field not in r:
                return True, "the ordering field is not in the answer any more"
            keys.append(_numeric(r[field]) if numeric else str(r[field]))
        if any(a is None for a in keys):
            return False, f"{field} is not numeric in every record"
        if keys != sorted(keys):
            first = next(i for i in range(1, len(keys)) if keys[i - 1] > keys[i])
            return False, f"order breaks at position {first} of {field}"
    return True, ""


def _p_group(pred: Any, p: dict) -> Tuple[bool, str]:
    if not isinstance(pred, dict):
        return False, f"grouping must give an object, got {type(pred).__name__}"
    ks = list(pred.keys())
    if ks != sorted(ks, key=str):
        return False, "group keys are not in lexicographic order"
    field = p["field"]
    for k, v in pred.items():
        if not isinstance(v, list) or not v:
            return False, f"group {k!r} is not a non-empty list"
        for r in v:
            if not isinstance(r, dict):
                return False, f"group {k!r} holds {type(r).__name__}, not a record"
            if field in r and str(r[field]) != str(k):
                return False, f"record with {field}={r[field]!r} is in group {k!r}"
    return True, ""


def _p_project(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    want = set(p["fields"])
    for r in records:
        # Which fields survive is the rule; what order they end up in is not,
        # and the answer-format block promises key order is free here.
        if set(r.keys()) != want:
            return False, f"record keys {sorted(r.keys())[:5]} != {sorted(want)}"
    return True, ""


def _p_numbers(pred: Any, p: dict) -> Tuple[bool, str]:
    bad: List[str] = []
    stack = [pred]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str) and _DIGIT_STRING.match(cur):
            bad.append(cur)
    return not bad, f"numeric strings left untouched: {bad[:3]}"


def _p_drop_empty(pred: Any, p: dict) -> Tuple[bool, str]:
    bad: List[str] = []
    stack: List[Tuple[Any, str]] = [(pred, "$")]
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if v == "" or v == [] or v == {}:
                    bad.append(f"{path}.{k}")
                stack.append((v, f"{path}.{k}"))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                stack.append((v, f"{path}[{i}]"))
    return not bad, f"empty entries left in place: {bad[:3]}"


RULE_PROPERTY: Dict[str, Callable[[Any, dict], Tuple[bool, str]]] = {
    "sort_keys": _p_sort_keys,
    "flatten": _p_flatten,
    "unflatten": _p_unflatten,
    "camel_keys": _p_camel_keys,
    "filter_records": _p_filter,
    "sort_records": _p_sort_records,
    "group_records": _p_group,
    "project_fields": _p_project,
    "numbers_from_strings": _p_numbers,
    "drop_empty": _p_drop_empty,
}


def canonical_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reference_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_dump(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Per-sample context
# --------------------------------------------------------------------------- #

def _meta_json(doc: Dict[str, Any], key: str, default: Any) -> Any:
    raw = (doc.get("meta") or {}).get(key)
    if raw in (None, ""):
        return default
    return json.loads(raw) if isinstance(raw, str) else raw


class Ctx:
    """Everything the constraints need about one (question, answer) pair.

    Written as a plain class rather than a dataclass on purpose. lm-eval loads
    this file through ``spec.loader.exec_module`` without registering it in
    ``sys.modules``, and on Python 3.11 ``@dataclass`` then dies looking up
    ``sys.modules[cls.__module__].__dict__``. The task would fail to load at
    all, before a single question was asked.
    """

    def __init__(self, doc: Dict[str, Any], response: str) -> None:
        self.doc = doc
        self.response = response
        meta = self.doc["meta"]
        cats = meta.get("categories") or {}
        self.family: str = cats.get("family", "")
        self.target_format: str = cats.get("target_format", "json")
        self.source_format: str = cats.get("source_format", "")
        self.expected_tag: str = meta.get("fence_tag") or FORMAT_TAGS.get(
            self.target_format, self.target_format)
        self.task_meta: dict = _meta_json(self.doc, "task_meta", {})

        checks = _meta_json(self.doc, "checks", {})
        # The input document, read under the convention the prompt states, so
        # the family checks can derive what they need from the source instead of
        # taking a second look at the reference.
        self.source = checks.get("source", checks.get("source_canonical"))
        self.gates: Dict[str, bool] = checks.get("gates") or {}
        self.schema_kind: str = self.task_meta.get("schema_kind", "minimal")
        self.program: List[dict] = self.task_meta.get("program") or []
        # A repair asked for in the input's own notation, as opposed to one that
        # also has to be re-expressed.
        self.same_notation: bool = bool(
            self.task_meta.get("same_notation",
                               self.source_format == self.target_format))

        self.blocks, self.outside, self.unclosed = extract_blocks(self.response)
        if len(self.blocks) == 1:
            self.body = self.blocks[0][1]
        elif self.blocks:
            tagged = [b for t, b in self.blocks if t == self.expected_tag]
            self.body = tagged[-1] if tagged else self.blocks[-1][1]
        else:
            self.body = (self.response or "").strip()

        self.pred_obj, self.pred_err = parse_as(self.target_format, self.body)
        if (self.family == "schema_gen" and isinstance(self.pred_obj, dict)
                and "$schema" in self.pred_obj):
            self.pred_obj = {k: v for k, v in self.pred_obj.items() if k != "$schema"}
        self.gold_obj = self._load_reference()
        self.input_text: str = (self.doc.get("inputs") or {}).get("input_data", "")
        self.coerce = self.target_format in TYPELESS_FORMATS
        self._diff: List[Tuple[str, str, str]] | None = None

    def _load_reference(self) -> Any:
        """The reference document, parsed with the target-format parser.

        Read from ``meta.reference`` so scoring survives the MERA convention of
        blanking ``outputs`` before a Hub upload; falls back to ``outputs`` when
        a local copy still carries it. The reference ships as the raw document
        rather than pre-parsed JSON so that both sides of every comparison go
        through the same parser — YAML and TOML admit non-string mapping keys,
        and a JSON round-trip would stringify them on one side only.
        """
        ref = (self.doc["meta"] or {}).get("reference")
        if ref not in (None, ""):
            obj, _ = parse_as(self.target_format, ref)
            if obj is not None:
                return obj
        out = self.doc.get("outputs")
        if out:
            obj, _ = parse_as(self.target_format, out)
            return obj
        return None

    @property
    def diffs(self) -> List[Tuple[str, str, str]]:
        if self._diff is None:
            if self.pred_obj is None or self.gold_obj is None:
                self._diff = []
            else:
                self._diff = diff_values(self.pred_obj, self.gold_obj,
                                         coerce_numeric=self.coerce)
        return self._diff

    def cat(self, kind: str) -> List[Tuple[str, str, str]]:
        return [d for d in self.diffs if d[0] == kind]


# --------------------------------------------------------------------------- #
# Constraint registry
# --------------------------------------------------------------------------- #

CheckFn = Callable[[Ctx], Tuple[bool, str]]


class Spec:
    """One registered constraint. Plain class for the reason given on Ctx."""

    __slots__ = ("cid", "layer", "applies", "fn")

    def __init__(self, cid: str, layer: str,
                 applies: Callable[[Ctx], bool], fn: CheckFn) -> None:
        self.cid, self.layer, self.applies, self.fn = cid, layer, applies, fn


REGISTRY: List[Spec] = []


def constraint(cid: str, layer: str,
               applies: Callable[[Ctx], bool] = lambda c: True):
    def deco(fn: CheckFn) -> CheckFn:
        REGISTRY.append(Spec(cid, layer, applies, fn))
        return fn
    return deco


def constraint_ids() -> List[str]:
    return [s.cid for s in REGISTRY]


# ------------------------------------------------------------------- pack.*

@constraint("pack.fence_present", "pack")
def _c_fence_present(c: Ctx) -> Tuple[bool, str]:
    return bool(c.blocks), "" if c.blocks else "no fenced block"


@constraint("pack.fence_exactly_one", "pack")
def _c_fence_one(c: Ctx) -> Tuple[bool, str]:
    if c.unclosed:
        return False, "unclosed fence"
    return len(c.blocks) == 1, f"blocks: {len(c.blocks)}"


@constraint("pack.fence_tag_correct", "pack", applies=lambda c: bool(c.blocks))
def _c_fence_tag(c: Ctx) -> Tuple[bool, str]:
    tags = [t for t, _ in c.blocks]
    return all(t == c.expected_tag for t in tags), f"tags={tags} expected={c.expected_tag!r}"


@constraint("pack.nothing_outside_fence", "pack")
def _c_outside(c: Ctx) -> Tuple[bool, str]:
    return not c.outside, f"outside the block: {c.outside[:60]!r}"


# -------------------------------------------------------------------- syn.*

@constraint("syn.parses_target", "syn")
def _c_parses(c: Ctx) -> Tuple[bool, str]:
    return c.pred_obj is not None, c.pred_err or ""


# -------------------------------------------------------------------- val.*

@constraint("val.structure_match", "val",
            applies=lambda c: c.pred_obj is not None and c.gold_obj is not None)
def _c_structure(c: Ctx) -> Tuple[bool, str]:
    bad = c.cat("structure")
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("val.numeric_within_tol", "val",
            applies=lambda c: c.pred_obj is not None and c.gold_obj is not None)
def _c_numeric(c: Ctx) -> Tuple[bool, str]:
    bad = c.cat("numeric")
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("val.strings_byte_exact", "val",
            applies=lambda c: c.pred_obj is not None and c.gold_obj is not None)
def _c_strings(c: Ctx) -> Tuple[bool, str]:
    bad = c.cat("string")
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


# ---------------------------------------------------------------- convert.*

@constraint("convert.lossless_vs_source", "family",
            applies=lambda c: c.family == "convert" and c.pred_obj is not None
            and c.source is not None
            and c.gates.get("convert.lossless_vs_source", False))
def _c_convert(c: Ctx) -> Tuple[bool, str]:
    """Checked against the SOURCE document, independently of the reference."""
    bad = diff_values(c.pred_obj, c.source, coerce_numeric=True)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


# ---------------------------------------------------------------- extract.*

@constraint("extract.keys_exact", "family",
            applies=lambda c: c.family == "extract" and isinstance(c.pred_obj, dict))
def _c_extract_keys(c: Ctx) -> Tuple[bool, str]:
    """The answer is keyed by the requested paths, written as they were given."""
    exp = set(c.task_meta.get("paths") or [])
    got = set(c.pred_obj.keys())
    return exp == got, f"extra={sorted(got - exp)[:3]} missing={sorted(exp - got)[:3]}"


@constraint("extract.is_object", "family",
            applies=lambda c: c.family == "extract" and c.pred_obj is not None)
def _c_extract_object(c: Ctx) -> Tuple[bool, str]:
    return isinstance(c.pred_obj, dict), f"type={type(c.pred_obj).__name__}"


@constraint("extract.absent_paths_null", "family",
            applies=lambda c: c.family == "extract" and isinstance(c.pred_obj, dict)
            and bool(c.task_meta.get("absent")))
def _c_extract_absent(c: Ctx) -> Tuple[bool, str]:
    """A path that does not resolve carries null, as the instruction says.

    Independent of the reference: the question names the rule, so this checks
    the rule. A model that quietly omits the entry, or invents a value for it,
    is caught here and the report names which path.
    """
    bad = [p for p in c.task_meta["absent"]
           if p not in c.pred_obj or c.pred_obj[p] is not None]
    return not bad, f"absent paths not reported as null: {bad[:3]}"


# ----------------------------------------------------------------- repair.*

@constraint("repair.parses_as_source", "family",
            applies=lambda c: c.family == "repair" and c.same_notation)
def _c_repair_parses(c: Ctx) -> Tuple[bool, str]:
    obj, err = parse_as(c.source_format, c.body)
    return obj is not None, err or ""


@constraint("repair.minimal_edit", "family",
            applies=lambda c: c.family == "repair" and c.same_notation
            and bool(c.input_text))
def _c_repair_minimal(c: Ctx) -> Tuple[bool, str]:
    """A repair stays close to the broken input.

    Catches rewriting the document from scratch, not subtle differences. The
    threshold sits well below the minimum observed on the reference answers
    (see ``validate_task.py``, which recomputes the margin).

    Only where the answer is in the input's own notation. The harder repair
    questions ask for the mended document in a different one, and there a
    similarity ratio against the input measures the distance between two
    notations rather than the size of the edit.
    """
    ratio = difflib.SequenceMatcher(None, c.input_text, c.body).ratio()
    return ratio >= 0.80, f"similarity to input={ratio:.3f} threshold=0.80"


# -------------------------------------------------------------- transform.*

@constraint("transform.rule_applied", "family",
            applies=lambda c: c.family == "transform" and bool(c.program)
            and c.pred_obj is not None)
def _c_transform_rule(c: Ctx) -> Tuple[bool, str]:
    """Every rule the question named left its mark on the answer.

    Checked as a property of the answer, not by re-running the rule, so it is
    independent of the reference — and a reference comparison could not see it
    anyway: the evaluator canonicalises key order, which is exactly what
    ``sort_keys`` is about. Where a question chains two rules, both promises
    are checked, and the chains are chosen so that the first rule's promise
    still holds after the second has run.
    """
    for step in c.program:
        fn = RULE_PROPERTY.get(step.get("rule", ""))
        if fn is None:
            return False, f"no verifier for rule {step.get('rule')!r}"
        ok, detail = fn(c.pred_obj, step.get("params") or {})
        if not ok:
            return False, f"{step['rule']}: {detail}"
    return True, ""


@constraint("transform.list_order_preserved", "family",
            applies=lambda c: c.family == "transform"
            and c.pred_obj is not None and c.gold_obj is not None)
def _c_transform_lists(c: Ctx) -> Tuple[bool, str]:
    bad = list_order_violations(c.pred_obj, c.gold_obj)
    return not bad, f"list order changed at: {bad[:3]}"


@constraint("transform.values_unchanged", "family",
            applies=lambda c: c.family == "transform"
            and c.pred_obj is not None and c.gold_obj is not None)
def _c_transform_values(c: Ctx) -> Tuple[bool, str]:
    a, b = leaf_multiset(c.pred_obj), leaf_multiset(c.gold_obj)
    if a == b:
        return True, ""
    only_p = [x for x in a if x not in b][:3]
    only_g = [x for x in b if x not in a][:3]
    return False, f"extra={only_p} lost={only_g}"


# ------------------------------------------------------------- schema_gen.*

@constraint("schema.required_sorted", "family",
            applies=lambda c: c.family == "schema_gen" and c.pred_obj is not None)
def _c_schema_required(c: Ctx) -> Tuple[bool, str]:
    stack, bad = [(c.pred_obj, "$")], []
    while stack:
        cur, path = stack.pop()
        if isinstance(cur, dict):
            req = cur.get("required")
            if isinstance(req, list) and req != sorted(req, key=str):
                bad.append(path)
            for k, v in cur.items():
                stack.append((v, f"{path}.{k}"))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                stack.append((v, f"{path}[{i}]"))
    return not bad, f"required not sorted at: {bad[:3]}"


@constraint("schema.validates_source", "family",
            applies=lambda c: c.family == "schema_gen" and c.pred_obj is not None
            and c.source is not None
            and c.gates.get("schema.validates_source", False))
def _c_schema_validates(c: Ctx) -> Tuple[bool, str]:
    """The produced schema must actually validate the source document."""
    try:
        import jsonschema
    except ImportError:
        return True, "jsonschema not installed — check skipped"
    try:
        jsonschema.validate(instance=c.source, schema=c.pred_obj)
        return True, ""
    except Exception as e:                   # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:120]}"


@constraint("schema.type_mapping_exact", "family",
            applies=lambda c: c.family == "schema_gen" and c.pred_obj is not None
            and c.source is not None)
def _c_schema_types(c: Ctx) -> Tuple[bool, str]:
    """Independent re-derivation of the schema from the source document."""
    bad = diff_values(c.pred_obj, infer_schema(c.source, c.schema_kind))
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("schema.no_extra_keywords", "family",
            applies=lambda c: c.family == "schema_gen" and c.pred_obj is not None)
def _c_schema_extra(c: Ctx) -> Tuple[bool, str]:
    """Only the keywords this kind of schema is allowed to use.

    The question says which ones those are, and they differ by kind: a
    ``bounds`` schema is supposed to carry ``minItems``, a ``minimal`` one is
    not, and putting them where they were not asked for adds a constraint the
    document does not carry.
    """
    allowed = SCHEMA_ALLOWED_KEYWORDS.get(c.schema_kind,
                                          SCHEMA_ALLOWED_KEYWORDS["minimal"])
    extra = sorted(collect_schema_keywords(c.pred_obj) - allowed)
    return not extra, f"disallowed keywords: {extra[:5]}"


# --------------------------------------------------------------------------- #
# Combinations
# --------------------------------------------------------------------------- #

COMBOS: Dict[str, List[str]] = {
    "format": ["pack.fence_present", "pack.fence_exactly_one",
               "pack.fence_tag_correct", "pack.nothing_outside_fence",
               "syn.parses_target"],
    "content": ["val.structure_match", "val.numeric_within_tol",
                "val.strings_byte_exact"],
    "task": [s.cid for s in REGISTRY if s.layer == "family"],
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def normalize_generation(generation: str) -> str:
    """Strip reasoning traces, then outer whitespace — nothing else.

    No lenient extraction beyond this: peeling off a preamble or unwrapping a
    stray fence would score a response the instruction did not ask for, and the
    packaging rules are part of what the benchmark measures.
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


def score_response(doc: Dict[str, Any], response: str) -> List[dict]:
    """One ``{id, layer, applicable, pass, detail}`` record per constraint."""
    ctx = Ctx(doc=doc, response=response)
    out: List[dict] = []
    for spec in REGISTRY:
        try:
            applicable = bool(spec.applies(ctx))
        except Exception:                    # noqa: BLE001
            applicable = False
        if not applicable:
            out.append({"id": spec.cid, "layer": spec.layer,
                        "applicable": False, "pass": True, "detail": "n/a"})
            continue
        try:
            ok, detail = spec.fn(ctx)
        except Exception as exc:             # noqa: BLE001 — a broken check must not abort a run
            eval_logger.warning("check %s failed on sample %s: %s",
                                spec.cid, doc["meta"].get("id"), exc)
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        out.append({"id": spec.cid, "layer": spec.layer,
                    "applicable": True, "pass": bool(ok), "detail": detail})
    return out


def _combo_pass(checks: List[dict], names: List[str]) -> float | None:
    """All applicable constraints of a combo hold; ``None`` if none applied.

    ``None`` rather than 1.0 on purpose. Every constraint in the value and family
    layers needs a parsed answer, so an unparseable one makes them all
    inapplicable — and scoring an empty combo as satisfied would hand a refusal
    full marks for content. The caller decides what an empty combo means.
    """
    rel = [c for c in checks if c["id"] in names and c["applicable"]]
    return float(all(c["pass"] for c in rel)) if rel else None


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Score one generation against the whole constraint stack of its question.

    An empty response never passes. Several constraints are vacuously true on an
    empty string (nothing outside the fence, no disallowed keywords), so without
    this guard a model that answered nothing would collect partial credit.
    """
    response = normalize_generation(_extract_prediction(results))
    checks = score_response(doc, response)
    applicable = [c for c in checks if c["applicable"]]
    if not applicable:                       # cannot happen with a well-formed doc
        eval_logger.warning("sample %s has no applicable constraints",
                            doc["meta"].get("id"))
        return {"sample_pass_rate": 0.0, "constraint_pass_rate": 0.0,
                "format_pass_rate": 0.0, "content_pass_rate": 0.0,
                "task_pass_rate": 0.0}

    if not response.strip():
        return {"sample_pass_rate": 0.0, "constraint_pass_rate": 0.0,
                "format_pass_rate": 0.0, "content_pass_rate": 0.0,
                "task_pass_rate": 0.0}

    n_pass = sum(1 for c in applicable if c["pass"])
    parsed = next(c["pass"] for c in checks if c["id"] == "syn.parses_target")

    # An answer that does not parse has neither correct content nor a satisfied
    # task requirement, whatever the empty combos would otherwise imply.
    content = 0.0 if not parsed else (_combo_pass(checks, COMBOS["content"]) or 0.0)
    if not parsed:
        task = 0.0
    else:
        task = _combo_pass(checks, COMBOS["task"])
        if task is None:
            # No family-level check applies to this question — the convert
            # questions whose source-derived gate is closed. There the family
            # requirement ("lossless conversion") is exactly "matches the
            # reference", so the content layer already states it.
            task = content

    return {
        "sample_pass_rate": float(n_pass == len(applicable)),
        "constraint_pass_rate": n_pass / len(applicable),
        "format_pass_rate": _combo_pass(checks, COMBOS["format"]) or 0.0,
        "content_pass_rate": content,
        "task_pass_rate": task,
    }


# --------------------------------------------------------------------------- #
# Prompt construction (local JSON files only)
# --------------------------------------------------------------------------- #

_PROMPTS_CACHE: List[str] | None = None
_META_CANDIDATES = (
    "../../datasets/SOBHard/dataset_meta.json",
    "../../../datasets/SOBHard/dataset_meta.json",
)


def _load_prompts() -> List[str]:
    """Prompt templates from ``dataset_meta.json`` (repo-local runs only).

    On the Hugging Face copy ``instruction`` already holds the prompt text, so
    this is never called there.
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
        "instruction is an index but dataset_meta.json was not found; expected "
        "one of: " + ", ".join(_META_CANDIDATES)
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    instruction = doc["instruction"]
    if isinstance(instruction, int):
        instruction = _load_prompts()[instruction]
    return instruction.format(**doc["inputs"])


def _resolve_instruction(doc: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(doc.get("instruction"), int):
        doc["instruction"] = _load_prompts()[doc["instruction"]]
    return doc


def process_docs(dataset):
    """Replace prompt indices with prompt text (local ``datasets/`` JSON only).

    Resolved up front rather than inside ``doc_to_text`` so the few-shot path
    works too: the sampler swaps ``doc_to_text`` for the Jinja ``query``
    template, which calls ``instruction.format`` directly.
    """
    return dataset.map(_resolve_instruction)
