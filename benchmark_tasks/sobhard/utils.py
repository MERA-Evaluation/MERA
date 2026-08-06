"""SOBHard2 — deterministic structured-output scoring for lm-evaluation-harness.

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
This module imports nothing from the SOBHard2 generator. Checks that need the
*source* document in the pipeline's canonical shape (which requires parsers for
twenty exotic input formats) are precomputed at build time and shipped inside
``meta.checks``; here they are plain JSON comparisons. Only the seven *target*
formats are parsed at scoring time, and those parsers are byte-compatible ports
of the generator's own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from configparser import ConfigParser, Error as CFGError, MissingSectionHeaderError
from typing import Any, Callable, Dict, List, Tuple
from lm_eval.api.registry import register_filter, FILTER_REGISTRY
from lm_eval.api.filter import Filter

eval_logger = logging.getLogger(__name__)

RTOL = 1e-6

#: Formats with no type system: every value is a string after parsing, so the
#: promised 1e-6 numeric tolerance only applies with coercion.
TYPELESS_FORMATS = {"ini", "csv", "tsv", "properties", "sqlinsert"}

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
    "toml": "toml", "ini": "ini", "csv": "csv", "tsv": "tsv",
    "properties": "properties", "sqlinsert": "sql",
}

#: Reasoning traces are scaffolding, not the answer. Providers that keep the
#: chain in its own response field never reach this code — lm-eval reads
#: ``message.content`` and the relay is a pass-through — so what is handled
#: here is the other habit: a model that writes its thinking inline, in front
#: of the answer, inside a tag.
#:
#: The tag names are the ones seen in the wild. They are matched case
#: insensitively and only OUTSIDE fenced blocks (see ``normalize_generation``):
#: a document being converted can legitimately contain the characters
#: ``<think>``, and eating them out of the answer would fail a correct model
#: for the scorer's reason.
_THINK_NAMES = "think|thinking|reason|reasoning|thought|scratchpad"
_THINK_BLOCK_RE = re.compile(rf"<({_THINK_NAMES})>[\s\S]*?</\1>", re.IGNORECASE)
_THINK_TAG_RE = re.compile(rf"</?({_THINK_NAMES})>", re.IGNORECASE)


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
                "SOBHard2 needs a TOML parser for the 50 questions whose target "
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


def _parse_table(text: str, delimiter: str) -> List[Dict[str, str]]:
    """CSV/TSV under the convention the prompt states — a byte-compatible port
    of the generator's reader. The first row names the columns, every later
    row is one record, all values are strings. Quoting is the standard one:
    a quoted field may hold the delimiter, doubled quotes and line breaks."""
    import csv
    import io

    s = _strip_code_fence(text)
    if not s.strip():
        raise FormatError("empty input")
    rows = list(csv.reader(io.StringIO(s), delimiter=delimiter))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        raise FormatError("a table needs a header row and at least one record")
    header = rows[0]
    if any(not h.strip() for h in header):
        raise FormatError("blank column name in the header")
    if len(set(header)) != len(header):
        raise FormatError("duplicate column name in the header")
    out: List[Dict[str, str]] = []
    for i, r in enumerate(rows[1:], 2):
        if len(r) != len(header):
            raise FormatError(f"row {i} has {len(r)} cells for "
                              f"{len(header)} columns")
        out.append(dict(zip(header, r)))
    return out


def _parse_csv(text: str) -> List[Dict[str, str]]:
    return _parse_table(text, ",")


def _parse_tsv(text: str) -> List[Dict[str, str]]:
    return _parse_table(text, "\t")


def _parse_properties(text: str) -> Dict[str, str]:
    """``.properties`` under the convention the prompt states — a port of the
    generator's reader. Flat ``key = value`` lines, values are strings, ``#``
    and ``!`` open comments, the first ``=`` splits, edges are trimmed."""
    s = _strip_code_fence(text)
    if not s.strip():
        raise FormatError("empty input")
    out: Dict[str, str] = {}
    for i, raw in enumerate(s.split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if "=" not in line:
            raise FormatError(f"line {i} has no '='")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise FormatError(f"line {i} has an empty key")
        out[key] = value.strip()
    if not out:
        raise FormatError("no entries")
    return out


_SQL_TABLE = "records"
_SQL_STMT = re.compile(
    r"^INSERT\s+INTO\s+" + _SQL_TABLE + r"\s*\((?P<cols>[^)]*)\)\s*"
    r"VALUES\s*\((?P<vals>.*)\)\s*;$", re.IGNORECASE | re.DOTALL)


def _sql_values(text: str) -> List[Any]:
    """One VALUES list, read by the stated rule: quoted strings with doubled
    quotes inside, or a bare NULL, separated by commas."""
    out: List[Any] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == "'":
            i += 1
            buf: List[str] = []
            closed = False
            while i < n:
                if text[i] == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        buf.append("'")
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                buf.append(text[i])
                i += 1
            if not closed:
                raise FormatError("unterminated string literal")
            out.append("".join(buf))
        else:
            start = i
            while i < n and text[i] != ",":
                i += 1
            token = text[start:i].strip()
            if token.upper() == "NULL":
                out.append(None)
            else:
                raise FormatError(f"value {token!r} is neither quoted nor NULL")
        while i < n and text[i].isspace():
            i += 1
        if i < n:
            if text[i] != ",":
                raise FormatError("values are separated by commas")
            i += 1
    return out


def _parse_sqlinsert(text: str) -> List[Dict[str, Any]]:
    """The INSERT dialect the question states, read back into records.

    Written here from the prompt's four sentences, independently of the writer
    that produced the reference — the same trust model as every other notation
    in this file. Every value comes back a string except NULL, which comes back
    as null, and every statement has to list the same columns as the first.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        raise FormatError("no INSERT statements")
    rows: List[Dict[str, Any]] = []
    columns: List[str] | None = None
    for i, line in enumerate(lines, 1):
        m = _SQL_STMT.match(line)
        if not m:
            raise FormatError(f"line {i}: not an INSERT of the declared form")
        cols = [c.strip() for c in m.group("cols").split(",")]
        if any(not c for c in cols) or len(set(cols)) != len(cols):
            raise FormatError(f"line {i}: blank or duplicate column name")
        if columns is None:
            columns = cols
        elif cols != columns:
            raise FormatError(f"line {i}: columns differ from the first row")
        values = _sql_values(m.group("vals"))
        if len(values) != len(cols):
            raise FormatError(f"line {i}: {len(values)} values for "
                              f"{len(cols)} columns")
        rows.append(dict(zip(cols, values)))
    return rows


_PARSERS: Dict[str, Callable[[str], Any]] = {
    "json": _parse_json, "jsonl": _parse_jsonl, "yaml": _parse_yaml,
    "toml": _parse_toml, "ini": _parse_ini, "csv": _parse_csv,
    "tsv": _parse_tsv, "properties": _parse_properties,
    "sqlinsert": _parse_sqlinsert,
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


def check_dependencies(formats: Any = ("json", "jsonl", "yaml", "toml", "ini",
                                       "csv", "tsv", "properties")) -> None:
    """Fail fast if a format used by the dataset has no parser installed."""
    missing = []
    probe = {"json": "{}", "jsonl": "{}", "yaml": "a: 1", "toml": "a = 1",
             "ini": "[s]\na = 1", "csv": "a,b\n1,2", "tsv": "a\tb\n1\t2",
             "properties": "a = 1"}
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


def _string_mismatch(pred: str, gold: str, window: int = 26) -> str:
    """Two differing strings, shown around the first place they differ.

    Truncating both from the start is worse than useless once the answers are
    long: a `template` question whose rendered lines share a forty-character
    prefix reported `'wikidata id=https://www.wikidata.org/wik' != 'wikidata
    id=https://www.wikidata.org/wik'` — two identical-looking strings and no
    way to see what broke. Found by reading failures by hand.
    """
    i = 0
    while i < min(len(pred), len(gold)) and pred[i] == gold[i]:
        i += 1
    lo = max(0, i - window // 2)
    head = "…" if lo else ""
    def cut(s: str) -> str:
        piece = s[lo:i + window]
        return f"{head}{piece}{'…' if len(s) > i + window else ''}"
    where = f" at char {i}" if i else ""
    return f"{cut(pred)!r} != {cut(gold)!r}{where}"


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
            # key=str: a YAML answer can carry non-string mapping keys (`80:`
            # parses as an integer), and a bare sorted() over the mixed set
            # raises instead of reporting. The verdict is the same either
            # way — the key sets differ — but the report must say so.
            miss = sorted(set(gold) - set(pred), key=str)
            extra = sorted(set(pred) - set(gold), key=str)
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
            out.append(("string", path, _string_mismatch(pred, gold)))
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
            return False, f"record keys {sorted(r.keys(), key=str)[:5]} != {sorted(want)}"
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


def _p_dedup(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    field = p["field"]
    if not any(field in r for r in records):
        return True, "the deduplicated field is not in the answer any more"
    seen: set = set()
    for r in records:
        key = repr(r.get(field))
        if key in seen:
            return False, f"duplicate {field}={r.get(field)!r} survived"
        seen.add(key)
    return True, ""


def _p_renumber(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    field = p["field"]
    for i, r in enumerate(records, 1):
        if r.get(field) != i or isinstance(r.get(field), bool):
            return False, f"record {i} carries {field}={r.get(field)!r}"
    return True, ""


def _p_count_by(pred: Any, p: dict) -> Tuple[bool, str]:
    if not isinstance(pred, dict):
        return False, f"counting gives an object, got {type(pred).__name__}"
    ks = list(pred.keys())
    if ks != sorted(ks, key=str):
        return False, "count keys are not in lexicographic order"
    bad = [k for k, v in pred.items()
           if isinstance(v, bool) or not isinstance(v, int) or v < 1]
    return not bad, f"counts that are not positive integers: {bad[:3]}"


def _p_top_n(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    field, n = p["field"], p["n"]
    if not any(field in r for r in records):
        return True, "the grouping field is not in the answer any more"
    counts: Dict[str, int] = {}
    for r in records:
        key = str(r.get(field))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > n:
            return False, f"group {key!r} kept more than {n} records"
    return True, ""


def _p_pivot(pred: Any, p: dict) -> Tuple[bool, str]:
    if not isinstance(pred, dict):
        return False, f"a pivot is an object, got {type(pred).__name__}"
    ks = list(pred.keys())
    if ks != sorted(ks, key=str):
        return False, "outer keys are not in lexicographic order"
    for k, v in pred.items():
        if not isinstance(v, dict):
            return False, f"{k!r} does not hold an object"
        inner = list(v.keys())
        if inner != sorted(inner, key=str):
            return False, f"inner keys of {k!r} are not in order"
        nested = [ik for ik, iv in v.items() if isinstance(iv, (dict, list))]
        if nested:
            return False, f"{k!r} still holds containers at {nested[:2]}"
    return True, ""


def _p_melt(pred: Any, p: dict) -> Tuple[bool, str]:
    records = _records_in(pred)
    if records is None:
        return False, f"the answer is not a list of records ({type(pred).__name__})"
    fields = p["fields"]
    k = len(fields)
    if len(records) % k != 0:
        return False, f"{len(records)} records do not divide into blocks of {k}"
    for i, r in enumerate(records):
        if "metric" not in r or "value" not in r:
            return False, f"record {i} lacks metric or value"
        if r["metric"] != fields[i % k]:
            return False, (f"record {i} carries metric={r['metric']!r} where "
                           f"{fields[i % k]!r} was due")
    return True, ""


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
    "dedup_records": _p_dedup,
    "renumber": _p_renumber,
    "count_by": _p_count_by,
    "top_n_per_group": _p_top_n,
    "pivot_levels": _p_pivot,
    "melt": _p_melt,
}

#: Rules that reshape the document so radically that the promises of the
#: steps before them stop being observable in the answer: a count is not a
#: record list, and a melt multiplies every row. The rule check starts from
#: the last of these; what came before is carried by the value comparison
#: with the reference and by the family's own recomputation.
RESHAPERS = ("count_by", "melt")


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

        # A dialogue turn is an ordinary question of an ordinary family, asked
        # over the previous turn's result instead of over a document that is in
        # the prompt. The work it asks for is the same work, so it is checked by
        # the same family constraints: the turn reports the family of its own
        # operation, and ``is_session`` carries what is specific to the dialogue.
        self.session: dict = self.task_meta.get("session") or {}
        self.is_session: bool = self.family == "session"
        if self.is_session:
            self.family = self.task_meta.get("turn_family", "")

        checks = _meta_json(self.doc, "checks", {})
        # The input document, read under the convention the prompt states, so
        # the family checks can derive what they need from the source instead of
        # taking a second look at the reference.
        self.source = checks.get("source", checks.get("source_canonical"))
        # The second document of a two-document question: version 2 for diff,
        # document B for merge. Absent everywhere else.
        self.source_b = checks.get("source_b")
        # The state the conversation was in immediately before this turn. It is
        # the same as ``source`` everywhere except an undoing turn, where the
        # work restarts from the state two turns back — so the document that
        # was on the table and the document the operation applies to differ.
        self.prev_state = checks.get("prev_state", self.source)
        # On a question that composes operations, the document after the
        # rewrite steps: what a schema must describe, what a repair's data
        # must match. Precomputed at build time and re-derived by
        # validate_task from the source with the generator's own rules.
        self.derived = checks.get("derived") if "derived" in checks else None
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
    return exp == got, f"extra={sorted(got - exp, key=str)[:3]} missing={sorted(exp - got, key=str)[:3]}"


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

@constraint("repair.data_intact", "family",
            applies=lambda c: c.family == "repair" and c.pred_obj is not None
            and (c.derived is not None or c.source is not None)
            and c.gates.get("repair.data_intact", False))
def _c_repair_intact(c: Ctx) -> Tuple[bool, str]:
    """The mended document carries the source's data — checked against the
    structure the broken input encodes, independently of the reference. On a
    repair that also rewrites, the comparison target is the document after the
    rewrite steps, shipped as ``checks.derived``."""
    base = c.derived if c.derived is not None else c.source
    bad = diff_values(c.pred_obj, base, coerce_numeric=True)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


# -------------------------------------------------------------- transform.*

#: Families whose *answer* is the rewritten document itself, so each rule's
#: promise is observable in it. schema_gen composes rules too, but its answer
#: is a schema — there the rules are checked through the schema derivation.
#: merge and patch join the list for their composed questions: a rewrite
#: chained onto the join or onto the edits is observable in the answer the
#: same way.
_PROGRAM_FAMILIES = ("transform", "repair", "extract", "merge", "patch")


@constraint("transform.rule_applied", "family",
            applies=lambda c: c.family in _PROGRAM_FAMILIES and bool(c.program)
            and c.pred_obj is not None)
def _c_transform_rule(c: Ctx) -> Tuple[bool, str]:
    """Every rule the question named left its mark on the answer.

    Checked as a property of the answer, not by re-running the rule, so it is
    independent of the reference — and a reference comparison could not see it
    anyway: the evaluator canonicalises key order, which is exactly what
    ``sort_keys`` is about. Where a question chains rules, every promise is
    checked, and the chains are chosen so that each rule's promise still holds
    after the later ones have run — except across a reshaping step
    (``count_by``, ``melt``), after which the earlier list-shaped promises
    are no longer observable; the check starts from the last such step. The
    same check serves the repair and extract questions that carry a rewrite
    step.
    """
    steps = c.program
    rules_list = [s.get("rule") for s in steps]
    start = 0
    for reshaper in RESHAPERS:
        if reshaper in rules_list:
            start = max(start, rules_list.index(reshaper))
    for step in steps[start:]:
        fn = RULE_PROPERTY.get(step.get("rule", ""))
        if fn is None:
            return False, f"no verifier for rule {step.get('rule')!r}"
        ok, detail = fn(c.pred_obj, step.get("params") or {})
        if not ok:
            return False, f"{step['rule']}: {detail}"
    return True, ""


@constraint("transform.list_order_preserved", "family",
            applies=lambda c: c.family in _PROGRAM_FAMILIES and bool(c.program)
            and c.pred_obj is not None and c.gold_obj is not None)
def _c_transform_lists(c: Ctx) -> Tuple[bool, str]:
    bad = list_order_violations(c.pred_obj, c.gold_obj)
    return not bad, f"list order changed at: {bad[:3]}"


@constraint("transform.values_unchanged", "family",
            applies=lambda c: c.family in _PROGRAM_FAMILIES and bool(c.program)
            and c.pred_obj is not None and c.gold_obj is not None)
def _c_transform_values(c: Ctx) -> Tuple[bool, str]:
    a, b = leaf_multiset(c.pred_obj), leaf_multiset(c.gold_obj)
    if a == b:
        return True, ""
    only_p = [x for x in a if x not in b][:3]
    only_g = [x for x in b if x not in a][:3]
    if only_p or only_g:
        return False, f"extra={only_p} lost={only_g}"
    # Same values, different counts. Set arithmetic sees nothing here, and
    # reporting "extra=[] lost=[]" is a check that fails without saying why —
    # which this project treats as a defect. Found by reading failures by hand
    # on a run where a model had duplicated one record and dropped another.
    from collections import Counter as _Counter
    ca, cb = _Counter(a), _Counter(b)
    moved = [(v, ca[v], cb[v]) for v in sorted(set(ca) | set(cb))
             if ca[v] != cb[v]][:3]
    detail = "; ".join(f"{v}: {p} of them, expected {g}" for v, p, g in moved)
    return False, f"the same values in different numbers — {detail}"


# ------------------------------------------------------------------ patch.*

_PATH_STEP = re.compile(r"\[(\d+)\]|([^.\[\]]+)")


def _path_steps(path: str) -> List[Any]:
    return [int(a) if a else b for a, b in _PATH_STEP.findall(path)]


def apply_patch_ops(source: Any, ops: List[dict]) -> Any:
    """The patch dialect the question states, replayed over the source.

    Written here from the prompt's own three sentences rather than shared with
    the generator, so a mistake on either side surfaces as a disagreement the
    validator catches instead of hiding in common code. Operations apply in
    order; list indices address the current state; ``set`` on a key the parent
    object lacks adds the entry; ``delete`` shifts a list's tail left;
    ``insert`` shifts it right.
    """
    import copy as _copy

    cur = _copy.deepcopy(source)
    for i, op in enumerate(ops, 1):
        steps = _path_steps(op.get("path") or "")
        if not steps:
            raise ValueError(f"op {i}: empty path")
        node = cur
        for step in steps[:-1]:
            node = node[step]                    # KeyError/IndexError = bad op
        last = steps[-1]
        kind = op.get("op")
        if kind == "set":
            node[last] = _copy.deepcopy(op.get("value"))
        elif kind == "delete":
            del node[last]
        elif kind == "insert":
            if not isinstance(node, list) or not isinstance(last, int):
                raise ValueError(f"op {i}: insert needs a list index")
            node.insert(last, _copy.deepcopy(op.get("value")))
        else:
            raise ValueError(f"op {i}: unknown operation {kind!r}")
    return cur


# --------------------------------------------------------------- template.*

TEMPLATE_ESCAPE = "$${"


def render_template(template: str, record: Any) -> str:
    """The template dialect the question states, rendered independently.

    Written from the prompt's three sentences, not shared with the generator:
    ``${path}`` takes the value at that path written as the document writes it,
    a path the record does not have gives the empty string, ``$${`` is a
    literal ``${``, and everything else is text.
    """
    out: List[str] = []
    i = 0
    while i < len(template):
        if template.startswith(TEMPLATE_ESCAPE, i):
            out.append("${")
            i += len(TEMPLATE_ESCAPE)
            continue
        if template.startswith("${", i):
            end = template.find("}", i)
            if end == -1:
                out.append(template[i:])
                break
            node: Any = record
            for step in _path_steps(template[i + 2:end]):
                try:
                    node = node[step]
                except (KeyError, IndexError, TypeError):
                    node = _ABSENT
                    break
            if node is _ABSENT:
                pass                              # a missing path renders empty
            elif isinstance(node, bool):
                out.append("true" if node else "false")
            elif node is None:
                out.append("null")
            elif isinstance(node, (dict, list)):
                raise ValueError("a placeholder resolved to a container")
            else:
                out.append(str(node))
            i = end + 1
            continue
        out.append(template[i])
        i += 1
    return "".join(out)


_ABSENT = object()


def _template_records(c: "Ctx") -> Any:
    """The records the template is rendered over — after a rewrite if any."""
    return c.derived if c.derived is not None else c.source


@constraint("template.record_count", "family",
            applies=lambda c: c.family == "template" and c.pred_obj is not None
            and _template_records(c) is not None)
def _c_template_count(c: Ctx) -> Tuple[bool, str]:
    """One string per record, and nothing but strings.

    Separate from the exactness check because it fails differently: a model
    that merged two records, dropped the last one or answered with one blob of
    text is wrong in a way worth naming on its own.
    """
    records = _template_records(c)
    if not isinstance(c.pred_obj, list):
        return False, "the answer is not a list"
    if not all(isinstance(s, str) for s in c.pred_obj):
        return False, "the list holds something other than strings"
    want = len(records) if isinstance(records, list) else 1
    return len(c.pred_obj) == want, \
        f"{len(c.pred_obj)} strings for an expected {want}"


@constraint("template.no_placeholder_left", "family",
            applies=lambda c: c.family == "template"
            and isinstance(c.pred_obj, list)
            and not c.task_meta.get("has_escape"))
def _c_template_left(c: Ctx) -> Tuple[bool, str]:
    """No ``${`` survives into the answer.

    Only asked where the template carries no literal-``${`` escape: with the
    escape a ``${`` in the answer is correct, and a check that cannot tell the
    two apart would be stricter than the prompt.
    """
    left = [s for s in c.pred_obj if isinstance(s, str) and "${" in s]
    return not left, f"{len(left)} strings still hold a placeholder: {left[:2]}"


@constraint("template.rendered_exact", "family",
            applies=lambda c: c.family == "template" and c.pred_obj is not None
            and c.source_b is not None and _template_records(c) is not None)
def _c_template_exact(c: Ctx) -> Tuple[bool, str]:
    """Every line re-rendered here from the records and the template."""
    records = _template_records(c)
    if not isinstance(records, list):
        return False, "the document is not a list of records"
    try:
        expected = [render_template(str(c.source_b), r) for r in records]
    except Exception as exc:                      # noqa: BLE001
        return False, f"the template does not render: {exc}"
    bad = diff_values(c.pred_obj, expected, coerce_numeric=c.coerce)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


# -------------------------------------------------------------- applydiff.*

def apply_doc_diff(source: Any, diff: Any) -> Any:
    """The change object the question states, applied to version one.

    Written here from the prompt's own sentences rather than shared with the
    generator, for the same reason the patch replay is: two independent
    readings of the dialect that have to agree, and a validator line that fails
    when they do not.

    A path is a position, not a step in a sequence: ``removed`` and ``changed``
    name positions in version one, ``added`` names positions in version two. So
    a list's removals are taken from the highest index down and its additions
    from the lowest index up, and every path lands where it says it does.
    Applying the entries one at a time in printed order is the natural wrong
    answer, and it is a different document.
    """
    import copy as _copy

    if not isinstance(diff, dict):
        raise ValueError("the change object is not an object")
    cur = _copy.deepcopy(source)

    def parent_of(path: str):
        steps = _path_steps(str(path))
        if not steps:
            raise ValueError(f"empty path {path!r}")
        node = cur
        for step in steps[:-1]:
            node = node[step]
        return node, steps[-1]

    def numbered(paths, reverse: bool):
        """Paths grouped so list positions are taken in a safe order."""
        keyed = []
        for p in paths:
            steps = _path_steps(str(p))
            if not steps:
                raise ValueError(f"empty path {p!r}")
            last = steps[-1]
            keyed.append(((0, last) if isinstance(last, int) else (1, 0), p))
        keyed.sort(key=lambda it: (it[0][0], it[0][1]), reverse=reverse)
        return [p for _, p in keyed]

    removed = diff.get("removed") or {}
    changed = diff.get("changed") or {}
    added = diff.get("added") or {}
    for name, section in (("removed", removed), ("changed", changed),
                          ("added", added)):
        if not isinstance(section, dict):
            raise ValueError(f"{name} is not an object")

    for path in numbered(removed, reverse=True):
        node, last = parent_of(path)
        del node[last]
    for path, value in changed.items():
        node, last = parent_of(path)
        if not isinstance(value, dict) or DIFF_NEW not in value:
            raise ValueError(f"changed {path!r}: no {DIFF_NEW!r}")
        node[last] = _copy.deepcopy(value[DIFF_NEW])
    for path in numbered(added, reverse=False):
        node, last = parent_of(path)
        if isinstance(last, int):
            if not isinstance(node, list):
                raise ValueError(f"added {path!r}: parent is not a list")
            node.insert(last, _copy.deepcopy(added[path]))
        else:
            node[last] = _copy.deepcopy(added[path])
    return cur


@constraint("applydiff.result_exact", "family",
            applies=lambda c: c.family == "apply_diff" and c.pred_obj is not None
            and c.source is not None and c.source_b is not None)
def _c_apply_diff(c: Ctx) -> Tuple[bool, str]:
    """Version two, recomputed here from version one and the change object.

    The target is derived rather than read off the reference, so an answer that
    applied the sections in another order — the one thing the family's rule is
    about — fails against this replay with the first difference named. Where a
    rewrite runs over the result, the comparison target is ``checks.derived``,
    recomputed by the validator as rules(apply(v1, diff)).
    """
    if c.derived is not None:
        expected = c.derived
    else:
        try:
            expected = apply_doc_diff(c.source, c.source_b)
        except Exception as exc:                  # noqa: BLE001
            return False, f"the change object does not apply: {exc}"
    bad = diff_values(c.pred_obj, expected, coerce_numeric=c.coerce)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("patch.ops_applied", "family",
            applies=lambda c: c.family == "patch" and c.pred_obj is not None
            and c.source is not None and bool(c.task_meta.get("ops")))
def _c_patch_ops(c: Ctx) -> Tuple[bool, str]:
    """Independent replay of the operation list over the source document.

    The comparison target is recomputed here, not read from the reference, so
    a model that applied the operations against the original indices instead
    of the shifted ones — or skipped one — fails against the replay, with the
    first difference named. On the composed questions that rewrite the edited
    document afterwards, the target is ``checks.derived`` — which the
    validator recomputes as rules(replay(source)) with the generator's own
    machinery, the same trust model every composed family uses.
    """
    if c.derived is not None:
        expected = c.derived
    else:
        try:
            expected = apply_patch_ops(c.source, c.task_meta["ops"])
        except Exception as e:                   # noqa: BLE001
            return False, f"the operation list does not replay: {e}"
    bad = diff_values(c.pred_obj, expected, coerce_numeric=c.coerce)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


# ------------------------------------------------------------------ merge.*

def merge_join(a_records: List[dict], b_records: List[dict],
               params: Dict[str, str]) -> List[dict]:
    """The join the merge question declares, replayed from its parameters.

    Independent of the generator: written from the four stated choices —
    which side wins a shared field name, what happens to an unpaired A
    record, to an unpaired B record, and whose order the result follows.
    Pairs match on equality of the key field's value as read, no coercion:
    the documents are built so both sides hold the key as a string.
    """
    key = params["key"]

    def merged(a: dict, b: dict) -> dict:
        if params["winner"] == "B":
            return {**a, **{k: v for k, v in b.items()}}
        return {**a, **{k: v for k, v in b.items() if k not in a}}

    out: List[dict] = []
    if params["order"] == "A":
        first_b: Dict[Any, dict] = {}
        for r in b_records:
            first_b.setdefault(r.get(key), r)
        for a in a_records:
            b = first_b.get(a.get(key))
            if b is not None:
                out.append(merged(a, b))
            elif params["unmatched_a"] == "keep":
                out.append(dict(a))
        if params["unmatched_b"] == "append":
            seen_a = {a.get(key) for a in a_records}
            out += [dict(b) for b in b_records if b.get(key) not in seen_a]
    else:
        first_a: Dict[Any, dict] = {}
        for r in a_records:
            first_a.setdefault(r.get(key), r)
        for b in b_records:
            a = first_a.get(b.get(key))
            if a is not None:
                out.append(merged(a, b))
            elif params["unmatched_b"] == "append":
                out.append(dict(b))
        if params["unmatched_a"] == "keep":
            seen_b = {b.get(key) for b in b_records}
            out += [dict(a) for a in a_records if a.get(key) not in seen_b]
    return out


def _merge_expected(c: "Ctx") -> Any:
    """What the merge answer must be: the recomputed join — and, when the
    question chains a rewrite onto it, the shipped ``checks.derived``, which
    the validator recomputes from both sources with the generator's rules."""
    if c.derived is not None:
        return c.derived
    return merge_join(c.source, c.source_b, c.task_meta["join"])


@constraint("merge.join_exact", "family",
            applies=lambda c: c.family == "merge" and c.pred_obj is not None
            and c.source is not None and c.source_b is not None
            and bool(c.task_meta.get("join")))
def _c_merge_join(c: Ctx) -> Tuple[bool, str]:
    try:
        expected = _merge_expected(c)
    except Exception as e:                       # noqa: BLE001
        return False, f"the join does not replay: {e}"
    bad = diff_values(c.pred_obj, expected, coerce_numeric=c.coerce)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("merge.record_count", "family",
            applies=lambda c: c.family == "merge" and c.pred_obj is not None
            and c.source is not None and c.source_b is not None
            and bool(c.task_meta.get("join")))
def _c_merge_count(c: Ctx) -> Tuple[bool, str]:
    """The cheapest named symptom: a lost or invented record, reported as a
    count so the diagnosis is readable even when the full comparison drowns
    in field-level noise."""
    try:
        expected = _merge_expected(c)
    except Exception as e:                       # noqa: BLE001
        return False, f"the join does not replay: {e}"
    got = c.pred_obj if isinstance(c.pred_obj, list) else None
    if got is None:
        return False, f"the answer is {type(c.pred_obj).__name__}, not a list"
    return len(got) == len(expected), \
        f"{len(got)} records for an expected {len(expected)}"


# ------------------------------------------------------------------- diff.*

#: The changed-entry field names the diff question states.
DIFF_OLD, DIFF_NEW = "было", "стало"
DIFF_SECTIONS = ("added", "removed", "changed")


def compute_doc_diff(v1: Any, v2: Any) -> Dict[str, Dict[str, Any]]:
    """The comparison the diff question describes, replayed independently.

    Written from the prompt's own sentences, not shared with the generator:
    objects compare entry by entry, lists element by element by index, and a
    path where the values differ with a scalar (or an object against a list)
    on either side becomes one ``changed`` entry carrying both whole values.
    Type-strict on scalars — ``true`` is not ``1`` and ``"5"`` is not ``5`` —
    matching the byte-exactness the rest of the benchmark promises.
    """
    added: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    changed: Dict[str, Any] = {}
    stack: List[Tuple[Any, Any, str]] = [(v1, v2, "")]
    while stack:
        a, b, path = stack.pop()
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in a.items():
                sub = f"{path}.{k}" if path else str(k)
                if k in b:
                    stack.append((v, b[k], sub))
                else:
                    removed[sub] = v
            for k, v in b.items():
                if k not in a:
                    added[f"{path}.{k}" if path else str(k)] = v
            continue
        if isinstance(a, list) and isinstance(b, list):
            shared = min(len(a), len(b))
            for i in range(shared):
                stack.append((a[i], b[i], f"{path}[{i}]"))
            for i in range(shared, len(a)):
                removed[f"{path}[{i}]"] = a[i]
            for i in range(shared, len(b)):
                added[f"{path}[{i}]"] = b[i]
            continue
        if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
            same_kind = (isinstance(a, bool) == isinstance(b, bool)) and (
                type(a) is type(b)
                or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and not isinstance(a, bool) and not isinstance(b, bool)))
            if same_kind and a == b:
                continue
        changed[path] = {DIFF_OLD: a, DIFF_NEW: b}
    return {"added": dict(sorted(added.items())),
            "removed": dict(sorted(removed.items())),
            "changed": dict(sorted(changed.items()))}


@constraint("diff.matches_pair", "family",
            applies=lambda c: c.family == "diff" and c.pred_obj is not None
            and c.source is not None and c.source_b is not None)
def _c_diff_matches(c: Ctx) -> Tuple[bool, str]:
    """The answer is exactly the diff of the two shipped versions, recomputed
    here from ``checks.source`` and ``checks.source_b`` — never read off the
    reference."""
    expected = compute_doc_diff(c.source, c.source_b)
    bad = diff_values(c.pred_obj, expected)
    return not bad, "; ".join(f"{p}: {d}" for _, p, d in bad[:3])


@constraint("diff.top_level_form", "family",
            applies=lambda c: c.family == "diff" and c.pred_obj is not None)
def _c_diff_form(c: Ctx) -> Tuple[bool, str]:
    """Exactly the three declared sections, in the declared order, all
    present — the insertion order the model actually emitted, which
    ``json.loads`` and the YAML parser both preserve."""
    if not isinstance(c.pred_obj, dict):
        return False, f"the answer is {type(c.pred_obj).__name__}, not an object"
    ks = list(c.pred_obj.keys())
    return ks == list(DIFF_SECTIONS), f"top-level keys: {ks}"


@constraint("diff.sections_sorted", "family",
            applies=lambda c: c.family == "diff" and isinstance(c.pred_obj, dict))
def _c_diff_sorted(c: Ctx) -> Tuple[bool, str]:
    """Inside each section the paths appear in character-code order."""
    for sec in DIFF_SECTIONS:
        node = c.pred_obj.get(sec)
        if not isinstance(node, dict):
            continue
        ks = list(node.keys())
        if ks != sorted(ks, key=str):
            bad = next((k for a, k in zip(ks, sorted(ks, key=str)) if a != k),
                       "?")
            return False, f"{sec}: order breaks at {bad!r}"
    return True, ""


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
            and (c.derived is not None or c.source is not None)
            and c.gates.get("schema.validates_source", False))
def _c_schema_validates(c: Ctx) -> Tuple[bool, str]:
    """The produced schema must actually validate the document it describes —
    the source, or, where the question rewrites first, the rewritten one."""
    try:
        import jsonschema
    except ImportError:
        return True, "jsonschema not installed — check skipped"
    base = c.derived if c.derived is not None else c.source
    try:
        jsonschema.validate(instance=base, schema=c.pred_obj)
        return True, ""
    except Exception as e:                   # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:120]}"


@constraint("schema.type_mapping_exact", "family",
            applies=lambda c: c.family == "schema_gen" and c.pred_obj is not None
            and (c.derived is not None or c.source is not None))
def _c_schema_types(c: Ctx) -> Tuple[bool, str]:
    """Independent re-derivation of the schema from the document it describes.

    On the questions that rewrite before describing, the derivation starts
    from ``checks.derived`` — the document after the rewrite — so a model that
    described the *input* instead of the *result* fails here, with the paths
    named."""
    base = c.derived if c.derived is not None else c.source
    bad = diff_values(c.pred_obj, infer_schema(base, c.schema_kind))
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
    extra = sorted(collect_schema_keywords(c.pred_obj) - allowed, key=str)
    return not extra, f"disallowed keywords: {extra[:5]}"


@constraint("session.no_restatement", "family",
            applies=lambda c: c.is_session and c.pred_obj is not None
            and c.session.get("turn_index", 1) > 1)
def _c_session_no_restatement(c: Ctx) -> Tuple[bool, str]:
    """A later turn's answer is not the document the turn started from.

    Every other family gets this for free: the build refuses any question that
    can be answered by handing the input back, because the input is in the
    prompt and can be compared with the reference. From the second turn of a
    dialogue on, the input is *not* in the prompt — it exists only in the
    model's own earlier answers — so the degenerate strategy here is to write
    out the state the conversation was already in, and nothing else in the
    scorer would name it. Both documents are refused: the state on the table
    (``prev_state``) and, on an undoing turn where they differ, the state the
    work was supposed to restart from (``source``).
    """
    for label, state in (("состояние предыдущего шага", c.prev_state),
                         ("документ, к которому применяется шаг", c.source)):
        if state is None:
            continue
        if not diff_values(c.pred_obj, state, coerce_numeric=c.coerce):
            return False, f"ответ равен: {label}"
    return True, ""


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

    The stripping is confined to the text OUTSIDE fenced blocks. Reasoning
    written inline always comes before the answer, never inside it, while the
    answer is a document that may legitimately hold the characters ``<think>``
    — an XML source, a configuration file, a string value. Running the pattern
    over the whole response would then quietly delete part of a correct answer
    and blame the model. Doing it this way also means the same text goes in and
    comes out for every reference this benchmark ships, which ``validate_task``
    checks on all of them.
    """
    text = generation or ""
    # Transport, not content. A byte-order mark in front of the first fence
    # and CRLF line endings both make the fence line stop matching, so a model
    # whose answer travelled through a Windows-shaped pipe would fail
    # `pack.fence_present` — a rule about *its* packaging that it did not
    # break. Neither carries meaning the prompt asks about, and the documents
    # this benchmark ships are all LF, so folding them changes no reference.
    if text.startswith("﻿"):
        text = text[1:]
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    fences = [i for i, ln in enumerate(lines) if _FENCE_LINE.match(ln)]
    inside: set = set()
    for a, b in zip(fences[0::2], fences[1::2]):
        inside.update(range(a, b + 1))
    if len(fences) % 2:                          # an unclosed fence: from it on
        inside.update(range(fences[-1], len(lines)))
    out, buf = [], []

    def flush() -> None:
        if buf:
            chunk = _THINK_BLOCK_RE.sub("", "\n".join(buf))
            out.append(_THINK_TAG_RE.sub("", chunk))
            buf.clear()

    for i, ln in enumerate(lines):
        if i in inside:
            flush()
            out.append(ln)
        else:
            buf.append(ln)
    flush()
    return "\n".join(out).strip()


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


#: A cell whose score is zero counts as this much. Without a floor the
#: geometric mean below is zero the moment a model fails one cell outright,
#: and the benchmark stops separating everything beneath the frontier — a
#: model at 0.30 overall with one dead cell would tie with a model that
#: answers nothing. With the floor, one dead cell of forty-four multiplies the
#: score by 0.01**(1/44) ≈ 0.90: a tenth off, not an annihilation. Two cost
#: 0.81, five cost 0.59. The number is a declared convention, not a tuning
#: knob — it is written here, in the task description, and in the README.
BALANCE_FLOOR = 0.01


def balance_score(items: List[Any]) -> float:
    """Geometric mean of the per-cell pass rates, with a floor on zeros.

    The headline ``sample_pass_rate`` is an arithmetic mean over questions, and
    it answers "how much of this work does the model get right". It cannot
    answer the other question a pipeline owner asks — "is there a kind of work
    here it cannot do at all" — because a cell at zero is worth exactly as much
    to an arithmetic mean as a cell at zero-point-three is worth less than one
    at 0.6. A geometric mean is dominated by its smallest term, which is
    precisely the property wanted: 0.9/0.9/0.9/0.1 scores 0.70 as a mean and
    0.52 here, and the second number is the one that predicts whether the model
    can be put in front of a format that matters.

    Note what this is NOT: it is not robust to outliers. The robust statistic
    is the median, and on the same data the median cell is *higher* than the
    mean. This metric deliberately punishes imbalance.

    ``items`` is the list of ``(cell, pass)`` pairs ``process_results`` emits —
    the same shape MERA's own multi-class metrics use, which is what lets a
    per-cell statistic be computed by the harness rather than beside it.
    """
    cells: Dict[str, List[float]] = {}
    for item in items:
        try:
            cell, value = item
        except (TypeError, ValueError):
            cell, value = "?", float(item)
        cells.setdefault(str(cell), []).append(float(value))
    if not cells:
        return 0.0
    total = 0.0
    for values in cells.values():
        rate = sum(values) / len(values)
        total += math.log(max(rate, BALANCE_FLOOR))
    return math.exp(total / len(cells))


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Score one generation against the whole constraint stack of its question.

    An empty response never passes. Several constraints are vacuously true on an
    empty string (nothing outside the fence, no disallowed keywords), so without
    this guard a model that answered nothing would collect partial credit.
    """
    response = normalize_generation(_extract_prediction(results))
    checks = score_response(doc, response)
    applicable = [c for c in checks if c["applicable"]]
    cats = (doc["meta"].get("categories") or {})
    cell = f"{cats.get('family', '?')}/{cats.get('difficulty', '?')}"
    zero = {"sample_pass_rate": 0.0, "constraint_pass_rate": 0.0,
            "format_pass_rate": 0.0, "content_pass_rate": 0.0,
            "task_pass_rate": 0.0, "balance_score": (cell, 0.0)}
    if not applicable:                       # cannot happen with a well-formed doc
        eval_logger.warning("sample %s has no applicable constraints",
                            doc["meta"].get("id"))
        return zero

    if not response.strip():
        return zero

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

    sample = float(n_pass == len(applicable))
    return {
        "sample_pass_rate": sample,
        "constraint_pass_rate": n_pass / len(applicable),
        "format_pass_rate": _combo_pass(checks, COMBOS["format"]) or 0.0,
        "content_pass_rate": content,
        "task_pass_rate": task,
        # Carries its cell so the aggregation can group by it. lm-eval hands an
        # aggregation the flat list of per-sample values and nothing else, so a
        # per-cell statistic has to travel inside the value — the same shape
        # MERA's own multi-class metrics use.
        "balance_score": (cell, sample),
    }


def scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """The metrics that are plain numbers per question.

    ``balance_score`` is a ``(cell, pass)`` pair until the whole run is
    aggregated, so anything that averages, bounds-checks or prints per-sample
    metrics has to skip it rather than crash on it.
    """
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


# --------------------------------------------------------------------------- #
# Prompt construction (local JSON files only)
# --------------------------------------------------------------------------- #

_PROMPTS_CACHE: List[str] | None = None
_META_CANDIDATES = (
    "../../datasets/SOBHard2/dataset_meta.json",
    "../../../datasets/SOBHard2/dataset_meta.json",
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


if "remove_whitespace_and_nones" not in FILTER_REGISTRY:
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
