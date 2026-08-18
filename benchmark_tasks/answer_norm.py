"""Answer normalisation shared by the generative tasks.

SQuAD's ``normalize_answer`` strips ``string.punctuation``, which is ASCII only,
so the punctuation a Russian model actually types survives it::

    «А»  -> «а»   (does not match «А»)
    "А"  ->  а    (matches)

A model that quotes its answer the Russian way was therefore scored wrong while
the same answer in ASCII quotes was scored right. The marks below are removed
before the SQuAD comparison, the same way SQuAD removes the ASCII ones — cut
out, not replaced by a space.

The scoring service applies the identical set; see
``mera_scoring/src/domain/text_2_0/metrcis.py``.
"""

from transformers.data.metrics import squad_metrics

# Punctuation that `string.punctuation` misses: quotation marks, dashes and the
# ellipsis in their typographic forms. SQuAD already drops the ASCII hyphen, so
# dropping the dashes keeps «что—то» equal to «что-то» rather than adding a new
# way to be wrong.
TYPOGRAPHIC_PUNCTUATION = "«»‹›„“”‟‘’‚‛—–‒―‐‑−…"

_TYPOGRAPHIC_TABLE = {ord(char): None for char in TYPOGRAPHIC_PUNCTUATION}


def strip_typographic(text: str) -> str:
    """Drop the typographic punctuation SQuAD normalisation leaves behind."""
    return str(text or "").translate(_TYPOGRAPHIC_TABLE)


def compute_exact(gold: str, prediction: str) -> int:
    """SQuAD exact match that also ignores Russian typographic punctuation."""
    return squad_metrics.compute_exact(
        strip_typographic(gold), strip_typographic(prediction))
