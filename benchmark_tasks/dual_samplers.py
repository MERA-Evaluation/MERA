"""Dialogue sampler that works on BOTH lm-eval generations, byte-identically.

Old (MERA fork, 0.4.9.x): the sampler builds the context itself —
    ContextSampler(docs, task, fewshot_indices=None, rnd=None)
    .get_context(doc, num_fewshot, gen_prefix)             -> str
    .get_chat_context(doc, num_fewshot, multiturn, prefix) -> list[dict]

New (upstream >= 0.4.13): the sampler ONLY picks documents; Task.fewshot_context
builds the messages —
    ContextSampler(df, *, rnd=None, fewshot_indices=None, **kw)
    .sample(n, eval_doc=None, df=None, **kw)               -> list[dict]

This class implements both surfaces over one `pick()`, so the same file can be
referenced from the same YAML on either version.

Mixed pools are the normal case here: SOBHard2.1 keeps every family in one
`test` split, and with `num_fewshot > 0` the harness routes EVERY document
through the sampler — not only the dialogue ones. A document without
`meta.dialogue_id` gets an empty history, which reproduces the zero-shot
context byte for byte. Hence every lookup below is a `.get()`: a KeyError here
would take down the twelve single-turn families along with the dialogues.
"""

from lm_eval.api.samplers import ContextSampler

try:  # only present on the old generation; used for text-mode rendering there
    from lm_eval.utils import apply_template
except Exception:  # pragma: no cover
    apply_template = None


def _meta(doc):
    """`meta` of a document, tolerating absence and non-dict values."""
    if not isinstance(doc, dict):
        return {}
    meta = doc.get("meta")
    return meta if isinstance(meta, dict) else {}


class DualDialogueSampler(ContextSampler):
    def __init__(self, docs=None, task=None, fewshot_indices=None, rnd=None,
                 df=None, **kwargs):
        pool = docs if docs is not None else df
        try:  # new signature
            super().__init__(pool, rnd=rnd, fewshot_indices=fewshot_indices,
                             **kwargs)
        except TypeError:  # old signature
            super().__init__(docs=pool, task=task,
                             fewshot_indices=fewshot_indices, rnd=rnd)
        self._pool = list(pool or [])
        self._task = task
        # dialogue_id -> turns sorted by turn_index. Built once: the sampler is
        # called per evaluated document, and a linear scan of the whole split
        # per call is quadratic in the size of the benchmark.
        self._by_dialogue = {}
        for doc in self._pool:
            meta = _meta(doc)
            did = meta.get("dialogue_id")
            if did is None:
                continue
            self._by_dialogue.setdefault(did, []).append(doc)
        for turns in self._by_dialogue.values():
            turns.sort(key=lambda d: _meta(d).get("turn_index", 0))

    # ---- the single source of truth -------------------------------------
    def pick(self, eval_doc, n):
        """Previous turns of this dialogue, oldest first, at most n.

        Empty for anything that is not a dialogue turn, which is what keeps
        the single-turn families zero-shot under a multi-turn invocation.
        """
        meta = _meta(eval_doc)
        did = meta.get("dialogue_id")
        idx = meta.get("turn_index")
        if did is None or idx is None:
            return []
        prev = [d for d in self._by_dialogue.get(did, ())
                if _meta(d).get("turn_index", 0) < idx]
        return prev[-n:] if n else prev

    # ---- new-generation surface -----------------------------------------
    def sample(self, n, eval_doc=None, df=None, **kwargs):
        return self.pick(eval_doc, n)

    # ---- old-generation surface -----------------------------------------
    def get_context(self, doc, num_fewshot, gen_prefix=None):
        parts = []
        for d in self.pick(doc, num_fewshot):
            parts.append(self._render(d) + self.target_delimiter
                         + self._target(d) + self.fewshot_delimiter)
        return "".join(parts)

    def get_chat_context(self, doc, num_fewshot, fewshot_as_multiturn=False,
                         gen_prefix=None):
        shots = self.pick(doc, num_fewshot)
        if not shots:
            # No history: return nothing rather than an empty user turn, so the
            # context is identical to the zero-shot one.
            return []
        if not fewshot_as_multiturn:
            return [{"role": "user",
                     "content": self.get_context(doc, num_fewshot, gen_prefix)}]
        out = []
        for d in shots:
            out.append({"role": "user", "content": self._render(d)})
            out.append({"role": "assistant", "content": self._target(d)})
        return out

    # ---- rendering helpers (old generation only) -------------------------
    def _render(self, d):
        # The old ContextSampler wires these in __init__, honouring
        # fewshot_config.doc_to_text when it is set; going through them keeps
        # this identical to what the stock sampler would have produced.
        fn = getattr(self, "doc_to_text", None)
        if callable(fn):
            return fn(d)
        tpl = (self.config.fewshot_config or {}).get("doc_to_text") \
            or self.config.doc_to_text
        return apply_template(tpl, d) if isinstance(tpl, str) else tpl(d)

    def _target(self, d):
        fn = getattr(self, "doc_to_target", None)
        if callable(fn):
            target = fn(d)
        else:
            tpl = self.config.doc_to_target
            target = apply_template(tpl, d) if isinstance(tpl, str) else tpl(d)
        if isinstance(target, list):  # doc_to_target may return a list of forms
            target = target[0]
        return target if isinstance(target, str) else str(target)
