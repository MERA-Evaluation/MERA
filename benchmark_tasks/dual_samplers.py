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

Mixed pools are the normal case here: SOBHard keeps every family in one `test`
split, and with `num_fewshot > 0` the harness routes EVERY document through the
sampler — not only the dialogue ones. A document without `meta.dialogue_id`
gets an empty history, which reproduces the zero-shot context byte for byte.
Hence every lookup below is a `.get()`: a KeyError here would take down the
single-turn families along with the dialogues.
"""

import os

from lm_eval.api.samplers import ContextSampler

try:  # only present on the old generation; used for text-mode rendering there
    from lm_eval.utils import apply_template
except Exception:  # pragma: no cover
    apply_template = None

#: Split the history turns come from. `test` ships with `outputs` stripped, so
#: a history built from it would show the model empty assistant turns; `shots`
#: is the split MERA allows to carry answers, and every turn before the
#: evaluated one lives there.
HISTORY_SPLIT = "shots"

#: SOBHard's YAML, read only for the dataset coordinates when the harness hands
#: the sampler documents but no task (upstream 0.4.13). This module has exactly
#: one consumer -- sobhard.yaml -- so pointing at it is not a hidden assumption.
TASK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "sobhard", "sobhard.yaml")


def _dataset_args() -> dict:
    """`dataset_path` / `dataset_name` / `dataset_kwargs` from the task YAML.

    The same file the harness built the task from, so the history arrives from
    the same copy of the dataset as the questions: repoint `dataset_path` at a
    private copy and the `shots` split follows.
    """
    import yaml

    class _Tolerant(yaml.SafeLoader):
        """SafeLoader that does not trip over `!function`.

        Only three scalar fields are wanted here; loading the task's functions
        would mean importing this very module from inside itself.
        """

    _Tolerant.add_multi_constructor("!", lambda loader, suffix, node: None)

    with open(TASK_YAML, encoding="utf-8") as fh:
        cfg = yaml.load(fh, Loader=_Tolerant) or {}
    args = dict(cfg.get("dataset_kwargs") or {})
    args["path"] = cfg["dataset_path"]
    if cfg.get("dataset_name"):
        args["name"] = cfg["dataset_name"]
    return args


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
        # The pool the harness passes is deliberately unused: with
        # `fewshot_split: test` it is the test rows, and the history lives in
        # another split. The task reference is kept only to avoid reading the
        # dataset twice where it is already loaded.
        self._task = task
        self._by_dialogue = None

    # ---- the history split ----------------------------------------------
    def _load_history_docs(self):
        """Rows of the `shots` split -- from the task if present, else from disk.

        On the 0.4.9.x fork the sampler is handed the task, whose whole
        DatasetDict is already loaded, so this is a lookup of a sibling split.
        On upstream 0.4.13 the sampler gets only a list of documents, so the
        split is read separately; `load_dataset` hits the same cache the task
        loaded through and downloads nothing again.
        """
        dataset = getattr(self._task, "dataset", None)
        if dataset is not None and HISTORY_SPLIT in dataset:
            return list(dataset[HISTORY_SPLIT])

        import datasets

        args = _dataset_args()
        try:
            return list(datasets.load_dataset(split=HISTORY_SPLIT, **args))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"DualDialogueSampler: could not read split {HISTORY_SPLIT!r} of "
                f"dataset {args.get('path')!r}, which holds the history of the "
                f"multi-turn documents. Without it every turn after the first is "
                f"under-specified and the run would measure something else. "
                f"Original error: {exc}"
            ) from exc

    def _index(self):
        """History turns by dialogue_id, oldest first. Built once per run."""
        if self._by_dialogue is None:
            index = {}
            for doc in self._load_history_docs():
                did = _meta(doc).get("dialogue_id")
                if did is None:
                    continue
                index.setdefault(did, []).append(doc)
            for turns in index.values():
                turns.sort(key=lambda d: _meta(d).get("turn_index", 0))
            self._by_dialogue = index
        return self._by_dialogue

    # ---- the single source of truth -------------------------------------
    def pick(self, eval_doc, n):
        """Previous turns of this dialogue, oldest first, at most n.

        Empty for anything that is not a dialogue turn, which is what keeps
        the single-turn families zero-shot under a multi-turn invocation.

        `eval_doc` is required. Upstream hands the current document to the
        sampler only when `fewshot_split == test_split`, and without it there is
        nothing to build a history from -- so a missing one is raised rather
        than quietly answered with an empty history, which is a mistake that
        shows up only in the scores, and the scores would look plausible.
        """
        if eval_doc is None:
            raise RuntimeError(
                "DualDialogueSampler: no eval_doc. The harness passes the "
                "current document only when fewshot_split equals test_split -- "
                "set `fewshot_split: test` in the task YAML. Without it the "
                "multi-turn documents silently lose their history."
            )
        meta = _meta(eval_doc)
        did = meta.get("dialogue_id")
        idx = meta.get("turn_index")
        if did is None or idx is None:
            return []
        prev = [d for d in self._index().get(did, ())
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
