# -*- coding: utf-8 -*-
"""Сборка истории диалога для многоходовых вопросов GorillaHard.

Многоходовость в lm-eval — это механизм few-shot: при
``--apply_chat_template --fewshot_as_multiturn`` каждый «шот» уходит в запрос
отдельной парой сообщений ``user`` / ``assistant``, а текущий вопрос
добавляется последним ``user``. Значит, чтобы получить настоящий диалог,
достаточно, чтобы сэмплер выдавал не случайные примеры, а **предыдущие ходы
того же диалога по возрастанию номера**. Что и делает этот класс.

Почему один класс, а не два
---------------------------

MERA работает на вендоренном форке lm-eval 0.4.9.2, где контекст строит сам
сэмплер (``get_context`` / ``get_chat_context``). В актуальном upstream
(0.4.13) сборка переехала в ``Task.fewshot_context`` + ``build_qa_turn``, а
сэмплер только **выбирает документы** (``sample(n, eval_doc=...)``); методы
``get_context`` и ``get_chat_context`` там больше никто не вызывает. Класс,
который переопределяет только их, на upstream импортируется без ошибки и молча
игнорируется — числа получаются, и они неправильные.

Поэтому здесь реализованы **обе поверхности над одним ``pick()``**. Один и тот
же YAML работает на обеих версиях, и обновление lm-eval не потребует правок в
задаче. Проверяется это так: на обеих версиях собирается контекст всех строк
теста и сравнивается побайтово — совпасть должны и однокадровые промпты, и
история многоходовых вопросов.

Что именно отбирается
---------------------

Ходы одного диалога делят ``meta.dialog_id`` и нумеруются ``meta.turn_id`` с
нуля. Отбор ведётся ещё и по ``meta.wording``: в тесте лежат все пять
формулировок каждого хода, и без этого фильтра в историю попали бы пять копий
одного и того же хода. У однокадрового вопроса свой ``dialog_id`` и
``turn_id`` 0 — история пуста, и промпт побайтово совпадает с тем, что было до
многоходовости. Ветвления «диалог или не диалог» нет нигде.
"""
from __future__ import annotations

import logging

from lm_eval.api.samplers import ContextSampler

eval_logger = logging.getLogger(__name__)


class DialogSampler(ContextSampler):
    """Предыдущие ходы того же диалога, старший первым."""

    def __init__(self, docs=None, task=None, fewshot_indices=None, rnd=None,
                 df=None, **kwargs):
        pool = docs if docs is not None else df
        try:  # сигнатура upstream 0.4.13
            super().__init__(pool, rnd=rnd, fewshot_indices=fewshot_indices,
                             **kwargs)
        except TypeError:  # сигнатура форка 0.4.9.2
            super().__init__(docs=pool, task=task,
                             fewshot_indices=fewshot_indices, rnd=rnd)
        self._pool = list(pool or [])
        self._index = None

    # ---- единственный источник истины ------------------------------------
    def _by_dialog(self) -> dict:
        """Ходы, разложенные по (dialog_id, wording). Строится один раз."""
        if self._index is None:
            index: dict = {}
            for d in self._pool:
                meta = d.get("meta") or {}
                key = (meta.get("dialog_id"), meta.get("wording"))
                index.setdefault(key, []).append(d)
            for turns in index.values():
                turns.sort(key=lambda d: (d.get("meta") or {}).get("turn_id", 0))
            self._index = index
        return self._index

    def pick(self, eval_doc, n):
        """История хода: ходы своего диалога с меньшим turn_id, по порядку.

        ``eval_doc`` — обязателен. На upstream текущий документ передаётся
        сэмплеру только при ``fewshot_split == test_split``; без него собрать
        историю нечем, и молчаливый откат на случайные примеры превратил бы
        диалог в обычный few-shot, замаскированный под диалог. Такую ошибку
        конфига видно только по числам, поэтому здесь она — исключение.
        """
        if eval_doc is None:
            raise RuntimeError(
                "DialogSampler: lm-eval не передал текущий документ (eval_doc). "
                "Проверьте, что fewshot_split совпадает с test_split — только "
                "при этом условии Task.fewshot_context передаёт eval_doc."
            )
        meta = eval_doc.get("meta") or {}
        turn_id = meta.get("turn_id", 0)
        if not turn_id:                       # первый ход или однокадровый вопрос
            return []
        siblings = self._by_dialog().get((meta.get("dialog_id"), meta.get("wording")), [])
        history = [d for d in siblings
                   if (d.get("meta") or {}).get("turn_id", 0) < turn_id]
        return history[-n:] if n else history

    # ---- поверхность upstream 0.4.13 -------------------------------------
    def sample(self, n, eval_doc=None, df=None, **kwargs):
        return self.pick(eval_doc, n)

    # ---- поверхность форка 0.4.9.2 ---------------------------------------
    # Обе реализации повторяют базовые ``ContextSampler.get_context`` и
    # ``get_chat_context`` форка с точностью до отбора документов: рендеринг
    # идёт через те же ``self.doc_to_text`` / ``self.doc_to_target``, которые
    # базовый ``__init__`` уже связал с ``fewshot_config``. ``doc_to_choice`` у
    # задачи нет, поэтому ветки под варианты ответа опущены.
    def get_context(self, doc, num_fewshot, gen_prefix=None):
        prefix = gen_prefix + " " if gen_prefix else ""
        parts = []
        for d in self.pick(doc, num_fewshot):
            target = self.doc_to_target(d)
            if isinstance(target, list):
                target = str(target[0])
            parts.append(self.doc_to_text(d) + self.target_delimiter
                         + prefix + target + self.fewshot_delimiter)
        return "".join(parts)

    def get_chat_context(self, doc, num_fewshot, fewshot_as_multiturn=False,
                         gen_prefix=None):
        prefix = gen_prefix + " " if gen_prefix else ""
        if not fewshot_as_multiturn:
            context = self.get_context(doc, num_fewshot, gen_prefix=gen_prefix)
            return [{"role": "user", "content": context}] if context else []
        history = []
        for d in self.pick(doc, num_fewshot):
            target = self.doc_to_target(d)
            if isinstance(target, list):
                target = str(target[0])
            history.append({"role": "user", "content": self.doc_to_text(d)})
            history.append({"role": "assistant", "content": prefix + target})
        return history
