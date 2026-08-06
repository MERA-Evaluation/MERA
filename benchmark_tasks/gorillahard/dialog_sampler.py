# -*- coding: utf-8 -*-
"""Сборка истории диалога для многоходовых вопросов GorillaHard.

Многоходовость в lm-eval — это механизм few-shot: при
``--apply_chat_template --fewshot_as_multiturn`` каждый «шот» уходит в запрос
отдельной парой сообщений ``user`` / ``assistant``, а текущий вопрос
добавляется последним ``user``. Значит, чтобы получить настоящий диалог,
достаточно, чтобы сэмплер выдавал не случайные примеры, а **предыдущие ходы
того же диалога по возрастанию номера**. Что и делает этот класс.

Откуда берутся предыдущие ходы
------------------------------

Из сплита ``shots`` — и только из него. Состояние, над которым работает ход k,
по построению есть ответ на ход k−1, поэтому история хода — это чужие эталонные
ответы. Показывать их можно ровно тогда, когда сами эти ходы не оцениваются:
оценить ход k — значит показать ответы ходов 1..k−1, а оценить заодно и ход k+1
— значит показать ответ на k, который сам оценивается. Двух оцениваемых ходов в
одной цепочке быть не может, поэтому **из диалога оценивается ровно один ход —
последний**, а все предшествующие лежат в ``shots``, которому по правилам MERA
разрешено нести ответы (см. ноутбук заливки: ``hide_answers`` стирает ``outputs``
только в ``test``).

Раньше история собиралась из самого ``test``. На публичной копии, где ответы
стёрты, это давало пустую реплику ассистента (форк 0.4.9.2) или выброшенное
сообщение (upstream 0.4.13) — то есть публичный ``dialog_pass_rate`` был
систематически занижен и несравним с приватным. Теперь на обеих копиях история
одна и та же: ``shots`` публикуется с ответами.

Почему ``fewshot_split`` остался ``test``
-----------------------------------------

Казалось бы, достаточно написать в YAML ``fewshot_split: shots``. Не достаточно:
в upstream текущий документ уходит сэмплеру только при совпадении сплитов —

    for fs_doc in self.sampler.sample(
        n=num_fewshot,
        eval_doc=doc if self.fewshot_cfg.split == self.config.test_split else None,
    ):

— а без ``eval_doc`` сэмплер не знает, из какого диалога тянуть историю. Обхода
через конфиг нет: сэмплер строится как ``sampler_cls(fewshot_docs, rnd=None)`` и
произвольных ключей из YAML не получает, а ``fewshot_config.samples`` даёт пул в
обход ``split``, но тогда ``fewshot_cfg.split`` становится ``None`` и
``eval_doc`` снова не передаётся.

Поэтому ``fewshot_split`` совпадает с ``test_split`` — ради ``eval_doc``, — а пул,
который харнесс при этом передаёт (строки теста), сэмплер не использует вовсе:
он сам лениво читает сплит ``shots``. Один раз за прогон, при первом же ходе с
историей.

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
нуля. Отбор ведётся ещё и по ``meta.wording``: формулировка инструкции у ходов
одного диалога общая, и ключ из двух полей не даст собрать историю из чужой
тональности, если сборка датасета когда-нибудь начнёт их смешивать. У
однокадрового вопроса свой ``dialog_id`` и ``turn_id`` 0 — история пуста, и
промпт побайтово совпадает с тем, что было до многоходовости. Ветвления «диалог
или не диалог» нет нигде.

Молчаливых откатов здесь нет ни одного. Нет ``eval_doc``, нет сплита ``shots``,
нет истории у хода, который её требует, пуст эталон у хода истории — всё это
исключение, а не пустая история: такую ошибку конфига видно только по числам, а
числа при этом выглядят правдоподобно.
"""
from __future__ import annotations

import logging
import os

from lm_eval.api.samplers import ContextSampler

eval_logger = logging.getLogger(__name__)

#: Сплит, из которого берутся ходы истории. Он же — единственный сплит датасета,
#: которому разрешено уезжать на Hub с заполненными ``outputs``.
HISTORY_SPLIT = "shots"

#: YAML задачи лежит рядом с этим файлом; из него берутся координаты датасета,
#: когда харнесс не дал ссылку на задачу (upstream 0.4.13).
TASK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gorillahard.yaml")


def _dataset_args() -> dict:
    """``dataset_path`` / ``dataset_name`` / ``dataset_kwargs`` из YAML задачи.

    Читается тот же файл, по которому харнесс собрал задачу, поэтому история
    приезжает из той же копии датасета, что и вопросы: подменили в YAML
    ``dataset_path`` на приватную копию с ответами — сплит ``shots`` приедет
    оттуда же.
    """
    import yaml

    class _Tolerant(yaml.SafeLoader):
        """Тот же SafeLoader, но не спотыкается о ``!function``.

        Нам из конфига нужны три скалярных поля; функции задачи (``doc_to_text``,
        сам этот сэмплер) грузить незачем, и попытка это сделать заставила бы
        импортировать модуль задачи из самого модуля задачи.
        """

    _Tolerant.add_multi_constructor(
        "!", lambda loader, suffix, node: None)

    with open(TASK_YAML, encoding="utf-8") as fh:
        cfg = yaml.load(fh, Loader=_Tolerant) or {}
    args = dict(cfg.get("dataset_kwargs") or {})
    args["path"] = cfg["dataset_path"]
    if cfg.get("dataset_name"):
        args["name"] = cfg["dataset_name"]
    return args


class DialogSampler(ContextSampler):
    """Предыдущие ходы того же диалога из сплита ``shots``, старший первым."""

    def __init__(self, docs=None, task=None, fewshot_indices=None, rnd=None,
                 df=None, **kwargs):
        pool = docs if docs is not None else df
        try:  # сигнатура upstream 0.4.13
            super().__init__(pool, rnd=rnd, fewshot_indices=fewshot_indices,
                             **kwargs)
        except TypeError:  # сигнатура форка 0.4.9.2
            super().__init__(docs=pool, task=task,
                             fewshot_indices=fewshot_indices, rnd=rnd)
        # Пул, переданный харнессом (строки теста), сознательно не используется:
        # история живёт в другом сплите. Ссылка на задачу нужна только чтобы не
        # читать датасет второй раз там, где он уже загружен.
        self._task = task
        self._index = None

    # ---- сплит истории ----------------------------------------------------
    def _load_history_docs(self) -> list:
        """Строки сплита ``shots`` — из задачи, если она есть, иначе с диска.

        На форке 0.4.9.2 сэмплер получает саму задачу, у которой уже загружен
        весь ``DatasetDict``, — там это просто обращение к соседнему сплиту. На
        upstream 0.4.13 сэмплеру передают только список документов, поэтому
        сплит читается отдельно; ``load_dataset`` при этом попадает в тот же
        кеш, что и загрузка самой задачи, и ничего не скачивает повторно.
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
                "DialogSampler: не удалось прочитать сплит «%s» датасета %r, "
                "а в нём лежит история многоходовых вопросов. Без неё ходы "
                "после первого недоопределены, и прогон измерил бы не то. "
                "Исходная ошибка: %s" % (HISTORY_SPLIT, args.get("path"), exc)
            ) from exc

    def _by_dialog(self) -> dict:
        """Ходы истории, разложенные по (dialog_id, wording). Строится один раз."""
        if self._index is None:
            index: dict = {}
            for d in self._load_history_docs():
                meta = d.get("meta") or {}
                key = (meta.get("dialog_id"), meta.get("wording"))
                index.setdefault(key, []).append(d)
            for turns in index.values():
                turns.sort(key=lambda d: (d.get("meta") or {}).get("turn_id", 0))
            self._index = index
        return self._index

    # ---- единственный источник истины ------------------------------------
    def pick(self, eval_doc, n):
        """История хода: ходы своего диалога с меньшим turn_id, по порядку.

        ``eval_doc`` — обязателен. На upstream текущий документ передаётся
        сэмплеру только при ``fewshot_split == test_split``; без него собрать
        историю нечем, и молчаливый откат на случайные примеры превратил бы
        диалог в обычный few-shot, замаскированный под диалог.
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
        key = (meta.get("dialog_id"), meta.get("wording"))
        history = [d for d in self._by_dialog().get(key, [])
                   if (d.get("meta") or {}).get("turn_id", 0) < turn_id]
        if len(history) != turn_id:
            raise RuntimeError(
                "DialogSampler: у хода %s диалога %s в сплите «%s» нашлось %d "
                "предыдущих ходов вместо %d. Ход без полной истории "
                "недоопределён; пустая история дала бы правдоподобное, но "
                "неверное число." % (meta.get("id"), key, HISTORY_SPLIT,
                                     len(history), turn_id)
            )
        empty = [d["meta"]["id"] for d in history if not (d.get("outputs") or "").strip()]
        if empty:
            raise RuntimeError(
                "DialogSampler: у ходов истории %s пустой outputs. Это копия "
                "датасета со стёртыми ответами в «%s»; реплики ассистента в "
                "истории окажутся пустыми, и ходы после первого будут "
                "измерены неверно. Ответы в «%s» стирать нельзя — их стирают "
                "только в тесте." % (empty, HISTORY_SPLIT, HISTORY_SPLIT)
            )
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
