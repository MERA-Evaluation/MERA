# Humor

## Task Description

**Humor** is a Russian-language dataset for classifying humorous texts (jokes and short stories) and performing two related subtasks: identifying the comic-effect type of the main text and selecting, among four additional texts, the one that belongs to the same class.

The dataset contains 600 examples balanced across 12 classes (11 types of comic effect plus a "not a joke" class). Each main text is paired with four additional options (А, Б, В, Г).

Tested model skills: Text Analysis, Semantic Textual Similarity, Humor Detection, Humor Classification, Humor Recognition, Irony Recognition

Authors: Denis Shevelev, Alexandra Eliseeva, Alexander Kharitonov, Alexander Astafurov

## Motivation

The dataset evaluates whether a model can both classify the structural type of humor in a text and find another text with the same type among alternatives. The "not a joke" class acts as a control: the model should reliably distinguish humorous from non-humorous texts.

## Dataset Description

The dataset consists of two files with the same structure:

- `test.json` — **600** test examples with ids **1–600**
- `shots.json` — **4** few-shot examples with ids **601–604**

Both files have the form `{"access": "private", "data": [...]}`. Each item contains an `instruction` template, `inputs`, `outputs`, and `meta`.

Each example uses four additional-text fields (`option_a`–`option_d`). The `instruction` field contains placeholders for all four options. The "Формат ответа" block states that answer letters in the РЕШЕНИЕ line may only be uppercase Russian letters from the list (А, Б, В, Г).

### Data Fields

Each example contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model
- `inputs` — input data:
    - `context` [str] — descriptions of 12 classes of humorous effect
    - `main_text` [str] — main text to classify
    - `option_a` [str] — additional text A
    - `option_b` [str] — additional text B
    - `option_c` [str] — additional text C
    - `option_d` [str] — additional text D
- `outputs` [str] — a string of the form `class,letter` (e.g. `ирония,Г`)
- `meta` — metadata:
    - `id` [int] — example number

### Data Example

```json
{
    "instruction": "Помогите, пожалуйста, выполнить задание.\n\nЗадача:\nДан текст, который предположительно содержит шутку или анекдот. Определите, к какому из 12 классов юмористического эффекта, описания которых представлены ниже, его можно отнести. Какой из дополнительных текстов относится к такому же классу?\n\nКонтекст:\n{context}\n\nФормат ответа:\nОтвет выведите в следующем формате:\n\nОТВЕТ <название класса>\nРЕШЕНИЕ <буква>\n\nБуквами ответа в строке РЕШЕНИЕ могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г).\n\nТекст:\n{main_text}\n\nДополнительные тексты:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\n",
    "inputs": {
        "main_text": "Профессор валит студента:\n— Если ответишь на последний вопрос — ставлю пятёрку и свободен.\n— Давайте!\n— Как забить червяка в землю?\n— Обмазать его цементом и забить...\nНа следующий день профессор догоняет студента в коридоре:\n— Вот зачётка с пятёркой, а это — шоколадка от моей жены. Сказала, что ты гений!",
        "option_a": "Петька и Василий Иванович моются в бане, а Петька говорит:\n — Василий Иванович, у вас пятки грязнее, чем у меня.\n — Так я насколько тебя старше!",
        "option_b": "А сейчас японская певица Ясука исполнит популярную песню Атомулиядала, что в переводе Сомнение...",
        "option_c": "Я детям своим колыбельные любил петь. Часов до трех. Иногда с друзьями.",
        "option_d": "Судебное заседание. \r\nСудья:\r\n— Подсудимый, Вы обвиняетесь в зверском изнасиловании сотрудника ГИБДД.\r\nПодсудимый:\r\n— Это невозможно — я импотент, вот и справка от врача...\r\nПотерпевший:\r\n— А Вы его про жезлик, про жезлик спросите!",
        "context": "Юмор в анекдоте или шутке прежде всего строится на основе одного из 11 эффектов: \n - “игра слов” (когда обыгрывается омонимия - лексическая, синтаксическая или иная)\n - “игра паттернов” (когда участники видят ситуацию по-разному и это комично),\n - “секрет Полишинеля” (когда неожиданно и комично выясняется правда), \n - “каламбур” (когда из слов составляются причудливые игровые комбинации, создающие забавные, а иногда и неприличные отсылки), \n - “оксюморон” (совмещение несовместимого), \n - “логика абсурда” (когда ситуации даётся абсурдистское продолжение), \n - “обман ожиданий” (когда заданная в начале шутки или анекдота ситуация развивается не так, как ожидаешь)\n - “капитан очевидность” (когда юмор состоит в неожиданном сообщении очевидных фактов)\n - “эллипсис” (когда история сокращена настолько, что от начала резко переходит к концу),\n - “ирония” (когда ситуация развивается как и должна была, но в неподходящих обстоятельствах)\n - “остроумный ответ” (когда юмор заключается просто в неожиданности остроумного ответа).\n Каждый из этих эффектов даёт название классу текстов с комическим содержимым, а для некомических текстов дан дополнительный, двенадцатый класс “не анекдот”."
    },
    "outputs": "эллипсис,Г",
    "meta": {
        "id": 601
    }
}
```

### Prompts

Five prompt variants were prepared for the task. They are distributed across examples: each example is assigned one of five templates (index 0–4). The `instruction` field of each example contains the full text of the selected prompt with placeholders `{context}`, `{main_text}`, and `{option_a}`–`{option_d}`. Each template uses a consistent form of address (informal ты or formal вы). The model must return two lines: `ОТВЕТ <class>` and `РЕШЕНИЕ <letter>`.

### Dataset Creation

The dataset was created by an expert group of linguists: 100 texts were selected per class, tasks with four additional options were formed, and cross-checking was performed.

### Metrics

- **Exact match (EM)**: checks whether the model answer exactly matches the reference for both the class label and the letter choice.
