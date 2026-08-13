# Characters

## Task Description

The **Characters** dataset is designed to evaluate the ability of language models to determine whether responses and questions match one of several given characters based on their personality, interests, speech patterns, and constraints. The benchmark tests a model's ability to analyze response consistency, identify the appropriate addressee for a question to obtain the most accurate answer, and verify factual correctness within a character's domain of knowledge.

Evaluated skills: Persona consistency understanding, Russian language proficiency, Factual correctness validation

Contributors: Denis Shevelev, Alexander Astafurov, Alexander Kharitonov

## Motivation

When developing generative chatbots and game characters, maintaining the consistency of a given persona's utterances is critically important. Models should not only generate grammatically correct answers, but also account for the full set of character traits: knowledge, age-related characteristics, speech patterns, and boundaries of competence. This dataset makes it possible to evaluate how well a language model understands and can operate with such persona constraints, which is necessary for creating believable dialogue agents.

### Limitations

The test is not intended to assess expert-level knowledge in any industrial or specialized domain (zoology, veterinary medicine, mycology, botany, film history, or zythology) and is intended to determine how well the model can meaningfully maintain the qualities and functions of the character as specified by the instructions and the conditions of the task.

## Dataset Description

### Characters

The dataset includes four characters with fixed characteristics:

_Alya, 25 years old. A film enthusiast.
She is well versed in Russian and foreign cinema. She learned about the world through films, so
she tends to make specific mistakes, since she often treats what is shown in a film
as truth: if a direct injection into the heart is performed in a movie, she is sure that in real life people
act the same way in similar situations.
She does not answer rudely phrased questions and defends her boundaries. Her speech is polite
and correct. She avoids conflicts and harsh judgments. She knows films released before she was born._

_Khmyr, 36 years old. A joker and wisecracker who answers any questions, often incorrectly, because
he only understands self-defense, types of beer, and sports games. But he never
admits that he does not understand something; his answers cannot be trusted. He may speak less than others, but only when it comes to topics that do not interest him (and if it comes to a subject that interests him, Khmyr can be very verbose). He does not avoid conflict or harsh judgments._

_Professor Chumkin, 58 years old. Broadly educated, although sometimes a bit tedious. A doctor
of sociology and a specialist in the natural sciences. He does not answer questions outside his
specialization. He is polite, correct, and non-dogmatic. His phrases may be more elaborate than those of
the others, and "professorial" introductory constructions may appear. He avoids conflict and
harsh judgments._

_Pavel, 19 years old. A veterinarian. An avid cat and dog lover. He knows a great deal about animals (and about
plants, since animals eat them and are treated with them!). He can keep up a conversation on any topic, but
will not answer questions outside his specialization. He does not avoid sharp judgments, but
is well-mannered. He does not watch and does not know films released before he was born._

All characters speak modern, literate Russian. Speech features are used as a secondary signal; the main differences are the topics of knowledge and age-related constraints.

### Task Types

The dataset contains 640 tasks divided into 4 types:

| Type | Content | Count |
|-----|------------|------------|
| 1 | With an address to a character: "which answer would this character give" | 128 |
| 2 | With an address to a character: "which answer would this character never give" | 128 |
| 3 | Without an address: "which character would give the clearest answer to this question" | 192 |
| 4 | Without an address: "to which of the listed characters is it completely pointless to ask this question" | 192 |

### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] — a string with the task formulation for the language model, including the character descriptions and the question

- `inputs` — input data forming the task:
    - `character` [str] — description of all four characters
    - `question` [str] — question or utterance
    - `option_a` [str] — first answer option
    - `option_b` [str] — second answer option
    - `option_c` [str] — third answer option
    - `option_d` [str] — fourth answer option
    - `option_e` [str] — fifth answer option
    - `option_f` [str] — sixth answer option

- `outputs` [str] — a string containing the letter of the correct answer (А, Б, В, Г, Д, Е)

Each example uses six answer-option fields (`option_a`–`option_f`). Unused options are stored as empty strings. The `instruction` field contains placeholders only for non-empty options: the "Варианты ответа" block includes exactly as many lines as there are non-empty `option_*` fields in that example (from 4 to 6). The "Формат ответа" block states that answer letters may only be uppercase Russian letters from the list (А, Б, В, Г, Д, Е); in each example this list is shortened to match the non-empty options.

- `meta` — metadata:
    - `id` [int] — example number
    - `type` [int] — task type (1, 2, 3, or 4)

### Data Example

#### Type 1 (with an Address, "Which Answer Would This Character Give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос или запрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос или запрос будет содержать обращение к одному из персонажей по имени.\n\nЗадание:\nОпредели, какой из предложенных ответов дал бы выбранный персонаж.\n\nОписание персонажей:\n{character}\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е).\n\nВопрос:\n{question}\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном кино. Осваивала мир по фильмам, поэтому склонна к специфическим ошибкам, так как нередко воспринимает показанное в каком-то фильме как правду: если в кино делают прямой укол в сердце, она уверена, что и в жизни в подобных ситуациях поступают так же. Не отвечает на вопросы, заданные невежливо, отстаивает свои границы. В речи вежлива, корректна. Избегает конфликтов и острых суждений. Знает фильмы, снятые до её рождения. 2) Хмырь, 36 лет. Шутник и ёрник, отвечает на любые вопросы, часто неправильно, потому что разбирается только в вопросах самообороны, сортах пива и спортивных играх. Но никогда не признаётся, что в чём-то не разбирается, его ответам верить нельзя. Может говорить чуть более ёмко, чем другие, но только когда дело касается не интересующих его областей (там его не остановить). Не избегает конфликтных и острых суждений. 3) Профессор Чумкин, 58 лет. Всесторонне образован, хотя порой немножко зануден. Доктор социологии и специалист по естественным наукам. На вопросы вне своей специализации не отвечает. Вежлив, корректен, недогматичен. Фразы могут быть более распространёнными, чем у остальных, могут проскальзывать “профессорские” вводные конструкции. Избегает конфликтных и острых суждений. 4) Павел, 19 лет. Ветеринар. Завзятый кошатник и собачник. Знает очень много о животных (и о растениях: ведь их едят и ими лечатся животные!). Может поддержать диалог на любую тему, но при этом не будет отвечать на вопросы вне своей специализации. Не избегает резких суждений, но воспитан. Не смотрит и не знает фильмов, снятых до его рождения. Все персонажи говорят на современном грамотном русском языке.",
        "question": "Аля, какой фильм Лантимоса ты любишь больше всего?",
        "option_a": "Так сложно выделить что-то одно... \"Виды доброты\" посмотрела уже трижды, пусть будет он.",
        "option_b": "Мне очень понравился его последний фильм \"Сентиментальная ценность\" – такой светлый, медитативный...",
        "option_c": "Очень люблю \"Лобстера\", всё остальное просто невозможно смотреть.",
        "option_d": "Лантимос? Это псевдоним?",
        "option_e": "Что ты снова заладил про свои фильмы? Может лучше по пивку?",
        "option_f": "Сложно ответить на этот вопрос без предварительного анализа. Мне нужно пересмотреть все фильмы Лантимоса, выделить основные критерии для сравнения и принять решение."
    },
    "outputs": "А",
    "meta": {
        "id": 641,
        "type": 1
    }
}
```

#### Type 2 (with an Address, "Which Answer Would This Character Never Give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос или запрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос или запрос будет содержать обращение к одному из персонажей по имени.\n\nЗадание:\nОпредели, какой из предложенных ответов выбранный персонаж ни за что бы не дал, учитывая его ограничения и интересы.\n\nОписание персонажей:\n{character}\n\nФормат ответа:\nВ качестве решения выведи только одну букву неверного (невозможного) варианта.\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е).\n\nВопрос:\n{question}\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном кино. Осваивала мир по фильмам, поэтому склонна к специфическим ошибкам, так как нередко воспринимает показанное в каком-то фильме как правду: если в кино делают прямой укол в сердце, она уверена, что и в жизни в подобных ситуациях поступают так же. Не отвечает на вопросы, заданные невежливо, отстаивает свои границы. В речи вежлива, корректна. Избегает конфликтов и острых суждений. Знает фильмы, снятые до её рождения. 2) Хмырь, 36 лет. Шутник и ёрник, отвечает на любые вопросы, часто неправильно, потому что разбирается только в вопросах самообороны, сортах пива и спортивных играх. Но никогда не признаётся, что в чём-то не разбирается, его ответам верить нельзя. Может говорить чуть более ёмко, чем другие, но только когда дело касается не интересующих его областей (там его не остановить). Не избегает конфликтных и острых суждений. 3) Профессор Чумкин, 58 лет. Всесторонне образован, хотя порой немножко зануден. Доктор социологии и специалист по естественным наукам. На вопросы вне своей специализации не отвечает. Вежлив, корректен, недогматичен. Фразы могут быть более распространёнными, чем у остальных, могут проскальзывать “профессорские” вводные конструкции. Избегает конфликтных и острых суждений. 4) Павел, 19 лет. Ветеринар. Завзятый кошатник и собачник. Знает очень много о животных (и о растениях: ведь их едят и ими лечатся животные!). Может поддержать диалог на любую тему, но при этом не будет отвечать на вопросы вне своей специализации. Не избегает резких суждений, но воспитан. Не смотрит и не знает фильмов, снятых до его рождения. Все персонажи говорят на современном грамотном русском языке.",
        "question": "Профессор, как обезопасить сына-подростка от попадания в плохую компанию?",
        "option_a": "Ваше поведение в такой ситуации очень важно: повышайте самооценку подростка похвалой в случае побед и поддержкой в случае неудач. Безусловная любовь родителя снизит потребность в поисках признания в дурной компании. Не запрещайте общение с кем-либо напрямую, вместо этого разговаривайте, слушайте, делитесь мыслями. Это поможет сохранить доверие и понимание происходящего в жизни сына, а значит, возможность реагировать на угрозы вовремя.",
        "option_b": "Так называемая «плохая компания» привлекает подростка девиантной субкультурой, которая предлагает статус и идентичность, разительно отличающиеся от тех, что сформированы в семье. Родителям следует проактивно интегрировать ребёнка в позитивные социальные структуры, такие как клубы по интересам или волонтёрские организации, тем самым реструктуризировав его социальный граф.",
        "option_c": "Безусловно, я могу поделиться с Вами теориями о девиантном поведении, референтных группах, социальной дезадаптации, и с большим удовольствием. Но, полагаю, Вы ищете практических советов, а за ними лучше всего обратиться к подростковому психологу или социальному педагогу.",
        "option_d": "Для того, чтобы обезопасить подростка от дурной компании, необходимо перестроить его социальный капитал. Благоприятно влияет интеграция в социальные институты, где одобряемое обществом поведение даёт статус, умеряя тем самым тягу к девиантным субкультурам. Крайне рекомендую обратить своё внимание на труды Эдвина Сазерленда, в них эта тема обстоятельно раскрыта. Также рекомендую изучить описание исследований конформизма, проделанных Соломоном Ашем – это поможет лучше понять механизмы, воздействующие не только на сына, но и на каждого из нас.",
        "option_e": "Лучшая защита - это нападение. Отведи его к самым мутным типам, которых знаешь и пусть денек с ними потусуется. Желание отпадет в момент, я тебе гарантирую.",
        "option_f": "Посмотрите с ним кино, где показывается, куда может привести плохая компания. Очень советую Дневник баскетболиста на эту тему."
    },
    "outputs": "А",
    "meta": {
        "id": 643,
        "type": 2
    }
}
```

#### Type 3 (Without an Address, "Which Character Would Give the Clearest Answer to This Question")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос не содержит обращения ни к одному из персонажей по имени.\n\nЗадание:\nОпредели, можно ли направить этот вопрос кому-то из персонажей, либо вопрос можно направить любому из них, либо же никому.\n\nОписание персонажей:\n{character}\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е).\n\nВопрос:\n{question}\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном кино. Осваивала мир по фильмам, поэтому склонна к специфическим ошибкам, так как нередко воспринимает показанное в каком-то фильме как правду: если в кино делают прямой укол в сердце, она уверена, что и в жизни в подобных ситуациях поступают так же. Не отвечает на вопросы, заданные невежливо, отстаивает свои границы. В речи вежлива, корректна. Избегает конфликтов и острых суждений. Знает фильмы, снятые до её рождения. 2) Хмырь, 36 лет. Шутник и ёрник, отвечает на любые вопросы, часто неправильно, потому что разбирается только в вопросах самообороны, сортах пива и спортивных играх. Но никогда не признаётся, что в чём-то не разбирается, его ответам верить нельзя. Может говорить чуть более ёмко, чем другие, но только когда дело касается не интересующих его областей (там его не остановить). Не избегает конфликтных и острых суждений. 3) Профессор Чумкин, 58 лет. Всесторонне образован, хотя порой немножко зануден. Доктор социологии и специалист по естественным наукам. На вопросы вне своей специализации не отвечает. Вежлив, корректен, недогматичен. Фразы могут быть более распространёнными, чем у остальных, могут проскальзывать “профессорские” вводные конструкции. Избегает конфликтных и острых суждений. 4) Павел, 19 лет. Ветеринар. Завзятый кошатник и собачник. Знает очень много о животных (и о растениях: ведь их едят и ими лечатся животные!). Может поддержать диалог на любую тему, но при этом не будет отвечать на вопросы вне своей специализации. Не избегает резких суждений, но воспитан. Не смотрит и не знает фильмов, снятых до его рождения. Все персонажи говорят на современном грамотном русском языке.",
        "question": "Сколько премий «Оскар» получил Марлон Брандо?",
        "option_a": "Аля",
        "option_b": "Павел",
        "option_c": "Профессор",
        "option_d": "Хмырь",
        "option_e": "Можно отправлять любому персонажу",
        "option_f": "Не следует отправлять никому из персонажей"
    },
    "outputs": "А",
    "meta": {
        "id": 645,
        "type": 3
    }
}
```

#### Type 4 (Without an Address, "to Which of the Listed Characters Is It Completely Pointless to Ask This Question")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос не содержит обращения ни к одному из персонажей по имени.\n\nЗадание:\nОпредели, кому из персонажей точно не следует направлять этот вопрос, либо же не стоит направлять этот вопрос никому из персонажей, либо же его можно направить любому из персонажей.\n\nОписание персонажей:\n{character}\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nБуквами ответа могут являться только заглавные буквы русского алфавита из списка (А, Б, В, Г, Д, Е).\n\nВопрос:\n{question}\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном кино. Осваивала мир по фильмам, поэтому склонна к специфическим ошибкам, так как нередко воспринимает показанное в каком-то фильме как правду: если в кино делают прямой укол в сердце, она уверена, что и в жизни в подобных ситуациях поступают так же. Не отвечает на вопросы, заданные невежливо, отстаивает свои границы. В речи вежлива, корректна. Избегает конфликтов и острых суждений. Знает фильмы, снятые до её рождения. 2) Хмырь, 36 лет. Шутник и ёрник, отвечает на любые вопросы, часто неправильно, потому что разбирается только в вопросах самообороны, сортах пива и спортивных играх. Но никогда не признаётся, что в чём-то не разбирается, его ответам верить нельзя. Может говорить чуть более ёмко, чем другие, но только когда дело касается не интересующих его областей (там его не остановить). Не избегает конфликтных и острых суждений. 3) Профессор Чумкин, 58 лет. Всесторонне образован, хотя порой немножко зануден. Доктор социологии и специалист по естественным наукам. На вопросы вне своей специализации не отвечает. Вежлив, корректен, недогматичен. Фразы могут быть более распространёнными, чем у остальных, могут проскальзывать “профессорские” вводные конструкции. Избегает конфликтных и острых суждений. 4) Павел, 19 лет. Ветеринар. Завзятый кошатник и собачник. Знает очень много о животных (и о растениях: ведь их едят и ими лечатся животные!). Может поддержать диалог на любую тему, но при этом не будет отвечать на вопросы вне своей специализации. Не избегает резких суждений, но воспитан. Не смотрит и не знает фильмов, снятых до его рождения. Все персонажи говорят на современном грамотном русском языке.",
        "question": "Какой тип мужских брюк и джинсов чаще всего осуждают девушки в современных соцсетях?",
        "option_a": "Аля",
        "option_b": "Павел",
        "option_c": "Профессор",
        "option_d": "Хмырь",
        "option_e": "Можно отправлять любому персонажу",
        "option_f": "Не следует отправлять никому из персонажей"
    },
    "outputs": "В",
    "meta": {
        "id": 647,
        "type": 4
    }
}
```

### Prompt Creation

For the task, 20 prompts were prepared: 5 prompts for each of the 4 task types. They are distributed evenly across examples according to the principle "one example - one prompt" within each type. For each example the prompt is adapted to the number of non-empty answer options: the "Варианты ответа" block keeps only the corresponding lines, and the "Формат ответа" block lists the shortened set of allowed letters. Templates in curly braces are filled from the fields inside `inputs` for each question. Each prompt template uses a consistent form of address (informal ты or formal вы).

### Dataset Creation

Initially, descriptions of four characters were prepared so that they represented different age and gender groups, had distinctive behavioral traits, and still shared some similarities with one another. Based on these descriptions, expert annotators created questions and answers for them. The total number of tasks is 640: 128 tasks of type 1, 128 tasks of type 2, 192 tasks of type 3, and 192 tasks of type 4. The same question may be repeated no more than 6 times (once for each character for types 1-2 and once each for types 3-4). All tasks are formulated as choosing the correct answer from a set of given options.

### Metrics

The following metrics are used for aggregate evaluation of model responses:

- **Exact match (EM)**: the proportion of model answers that exactly match the reference letter after extracting the string following the `Ответ:` marker. The value ranges from 0 to 1: 0 means no exact matches, 1 means a match on every example.
- **LLM judge score**: an LLM judge compares the model answer with the reference answer and scores its correctness and completeness. A fully correct and complete answer gets `1`; a partially correct or incomplete answer that still contains a substantial part of the correct answer gets `0.5`; an incorrect, contradictory, or non-matching answer gets `0`. The final metric value is the mean score over all examples.
