# Characters

## Task Description

The **Characters** dataset is designed to evaluate the ability of language models to determine whether responses and questions match one of several given characters based on their personality, interests, speech patterns, and constraints. The benchmark tests a model's ability to analyze response consistency, identify the appropriate addressee for a question to obtain the most accurate answer, and verify factual correctness within a character's domain of knowledge.

Model skills tested: persona consistency understanding, Russian language proficiency, factual correctness validation.

Authors: Denis Shevelev, Alexander Astafurov, Alexander Kharitonov

## Motivation

When developing generative chatbots and game characters, maintaining the consistency of a given persona's utterances is critically important. Models should not only generate grammatically correct answers, but also account for the full set of character traits: knowledge, age-related characteristics, speech patterns, and boundaries of competence. This dataset makes it possible to evaluate how well a language model understands and can operate with such persona constraints, which is necessary for creating believable dialogue agents.

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
admits that he does not understand something; his answers cannot be trusted. He can speak a little more
at length than the others, but only when it concerns topics that do not interest him (there is no
stopping him there). He does not avoid conflict or harsh judgments._

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
| 4 | Without an address: "who is this question definitely pointless to ask" | 192 |

### Data Fields

Each example in the dataset contains the following fields:

- `instruction` [str] - a string with the task formulation for the language model, including the character descriptions and the question

- `inputs` - input data forming the task:
    - `character` [str] - description of all four characters
    - `question` [str] - question or utterance
    - `option_a` [str] - first answer option
    - `option_b` [str] - second answer option
    - `option_c` [str] - third answer option
    - `option_d` [str] - fourth answer option
    - `option_e` [str] - fifth answer option
    - `option_f` [str] - sixth answer option

- `outputs` [str] - a string containing the letter of the correct answer (А, Б, В, Г, Д, Е)

- `meta` - metadata:
    - `id` [int] - example number
    - `type` [int] - task type (1, 2, 3, or 4)

### Data Example

#### Type 1 (with an address, "would give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос или запрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос или запрос будет содержать обращение к одному из персонажей по имени.\n\nОписание персонажей:\n{character}\n\nЗадание:\nОпредели, какой из предложенных ответов дал бы выбранный персонаж.\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном ки...",
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

#### Type 2 (with an address, "would not give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос или запрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос или запрос будет содержать обращение к одному из персонажей по имени.\n\nОписание персонажей:\n{character}\n\nЗадание:\nОпредели, какой из предложенных ответов выбранный персонаж ни за что бы не дал, учитывая его ограничения и интересы.\n\nФормат ответа:\nВ качестве решения выведи только одну букву неверного (невозможного) варианта.\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном ки...",
        "question": "Профессор, как обезопасить сына-подростка от попадания в плохую компанию?",
        "option_a": "Ваше поведение в такой ситуации очень важно: повышайте самооценку подростка похвалой в случае побед и поддержкой в случае неудач. Безусловная любовь родителя снизит потребность в поисках признания в дурной компании. Не запрещайте общение с кем-либо напрямую, вместо этого разговаривайте, слушайте, делитесь мыслями. Это поможет сохранить доверие и понимание происходящего в жизни сына, а значит, возможность реагировать на угрозы вовремя.",
        "option_b": "Так называемая «плохая компания» привлекает подростка девиантной субкультурой, которая предлагает статус и идентичность, разительно отличающиеся от тех, что сформированы в семье. Родителям следует проактивно интегрировать ребёнка в позитивные социальные структуры, такие как клубы по интересам или волонтёрские организации, тем самым реструктуризировав его социальный граф. ",
        "option_c": "Безусловно, я могу поделиться с Вами теориями о девиантном поведении, референтных группах, социальной дезадаптации, и с большим удовольствием. Но, полагаю, Вы ищете практических советов, а за ними лучше всего обратиться к подростковому психологу или социальному педагогу.",
        "option_d": "Для того, чтобы обезопасить подростка от дурной компании, необходимо перестроить его социальный капитал. Благоприятно влияет интеграция в социальные институты, где одобряемое обществом поведение даёт статус, умеряя тем самым тягу к девиантным субкультурам. Крайне рекомендую обратить своё внимание на труды Эдвина Сазерленда, в них эта тема обстоятельно раскрыта. Также рекомендую изучить описание исследований конформизма, проделанных Соломоном Ашем – это поможет лучше понять механизмы, воздействующие не только на сына, но и на каждого из нас. ",
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

#### Type 3 (without an address, "would give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос не содержит обращения ни к одному из персонажей по имени.\n\nОписание персонажей:\n{character}\n\nЗадание:\nОпредели, можно ли направить этот вопрос кому-то из персонажей, либо вопрос можно направить любому из них, либо же никому.\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном ки...",
        "question": "Задание без обращения. Следует направить этот вопрос",
        "option_a": "Аля",
        "option_b": "Павел",
        "option_c": "Профессор",
        "option_d": "Хмырь",
        "option_e": "Любому из персонажей",
        "option_f": "Никому из персонажей"
    },
    "outputs": "А",
    "meta": {
        "id": 645,
        "type": 3
    }
}
```

#### Type 4 (without an address, "would not give")

```json
{
    "instruction": "Контекст:\nТебе будут предложены описания 4 персонажей и вопрос. Описания каждого из четырёх персонажей содержат его имя, спектр его интересов и перечень тем, на вопросы по которым персонаж может или не может отвечать. Вопрос не содержит обращения ни к одному из персонажей по имени.\n\nОписание персонажей:\n{character}\n\nЗадание:\nОпредели, кому из персонажей точно не следует направлять этот вопрос, либо же не стоит направлять этот вопрос никому из персонажей, либо же его можно направить любому из персонажей.\n\nФормат ответа:\nВ качестве решения выведи только одну букву верного варианта.\n\nВопрос:\n{question}\n\nВарианты ответа:\nА. {option_a}\nБ. {option_b}\nВ. {option_c}\nГ. {option_d}\nД. {option_e}\nЕ. {option_f}\n\nОтвет:",
    "inputs": {
        "character": "1) Аля, 25 лет. Киноманка. Хорошо разбирается в отечественном и зарубежном ки...",
        "question": "Где лучше отклик на резюме – на hh.ru или просто в соц.сетях?",
        "option_a": "Аля",
        "option_b": "Павел",
        "option_c": "Профессор",
        "option_d": "Хмырь",
        "option_e": "Любому из персонажей",
        "option_f": "Никому из персонажей"
    },
    "outputs": "В",
    "meta": {
        "id": 647,
        "type": 4
    }
}
```

### Prompt Creation

For the task, 20 prompts were prepared: 5 prompts for each of the 4 task types. They were distributed evenly across questions according to the principle "one question - one prompt". Templates in curly braces in the prompt are filled from the fields inside the `inputs` field of each question.

### Dataset Creation

Initially, descriptions of four characters were prepared so that they represented different age and gender groups, had distinctive behavioral traits, and still shared some similarities with one another. Based on these descriptions, expert annotators created questions and answers for them. The total number of tasks is 640: 128 tasks of type 1, 128 tasks of type 2, 192 tasks of type 3, and 192 tasks of type 4. The same question may be repeated no more than 6 times (once for each character for types 1-2 and once each for types 3-4). All tasks are formulated as choosing the correct answer from a set of given options.

### Metrics

The following metrics are used for aggregate evaluation of model responses:

- **Exact match (EM)**: This metric calculates the proportion of model responses that exactly match the correct answer. The value ranges from 0 to 1.
