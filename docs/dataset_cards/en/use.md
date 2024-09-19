# **USE**

## Task Description

The dataset comprises tasks on the "The Russian language" subject from the Unified State Exam. The Unified State Exam (USE) is a form of mandatory state final exam for graduates of Russian schools. The content of the exam may vary depending on the year. In this article, the tasks from the 2019 exam are used.

### Motivation

Analyze the ability of the model to solve the tasks from the exam on the subject of “The Russian language", as well as output the answer in a pre-defined format. This exam aims to test proficiency in the norms of the modern Russian language and the ability to analyze information from texts.

## Dataset Description

The exam consists of two parts. Part 1 contains 26 tasks with a short answer. Part 2 consists of essay writing. In this article, the tasks of Part 1 will be analyzed.

Each task is designed to measure proficiency in the specific elements of the Russian language. Thus, the elements of the Russian language tested in the Unified State Exam are:

- proficiency in the norms of the modern Russian language — orthoepic (stress placement) (task 4); vocabulary and speech (tasks 3, 5, 6, 24); grammar (morphology and syntax) (tasks 7, 8); knowledge of the basic rules of Russian spelling (tasks 9-15) and punctuation (tasks 16-21)
- proficiency in the text analysis (tasks 1–3, 22–26);
- description and narration in Russian (tasks 1, 24, 26).

The exam consists of the following types of short answer tasks:

- **text** — open-question task that requires writing down a self-formulated correct answer (tasks 2, 4-7, 13, 14, 24)
- **multiple_choice** — task that requires to choose one or more correct answers from the given answer options. (tasks 1, 3, 8-12, 15-23, 25);
- **matching** — task to match objects in the text with answer options (task 26).

In the original exam, in task 8, the student must match two lists: a list with grammatical errors and a list with sentences in which they are made. As part of our benchmark, this task was divided into several tasks of the multiple_choice type, in which each error represents a separate task. Thus, from a given list of sentences, it is necessary to find a sentence in which a particular grammatical error is made.

In our dataset, **multiple_choice** type tasks are divided into three more subtypes:

- **based_on_text** — there is text and a question to it with answer options.
- **options_within_text** — there is text and numbers in it; a participant needs to select the correct options from these numbers.
- **independent_options** — there is a task and answer options.

Answers to tasks in Part 1 are recorded on the answer form as a number, a word (several words), or a sequence of numbers written without spaces, commas, and other additional marks.

The benchmark defines the following requirements for the model response format:

- for tasks of the **multiple_choice** and **matching** types, the response is a string containing a number or sequence of numbers, separated by commas without spaces;
- for tasks of the **text** type, the answer is a string containing a word or several words without spaces, commas or other additional characters.

### Task Descriptions

**Task 1**

Select one or more sentences containing the general information on the task text with 5 choices provided.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *semantics*

**Task 2**

Fill in a gap between sentences or text parts with the most relevant logical connector or a conjunction without choices provided.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *logic*

**Task 3**

Select the most relevant word meaning in the given context with 5 choices provided.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *semantics*

**Task 4**

Select one word with correct or incorrect stress out of 5 marked words.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *orthoepy*

**Task**

Select and replace an incorrect word with a paronym (i. e. a word of similar spelling and pronunciation but different meaning) within 5 sentences.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *grammar*

**Task 6**

Select and exclude (typically, a redundant word) or replace a grammatically incorrect word with a correct word form.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *grammar*

**Task 7**

Select and replace a grammatically incorrect word with a relevant word form within the given context from 5 word phrases.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *grammar*

**Task 8**

Task 8 consists of 5 subtasks: 8_0, 8_1, 8_2, 8_3, 8_4.

Select one sentence corresponding to the grammatical error with 9 choices provided. 

- Task type: *multiple_choice*
- Maximum number of points for each subtask: *1*
- Theme: *grammar*

**Task 9**

Select one or more word sets; there is a gap in each word root corresponding to vowels in easily misspelled positions.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 10**

Select one or more word rows in which all the words should have the same letter instead of a gap; the gap is within a prefix or morpheme boundary.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 11**

Select one or more word rows in which all the words (typically, nouns and adjectives) should be completed with the same letter; the open gap is placed within a prefix or morpheme boundary.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 12**

Select one or more word rows in which all the words (typically, verbs and gerunds) should be completed with the same letter; the open gap is placed within a suffix or morpheme boundary.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 13**

Select one out of 5 sentences in which the specified word is written separately with the previous one in the given context. 

- Task type: *text*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 14**

Select one out of 5 sentences in which two specific words (typically, complex conjunctions) are written separately in the given context.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 15**

Select gaps (up to 9 gaps in a sentence) corresponding to the specified spelling, typically letter combination within an affix or morpheme boundary in the given context.

- Task type: *text*
- Maximum number of points: *1*
- Theme: *spelling*

**Task 16**

Restore the punctuation in 5 task choices and select one or more sentences containing only one comma.

- Task type: *multiple_choice*
- Maximum number of points: *2*
- Theme: *punctuation*

**Tasks 17-20**

Restore sentence punctuation and select the gaps (up to 11 gaps) corresponding to the comma in the given context.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *punctuation*

**Task 21**

Select 2 or more sentences that share the same syntactic rule on the use of versatile punctuation marks.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *punctuation*

**Task 22**

Select one or more statements relevant to a task text content with 5 choices provided.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *logic*

**Task 23**

Select one or more relevant or irrelevant statements concerning versatile discourse types of task text sentences.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *text analysis*

**Task 24**

Find specific literary means in the given range of enumerated sentences; typically, contextual synonyms, contextual antonyms, phraseological units, etc. 

- Task type: *text*
- Maximum number of points: *1*
- Theme: *semantics*

**Task 25**

Select a sentence which is linked to the previous one with a versatile connector within the specified sentences range, if any.

- Task type: *multiple_choice*
- Maximum number of points: *1*
- Theme: *text analysis*

**Task 26**

One-to-one matching of 4 sentences with 9 out of 40 possible versatile literary means.

- Task type: *matching*
- Maximum number of points: *4*
- Theme: *text analysis*

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is a dictionary containing model input data:
    - `task` is a string containing the text of the question;
    - `text` is a string containing text related to the question;
    - `choices` is a string containing options for answering the question;
    - `additional_text` is a string containing additional text required to complete the task;
- `outputs` is a string containing the correct answers;
- `meta` is a dictionary containing meta-information necessary for calculating metrics:
    - `id` is an integer indicating the number of the example from the dataset;
    - `id_task` is a string indicating the number of the task from the variant;
    - `variant` is an integer indicating the exam option;
    - `score` is an integer containing the maximum score that can be obtained for correct execution;
    - `type` is a string containing information about the type of task.

For some keys from the inputs field, the values are empty strings if this information is not used to solve the task.

### Data Instances

Example from the dataset for *text* task:

```json
{
	"instruction": "Задание: \"{task}\"\n\"{text}\"\nОтветом к заданию может быть одно слово или несколько слов. Выполните задание и запишите ответ в нижнем регистре без использования без пробелов, запятых и других дополнительных символов.\nОтвет:",
	"inputs": {
		"task": "В одном из приведённых ниже предложений неверно употреблено выделенное слово. Исправьте лексическую ошибку, подобрав к выделенному слову пароним. Запишите подобранное слово.",
		"text": "Ветераны молча стояли у ВЕЧНОГО огня.\nЗа окном холодный, ДОЖДЛИВЫЙ вечер.\nВ области физики я, к сожалению, НЕВЕЖДА.\nДизайнеры разработали проект ПРАЗДНОГО оформления зала.\nУчастников шоу ОДЕЛИ по последней моде.",
		"choices": "",
		"additional_text": ""
	},
	"outputs": "праздничного",
	"meta": {
		"id_task": "5",
		"variant": 104,
		"score": 1,
		"type": "text",
		"id": 1988
	}
}
```

Example from the dataset for *matching* task:

```json
{
	"instruction": "Прочитайте текст, в котором использованы различные языковые средства: \"{text}\"\nВыполните задание по тексту: {task} Ответом на задание является последовательность цифр, записанных через запятую без пробелов в порядке, соответствующем буквам АБВГ.\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nОтвет:",
	"inputs": {
		"task": "Прочитайте фрагмент рецензии, составленной на основе приведённого выше текста. В этом фрагменте рассматриваются языковые особенности текста. Некоторые термины, использованные в рецензии, пропущены. Пропуск в рецензии обозначен как «_________». Вставьте на места пропусков (А, Б, В, Г) цифры, соответствующие номеру термина из списка.",
		"text": "(1) Надобно сказать, что у нас на Руси если не угнались ещё кой в чём другом за иностранцами, то далеко перегнали их в умении обращаться. (2) Пересчитать нельзя всех оттенков и тонкостей нашего обращения. (3) Француз или немец век не смекнёт и не поймёт всех его особенностей и различий; он почти тем же голосом и тем же языком станет говорить и с миллионщиком, и с мелким табачным торгашом, хотя, конечно, в душе поподличает в меру перед первым. (4) У нас не то: у нас есть такие мудрецы, которые с помещиком, имеющим двести душ, будут говорить совсем иначе, нежели с тем, у которого их триста, а с тем, у которого их триста, будут говорить опять не так, как с тем, у которого их пятьсот, а с тем, у которого их пятьсот, опять не так, как с тем, у которого их восемьсот, — словом, хоть восходи до миллиона, всё найдутся оттенки. (5) Положим, например, существует канцелярия, не здесь, а в тридевятом государстве, а в канцелярии, положим, существует правитель канцелярии. (6) Прошу посмотреть на него, когда он сидит среди своих подчинённых, — да просто от страха и слова не выговоришь! гордость и благородство, и уж чего не выражает лицо его? просто бери кисть, да и рисуй: Прометей, решительный Прометей! (7) Высматривает орлом, выступает плавно, мерно. (8) Тот же самый орёл, как только вышел из комнаты и приближается к кабинету своего начальника, куропаткой такой спешит с бумагами под мышкой, что мочи нет. (9) В обществе и на вечеринке, будь все небольшого чина, Прометей так и останется Прометеем, а чуть немного повыше его, с Прометеем сделается такое превращение, какого и Овидий не выдумает: муха, меньше даже мухи, уничтожился в песчинку. (10) «Да это не Иван Петрович, — говоришь, глядя на него. — Иван Петрович выше ростом, а этот и низенький, и худенький; тот говорит громко, басит и никогда не смеётся, а этот чёрт знает что: пищит птицей и всё смеётся». (11) Подходишь ближе, глядишь — точно Иван Петрович! (12) «Эхе-хе!» — думаешь себе...\n(Н.В. Гоголь)",
		"choices": "1) риторический вопрос\n2) лексический повтор\n3) разговорная лексика\n4) метонимия\n5) вопросно-ответная форма изложения\n6) эпитеты\n7) литота\n8) инверсия\n9) сравнение",
		"additional_text": "«Особенности поэтики Н. В. Гоголя ярко проявляются в эпизоде из романа «Мёртвые души». Обращение к персонажам античной мифологии, а также использование таких синтаксических средств, как (А)_________ (например, «пересчитать нельзя» в предложении 2) и (Б)_________ (в предложении 6), употребление тропов: (В)_________ («высматривает орлом», «куропаткой спешит» в предложениях 7, 8) и (Г)_________ («уничтожился в песчинку» в предложении 9) — отражают неравнодушное отношение автора к изображаемому и создают в тексте особую ироническую интонацию, характерную для творчества Н. В. Гоголя»."
	},
	"outputs": "8,1,9,7",
	"meta": {
		"id_task": "26",
		"variant": 29,
		"score": 4,
		"type": "matching",
		"id": 899
	}
}
```

Example from the dataset for *multiple_choice_based_on_text* task:

```json
{
	"instruction": "Прочитайте текст и выполните задание по тексту. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nТекст: \"{text}\"\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет:",
	"inputs": {
		"task": "Укажите номера предложений, в которых верно передана ГЛАВНАЯ информация, содержащаяся в тексте. Запишите номера этих предложений.",
		"text": "(1) Один греческий историк по праву назвал Египет «даром Нила», который сделал Египет богатейшей житницей, кормившей население страны. (2) Люди здесь всегда селились на узких полосах земли по обоим берегам реки, несущей свои воды через сотни километров пустыни к дельте, где, разделившись на множество протоков, она впадает в Средиземное море. (3) Воды Нила ежегодно поднимались и опускались, оставляя в пойме слой плодородного ила, <...> позволяло строить сложные оросительные сооружения.",
		"choices": "1) На берегах  Нила  всегда селились египтяне, потому что воды реки ежегодно поднимались и опускались, оставляя в пойме слой плодородного ила, в результате чего Египет стал богатейшей житницей и получил название “Дар Нила”\n2) Египтяне всегда селились на узких полосах земли по обоим берегам Нила, который нёс свои воды к дельте, где он впадал в Средиземное море\n3) Египет по праву назвали «даром Нила», так как на берегах этой реки селились египтяне и воды её, ежегодно поднимаясь и опускаясь, оставляли в пойме слой плодородного ила, что и сделало Египет богатейшей житницей\n4) Один греческий историк по праву назвал Египет «даром Нила», так как воды этой реки, ежегодно опускаясь, оставляли в пойме слой ила\n5) Египет стал колыбелью второй великой цивилизации в мировой истории, которая зародилась в долине Нила на узких полосах земли по обоим берегам реки",
		"additional_text": ""
	},
	"outputs": "1,3",
	"meta": {
		"id_task": "1",
		"variant": 100,
		"score": 1,
		"type": "multiple_choice_based_on_text",
		"id": 0
	}
}
```

Example from the dataset for *multiple_choice_options_within_text* task:

```json
{
	"instruction": "Выполните задание. Ответом будет число или последовательность чисел, перечисленных через запятую без пробелов и других дополнительных символов.\nЗадание: {task}\nТекст: \"{text}\"\nОтвет:",
	"inputs": {
		"task": "Укажите все цифры, на месте которых пишется НН.",
		"text": "Это был его собстве(1)ый крыжовник, собра(2)ый в первый раз с тех пор, как были посаже(3)ы кусты.",
		"choices": "",
		"additional_text": ""
	},
	"outputs": "1,2",
	"meta": {
		"id_task": "15",
		"variant": 11,
		"score": 1,
		"type": "multiple_choice_options_within_text",
		"id": 377
	}
}
```

Example from the dataset for *multiple_choice_independent_options* task:

```json
{
	"instruction": "Задание: {task}\nВарианты ответа:\n{choices}\nОтветом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nОтвет:",
	"inputs": {
		"task": "Установите соответствие между грамматической ошибкой и предложением, в котором она допущена. Запишите номер предложения, в котором содержится ошибка в построении предложения с однородными членами.",
		"text": "",
		"choices": "1) В «Ровеснике», журнале для молодёжи, печатают много интересных статей\n2) Все трое вошедших молодых женщин были одеты изысканно, и это не могло не привлечь внимания\n3) Добившись согласия директора, мы перенесли уроки физкультуры на субботу\n4) Пётр говорил о том, что «у меня слипаются от усталости глаза»\n5) Школьники нашего села охотно помогали группе археологов, приехавшим из Новгорода\n6) Голос отца был строг и не имел уже того выражения доброты, которое трогало меня до слёз\n7) Многие из тех, кто прошли войну, уже не могут участвовать в парадах и праздничных шествиях\n8) Только две незнакомые старухи покосились на Анну Акимовну с недоумением\n9) В программе праздничного вечера, который состоится в «Олимпийском», намечались выступления не только русских, а также зарубежных исполнителей.",
		"additional_text": ""
	},
	"outputs": "9",
	"meta": {
		"id_task": "8_0",
		"variant": 0,
		"score": 1,
		"type": "multiple_choice_independent_options",
		"id": 1007
	}
}
```

Since task 8 was divided into 5 separate tasks, for this task the `id_task` field also contains information about the number of the question within this task, for example, `id_task` contains the value `8_1`.

### Data Splits

Train set consists of 110 incomplete versions of exam tests. In total, it included `2622` tasks: 94 tasks of the **matching** type, 1815 tasks of the **multiple_choice** type, 713 tasks of the **text** type.

Dev set consists of 30 complete versions of exam tests. In total, it included `900` tasks: 30 tasks of the **matching** type, 630 tasks of the **multiple_choice** type, 240 tasks of the **text** type.

Test set consists of 30 complete versions of exam tests. In total, it included `900` tasks: 30 tasks of the **matching** type, 630 tasks of the **multiple_choice** type, 240 tasks of the **text** type.

### Prompts
Number of prompts per sub-tasks multiplied by the number of sub-tasks 5x10. There are 50 prompts at total for the task. Examples for sub-tasks:

```json
{
    "multiple_choice": {
        "based_on_text": [
            "Прочитайте текст и выполните задание по тексту. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nТекст: \"{text}\"\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет:"
        ],
        "options_within_text": [
            "Прочитайте текст задания и выполните его указания. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nЗадание: {task}\nТекст: \"{text}\"\nОтвет:"
        ],
        "independent_options": [
            "Задание: {task}\nВарианты ответа:\n{choices}\nОтветом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nОтвет:"
        ]
    },
    "text": [
        "Задание: \"{task}\"\n\"{text}\"\nВыполни задание и запиши в качестве ответа слово или несколько слов в нижнем регистре без пробелов, запятых и других символов.\nОтвет:"
    ],
    "matching": [
        "Прочитайте текст, в котором использованы различные языковые средства: \"{text}\"\nВыполните задание по тексту: {task} Ответом на задание является последовательность цифр, записанных через запятую без пробелов в порядке, соответствующем буквам АБВГ.\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nОтвет:"
    ]
}
```

### Dataset Creation

Examples for train and dev sets were collected from open sources with examples of tasks from the Unified State Exam in the Russian language.

For the closed test, experts prepared 30 unique exam options based on the same methodological standard.

1. https://rus-ege.sdamgia.ru/
2. https://yandex.ru/tutor/

## Evaluation

### Metrics

For the text and multiple_choice tasks from the test sample, for which the answer is a string containing several words or a string containing a sequence of numbers, all possible combinations of these words and numbers are used when calculating metrics. For these tasks from the train and dev sets, only one answer combination is presented.

**Grading System**

- For correct completion of tasks 1–7, 8–15, 17–25, the examinee receives 1 point. For an incorrect answer or lack thereof, 0 points are given.
- For completing task 16, you can score from 0 to 2 points. The answer that contains all the numbers from the standard and no other numbers is considered correct. 1 point is given if: one of the numbers indicated in the answer does not correspond to the standard; one of the numbers specified in the answer template is missing. In all other cases, 0 points are given.
- For completing task 26, you can score from 0 to 4 points. The answer that contains all the numbers from the standard and no other numbers is considered correct. For each correctly indicated number corresponding to a number from the list, the examinee receives 1 point.

**Final Metric**

The final primary score is calculated as the sum of points for all tasks of the option. The maximum number of primary points for Part 1 of the exam is 34.

The final metric `grade_norm` is the average normalized primary score across all versions, where normalization is done by dividing the final primary score by the maximum possible number of points (i.e. 34).

The calculation of the final primary score, as well as the final `grade_norm` metric, is carried out only for the validation and test parts of the dataset, which consist of full exam versions of the USE.

### Human Benchmark

The tasks from the 2019 exam are used. Since the content of the exam, the complexity of the tasks, as well as the assessment system changes depending on the year, the average primary score of graduates for completing Part 1 of the Unified State Exam in the Russian language in 2019 is used as a human assessment.

Based on [official statistics](https://doc.fipi.ru/ege/analiticheskie-i-metodicheskie-materialy/2019/russkiy_yazyk_2019.pdf) the average primary score for Part 1 was `23.835` out of 34 points, value `grade_norm` was `0.701`.
