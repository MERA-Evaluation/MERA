# ruAIME


## Task description

This dataset contains problems from the American Invitational Mathematics Examination (AIME) from 1983 to 2025. The problems cover all key areas of advanced mathematics: algebra, geometry, calculus, combinatorics, number theory, probability, and statistics. They require the model to have a deep understanding of mathematical concepts, the ability to perform multi-step transformations, identify non-obvious connections, and apply creative problem-solving approaches. The dataset is designed to evaluate models' capabilities for complex mathematical reasoning in conditions approximating olympiad-level competitions.

Evaluated skills: Mathematical reasoning, Problem solving, Algebraic manipulation, Multistep deduction, Pattern recognition

Contributors: Vladislav Savenkov, Anton Gabov, Doubletapp


## Motivation

The dataset is focused on evaluating the abilities of advanced models to solve non-standard, high-level mathematical problems from various domains. It is NOT suitable for evaluating basic calculation or text models that lack developed logical-mathematical abilities.

The target users are AI researchers assessing model capabilities in mathematical thinking and developers of educational systems. The evaluation results will show whether a model can act as an 'olympiad problem solver,' which is critical for creating intelligent tutors and assistants.

Evaluated Capabilities:
1. **Deep Understanding of Mathematical Text**: The model must correctly interpret complex problem statements, including modules, radicals, parameters, and geometric constructions.
2. **Strategy Selection**: The ability to determine which mathematical apparatus (theorem, method, technique) to apply in a given situation.
3. **Multi-Step Reasoning**: The ability to break down a complex problem into a sequence of logical steps while preserving intermediate results.
4. **Computational Accuracy**: Minimizing errors in arithmetic and algebraic transformations on the path to the final answer.

The problem design (requiring a precise numerical answer) allows for an unambiguous evaluation of the solution's correctness. This format eliminates subjectivity in checking and focuses the assessment on the final result of the model's reasoning. The metric—the proportion of correct answers (accuracy)—is strict and interpretable: it shows what fraction of problems in the set the model can solve completely and without errors.


## Data description

### Data fields

Each dataset question includes data in the following fields:

- `instruction` [str] — Instruction for solving the problem;
- `inputs` — Input data that forms the task for the model. Can include one or multiple modalities - video, audio, image, text.
    - `question` [str] — Mathematical problem statement;
- `outputs` [str] — Numeric answer to the problem;
- `meta` — Metadata related to the test example, not used in the question (hidden from the tested model).
    - `id` [int] — Unique problem identifier in dataset;
    - `url` [str] — URL of original problem on AoPS Wiki;
    - `categories` — Categorial features characterizing the test example.
        - `contest_type` [str] — Type of mathematical competition;
        - `year` [int] — Contest year;
        - `problem_id` [int] — Problem number in contest;
        - `math_domain` [str] — Mathematical domain of the problem;
    - `solutions` — Solutions to the problem, can contain multiple solution variants.
        - `solution_i` — One solution variant for the problem.
            - `id` [int] — Solution variant identifier;
            - `text` [str] — Problem solution text


### Data formatting example

```json
{
    "instruction": "ТОЛЬКО РЕЗУЛЬТАТ! Разберись с задачей: {question}. Выведи только числовой результат, комментариев и пояснений давать не нужно!",
    "inputs": {
        "question": "В окружности проведены две пересекающиеся хорды. Радиус окружности равен 5, BC=6, и AD делится пополам хордой BC. Найдите произведение mn, где m/n - несократимая дробь, представляющая синус центрального угла малой дуги AB."
    },
    "outputs": "175",
    "meta": {
        "id": 999,
        "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_Problems/Problem_0",
        "categories": {
            "contest_type": "AIME",
            "year": 2024,
            "problem_id": 0,
            "math_domain": "Геометрия"
        },
        "solutions": {
            "solution_i": {
                "id": 1,
                "text": "Пусть O - центр окружности. Поскольку BC=6, расстояние от O до BC равно 4. Рассмотрим геометрическое место середин хорд из A - это окружность с диаметром AO. Условие единственности означает, что эта окружность касается BC. Проведя необходимые вычисления, получаем sin(∠AOB) = 7/25. Таким образом, m=7, n=25, и их произведение равно 175."
            }
        }
    }
}
```


### Prompts

For the task, 10 prompts were prepared and evenly distributed among the questions on the principle of "one prompt per question". The templates in curly braces in each prompt are filled in from the fields inside the `inputs` field in each question.

Prompt example:

```
ТОЛЬКО ЧИСЛО! Реши задачу: {question}. В ответе напиши только число, без объяснений!
```


### Dataset creation

The dataset was created based on official AIME materials from 1983 to 2025. The data source was the Art of Problem Solving (AoPS) resource.

The collection and validation process included:
1. Extracting problem statements, their solutions, and answers from HTML pages.
2. Semi-automated verification of the correctness of extracted numerical answers and formulas.
3. Manual verification of a random sample of problems for each year to ensure correspondence with the original.
4. Translating the problem statements and solutions into Russian.
5. Structuring the data into a unified JSON format with clear separation into problem statements, solutions, and meta-information.

The translation into Russian was performed using the DeepSeek Reasoner API. The main objective was to create a parallel English-Russian corpus of mathematical texts while preserving formula accuracy and language naturalness.

The translation process was carried out by a specialized script that processed text files with problems. The model received clear instructions to translate only the text between special markers, preserving all mathematical expressions in LaTeX format, boxed{} tags, and the original structure. The script was configured to handle large volumes of data with the capability to pause and resume processing.

For quality control, selective manual verification of translations was conducted, which confirmed the correct processing of mathematical expressions and compliance with requirements. As a result, a complete parallel corpus of high-quality translations was obtained, ready for training Russian-language models.


## Evaluation


### Metrics

Metrics for aggregated evaluation of responses:

- `Accuracy`: Accuracy is the proportion of correct model predictions among the total number of cases processed.
