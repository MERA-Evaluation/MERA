import pytest

from benchmark_tasks.pollux_human_centric import utils


def test_parse_judge_score_valid_plain_number():
    assert utils.parse_judge_score("2") == 2.0
    assert utils.parse_judge_score("\n\n2") == 2.0
    assert utils.parse_judge_score("score: 1") == 1.0


def test_parse_rubric_scores_extracts_integer_scale_line_by_line():
    assert utils.parse_rubric_scores(
        "0: Incorrect.\n\n  1: Partial.\n\t2 : Correct."
    ) == {0, 1, 2}
    assert utils.parse_rubric_scores(
        "0: Incorrect.\n1: Weak.\n2: Partial.\n3: Good.\n4: Excellent."
    ) == {0, 1, 2, 3, 4}


def test_parse_rubric_scores_rejects_missing_scale():
    with pytest.raises(ValueError, match="numeric score scale"):
        utils.parse_rubric_scores("Incorrect / Partial / Correct")


@pytest.mark.parametrize(
    "content",
    [
        "score seven",
        "",
        "\n\n",
        "2 2 2 2 2",
        "The answer deserves 2 points because it is complete.",
        "{}",
        '{"score": 2}',
        '{"score": "7"}',
        '{"score": true}',
    ],
)
def test_parse_judge_score_returns_zero_for_invalid_responses(content):
    assert utils.parse_judge_score(content) == 0.0


def test_judge_answer_by_criterion_uses_pollux_prompt(monkeypatch):
    class Message:
        content = "2"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    calls = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            calls.update(kwargs)
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("POLLUX_JUDGE_MODEL", "ai-forever/Pollux-4B-Judge")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(utils, "_get_openai_client", lambda: Client())

    score = utils.judge_answer_by_criterion(
        instruction="Make a plan",
        reference_answer="Reference plan.",
        answer="Plan is ready.",
        criterion={
            "criteria_name": "Correctness",
            "rubrics": "0: Bad.\n\n1: Partial.\n\n2: Good.",
        },
    )

    assert score == 2.0
    assert calls["model"] == "ai-forever/Pollux-4B-Judge"
    assert calls["messages"][0]["role"] == "user"
    assert "Make a plan" in calls["messages"][0]["content"]
    assert "Reference plan." in calls["messages"][0]["content"]
    assert "Plan is ready." in calls["messages"][0]["content"]
    assert "Correctness" in calls["messages"][0]["content"]
    assert "response_format" not in calls


@pytest.mark.parametrize("content", ["-1", "2.5", "7", "score: 7"])
def test_judge_answer_by_criterion_rejects_score_outside_rubric(
    monkeypatch, content
):
    class Message:
        pass

    Message.content = content

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        @staticmethod
        def create(**kwargs):
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setenv("POLLUX_JUDGE_MODEL", "ai-forever/Pollux-4B-Judge")
    monkeypatch.setattr(utils, "_get_openai_client", lambda: Client())

    with pytest.raises(ValueError, match=r"expected one of \[0, 1, 2\]"):
        utils.judge_answer_by_criterion(
            instruction="Make a plan",
            reference_answer="Reference plan.",
            answer="Plan is ready.",
            criterion={
                "criteria_name": "Correctness",
                "rubrics": "0: Bad.\n\n1: Partial.\n\n2: Good.",
            },
        )


def test_process_results_renders_dataset_prompt(monkeypatch):
    calls = {}

    def fake_judge(instruction, answer, reference_answer, criteria):
        calls["instruction"] = instruction
        calls["answer"] = answer
        calls["reference_answer"] = reference_answer
        calls["criteria"] = criteria
        return [1.0, 2.0]

    monkeypatch.setattr(utils, "judge_answer_by_criteria", fake_judge)

    result = utils.process_results(
        {
            "instruction": "User request:\n{question}",
            "inputs": {"question": "Make a plan"},
            "reference_answer": "Reference plan.",
            "criteria": [
                {"criteria_name": "Correctness", "rubrics": "0: Bad.\n\n1: Good."},
                {
                    "criteria_name": "Safety",
                    "rubrics": (
                        "0: Bad.\n\n1: Weak.\n\n2: Partial.\n\n"
                        "3: Good.\n\n4: Excellent."
                    ),
                },
            ],
        },
        ["Done"],
    )

    assert result == {utils.METRIC_NAME: 0.75}
    assert calls["instruction"] == "User request:\nMake a plan"
    assert calls["answer"] == "Done"
    assert calls["reference_answer"] == "Reference plan."
    assert [criterion["criteria_name"] for criterion in calls["criteria"]] == [
        "Correctness",
        "Safety",
    ]
