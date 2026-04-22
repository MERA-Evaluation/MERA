import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from mera_openjudge import OpenJudgeScorer, load_judge_config
from mera_openjudge.exceptions import JudgeConfigError, JudgeParseError
from mera_openjudge.parsing import parse_judge_score
from mera_openjudge.prompting import render_judge_prompt


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompts):
        batch_size = len(prompts)
        batch = self.responses[:batch_size]
        self.responses = self.responses[batch_size:]
        return batch


class _MockJudgeHandler(BaseHTTPRequestHandler):
    responses = []

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        _ = self.rfile.read(content_length)
        response_text = self.responses.pop(0)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": response_text,
                    }
                }
            ]
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


class OpenJudgeTests(unittest.TestCase):
    def test_load_judge_config_validates_scales(self):
        config = load_judge_config(
            {
                "criteria": [
                    {
                        "key": "correctness",
                        "name": "Correctness",
                        "scale": {"0": "bad", "1": "good"},
                    },
                    {
                        "key": "fluency",
                        "name": "Fluency",
                        "scale": {0: "bad", 1: "good"},
                    },
                ]
            }
        )
        self.assertEqual(config.allowed_scores, (0, 1))
        with self.assertRaises(JudgeConfigError):
            load_judge_config(
                {
                    "criteria": [
                        {
                            "key": "correctness",
                            "name": "Correctness",
                            "scale": {"bad": "nope"},
                        }
                    ]
                }
            )
        with self.assertRaises(JudgeConfigError):
            load_judge_config(
                {
                    "criteria": [
                        {
                            "key": "correctness",
                            "name": "Correctness",
                            "scale": {0: "bad", 1: "ok"},
                        },
                        {
                            "key": "fluency",
                            "name": "Fluency",
                            "scale": {0: "bad", 2: "ok"},
                        },
                    ]
                }
            )

    def test_render_prompt_supports_default_and_custom_template(self):
        judge_config = load_judge_config(
            {
                "criteria": [
                    {
                        "key": "correctness",
                        "name": "Correctness",
                        "scale": {0: "bad", 1: "good"},
                    }
                ],
                "reference_field": "meta.reference_answer",
            }
        )
        doc = {
            "instruction": "Answer the question: {question}",
            "inputs": {"question": "2+2?"},
            "meta": {"reference_answer": "4"},
        }
        prompt = render_judge_prompt(
            doc=doc,
            answer="4",
            criterion=judge_config.criteria[0],
            judge_config=judge_config,
        )
        self.assertIn("Answer the question: 2+2?", prompt)
        self.assertIn("4", prompt)

        custom_config = load_judge_config(
            {
                "prompt_template": "{{ criterion_name }}|{{ answer }}|{{ reference_answer }}",
                "criteria": [
                    {
                        "key": "correctness",
                        "name": "Correctness",
                        "scale": {0: "bad", 1: "good"},
                    }
                ],
            }
        )
        custom_prompt = render_judge_prompt(
            doc=doc,
            answer="4",
            criterion=custom_config.criteria[0],
            judge_config=custom_config,
        )
        self.assertEqual(custom_prompt, "Correctness|4|4")

    def test_parse_judge_score(self):
        self.assertEqual(parse_judge_score("2", [0, 1, 2]), 2)
        self.assertEqual(parse_judge_score("2\nОбоснование", [0, 1, 2]), 2)
        self.assertEqual(parse_judge_score("Оценка: 2", [0, 1, 2]), 2)
        with self.assertRaises(JudgeParseError):
            parse_judge_score("No score here", [0, 1, 2])

    def test_score_answers_aggregates_metrics(self):
        scorer = OpenJudgeScorer(
            judge_config={
                "criteria": [
                    {
                        "key": "correctness",
                        "name": "Correctness",
                        "scale": {0: "bad", 1: "ok", 2: "good"},
                    },
                    {
                        "key": "fluency",
                        "name": "Fluency",
                        "scale": {0: "bad", 1: "ok", 2: "good"},
                    },
                ]
            },
            backend=FakeBackend(["2", "1", "1", "2"]),
        )
        metrics = scorer.score_answers(
            docs=[
                {"instruction": "Q: {question}", "inputs": {"question": "x"}},
                {"instruction": "Q: {question}", "inputs": {"question": "y"}},
            ],
            answers=["a", "b"],
        )
        self.assertEqual(
            metrics,
            [
                {"judge_correctness": 2.0, "judge_fluency": 1.0, "judge_avg": 1.5},
                {"judge_correctness": 1.0, "judge_fluency": 2.0, "judge_avg": 1.5},
            ],
        )

    def test_openai_compatible_backend_with_mock_server(self):
        _MockJudgeHandler.responses = ["2\nok"]
        server = HTTPServer(("127.0.0.1", 0), _MockJudgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            scorer = OpenJudgeScorer(
                judge_config={
                    "criteria": [
                        {
                            "key": "correctness",
                            "name": "Correctness",
                            "scale": {0: "bad", 1: "ok", 2: "good"},
                        }
                    ]
                },
                runtime_config={
                    "backend": "openai_compatible",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "model_name": "pollux",
                    "max_new_tokens": 16,
                },
            )
            metrics = scorer.score_answers(
                docs=[{"instruction": "Q: {question}", "inputs": {"question": "x"}}],
                answers=["a"],
            )
            self.assertEqual(metrics[0]["judge_avg"], 2.0)
            self.assertEqual(metrics[0]["judge_correctness"], 2.0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
