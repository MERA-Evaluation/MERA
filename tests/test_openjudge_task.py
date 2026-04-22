import json
import unittest
from pathlib import Path
import shutil

from modules.scoring.src.dataset.dataset import Dataset
from modules.scoring.src.tasks.openjudge_task import OpenJudgeTask


class FakeScorer:
    def score_answers(self, docs, answers):
        del docs, answers
        return [
            {"judge_correctness": 2.0, "judge_style": 1.0, "judge_avg": 1.5},
            {"judge_correctness": 1.0, "judge_style": 2.0, "judge_avg": 1.5},
        ]


class SyntheticOpenJudgeTask(OpenJudgeTask):
    def __init__(self, conf, task_conf_path, judge_conf_path):
        self._task_conf_path = task_conf_path
        self._judge_conf_path = judge_conf_path
        super().__init__(conf)

    @property
    def task_conf_path(self):
        return self._task_conf_path

    @property
    def judge_config_path(self):
        return self._judge_conf_path

    def _get_scorer(self):
        if self._scorer is None:
            self._scorer = FakeScorer()
        return self._scorer

    def load_gold(self):
        examples = {
            1: {
                "instruction": "Answer: {question}",
                "inputs": {"question": "a"},
                "outputs": "",
                "meta": {"id": 1, "reference_answer": "alpha"},
            },
            2: {
                "instruction": "Answer: {question}",
                "inputs": {"question": "b"},
                "outputs": "",
                "meta": {"id": 2, "reference_answer": "beta"},
            },
        }
        self.gold = Dataset(local_path="", name=self.name, log=self.log, examples=examples)
        return []


class OpenJudgeTaskTests(unittest.TestCase):
    def test_openjudge_task_aggregates_and_uses_judge_avg(self):
        tmpdir = Path("tests/.tmp/openjudge_task_case")
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)

        main_conf_path = tmpdir / "main.yaml"
        task_conf_path = tmpdir / "syntheticopenjudgetask.yaml"
        judge_conf_path = tmpdir / "judge.yaml"
        submission_path = tmpdir / "submission.json"

        with open(main_conf_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "args:",
                        f"  working_dir: \"{(tmpdir / 'working').as_posix()}\"",
                        f"  log_dir: \"{(tmpdir / 'logs').as_posix()}\"",
                        "  verbose: false",
                        "  sample_submission_dir_name: sample_submission",
                        "  seed: 1234",
                        "  errors_comments: modules/scoring/configs/errors_comments.yaml",
                        "judges:",
                        "  pollux:",
                        "    backend: hf",
                        "    model_name: ai-forever/pollux-judge-7b",
                    ]
                )
            )
        with open(task_conf_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "extension: .json",
                        "split: test",
                        "use_in_total: true",
                    ]
                )
            )
        with open(judge_conf_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        "criteria:",
                        "  - key: correctness",
                        "    name: Correctness",
                        "    scale:",
                        "      0: bad",
                        "      1: ok",
                        "      2: good",
                        "  - key: style",
                        "    name: Style",
                        "    scale:",
                        "      0: bad",
                        "      1: ok",
                        "      2: good",
                    ]
                )
            )
        with open(submission_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "data": {
                        "test": [
                            {"outputs": "answer-1", "meta": {"id": 1}},
                            {"outputs": "answer-2", "meta": {"id": 2}},
                        ]
                    }
                },
                handle,
            )

        task = SyntheticOpenJudgeTask(
            conf=str(main_conf_path),
            task_conf_path=str(task_conf_path),
            judge_conf_path=str(judge_conf_path),
        )
        task.load_gold()
        results, errors = task.evaluate(str(submission_path))

        self.assertEqual(errors, [])
        self.assertEqual(results["judge_avg"], 1.5)
        self.assertEqual(results["judge_correctness"], 1.5)
        self.assertEqual(results["judge_style"], 1.5)
        self.assertEqual(task.average_results(results), 1.5)


if __name__ == "__main__":
    unittest.main()
