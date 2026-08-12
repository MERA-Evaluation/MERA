"""Pack lm-evaluation-harness logs for the new MERA tasks into a submission.

Same shape as ``log_to_submission.py``: one JSON per task with the answers and
their ids, the raw logs zipped alongside, and the whole directory archived.

What the new tasks let us drop:

* every one of them is ``generate_until``, so there is no loglikelihood branch,
  no per-task ``choices``, and no ``_gen`` twin of a task to disambiguate;
* the ids live in the logs (``doc.meta.id``, a unique int in all twelve), so no
  dataset is downloaded and the packing works offline;
* nothing here replays test answers inside a prompt the way ruTiE does, so the
  prompts are not hashed out of the public logs. See ``truncate_prompts`` if a
  future task needs it.

Usage:
    python scripts/log_to_submission_v2.py --outputs_dir ./output/<run> \
        --dst_dir ./submission_v2
"""

import argparse
import glob
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List

from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from lm_eval.utils import sanitize_model_name


# Task name as it appears in the log file names -> submission file name.
# The names on the right are the `dataset_name` values from
# datasets/*/dataset_meta.json, which is what the leaderboard expects.
TASKS: Dict[str, str] = {
    "characters": "Characters",
    "enantiosemy": "Enantiosemy",
    "gorillahard": "GorillaHard",
    "humor": "Humor",
    "ifhardbench": "IFHardBench",
    "limur": "LIMUR",
    "newreasoning": "NewReasoning",
    "riddles": "Riddles",
    "rubin": "RuBIN",
    "russianregions": "RussianRegions",
    "sage": "SAGE",
    "sobhard": "SOBHard",
}

SAMPLES_PREFIX = "samples_"
RESULTS_PREFIX = "results_"
DATE_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"
LOGS_DIR_NAME = "logs_public"


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def run_stamp(file_name: str) -> str:
    """The timestamp lm-eval put in a log's name.

    One invocation writes ``results_<stamp>.json`` and one
    ``samples_<task>_<stamp>.jsonl`` per task it ran, all with the same stamp —
    which is the only thing tying a results file to its tasks, since its name
    carries no task.
    """
    return os.path.basename(file_name).split(".json")[0].split("_")[-1]


def extract_date(file_name: str) -> datetime:
    return datetime.strptime(run_stamp(file_name), DATE_FORMAT)


def latest_samples_file(outputs_dir: str, task: str) -> str:
    """The most recent ``samples_<task>_<date>.jsonl`` for one task.

    The date is in the file name, so a directory holding several runs keeps the
    newest. The trailing underscore in the pattern is what makes the match
    exact: without it a task name that prefixes another one would pick up both.
    """
    files = glob.glob(os.path.join(outputs_dir, f"{SAMPLES_PREFIX}{task}_*.json*"))
    if not files:
        raise FileNotFoundError(f"no samples file for `{task}` in {outputs_dir}")
    return sorted(files, key=extract_date, reverse=True)[0]


def outputs_to_submission(rows: List[dict]) -> dict:
    """Answers and their ids, in the submission's shape."""
    data = []
    for row in rows:
        answer = row["filtered_resps"]
        # generate_until always yields a one-element list; keep the string.
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        data.append({
            "outputs": answer,
            "meta": {"id": int(row["doc"]["meta"]["id"])},
        })
    return {"data": {"test": data}}


def convert_task(task: str, dst_name: str, outputs_dir: str, dst_dir: str) -> int:
    """Write one task's submission file. Returns how many answers it holds."""
    path = latest_samples_file(outputs_dir, task)
    rows = load_jsonl(path)
    submission = outputs_to_submission(rows)
    save_json(submission, os.path.join(dst_dir, f"{dst_name}.json"))

    answers = submission["data"]["test"]
    ids = [item["meta"]["id"] for item in answers]
    empty = sum(1 for item in answers if not str(item["outputs"]).strip())
    print(f"  {task:<16} -> {dst_name}.json  {len(answers)} answers"
          f"{f', {empty} EMPTY' if empty else ''}"
          f"{f', {len(ids) - len(set(ids))} DUPLICATE ids' if len(ids) != len(set(ids)) else ''}"
          f"  [{os.path.basename(path)}]")
    return len(answers)


def pack_logs(packed_files: List[str], outputs_dir: str, dst_dir: str) -> str:
    """Zip the sample logs that were packed, plus their own results files.

    A directory may hold several runs — re-running six of the twelve tasks
    leaves the other six pointing at an older run. The sample logs sort that out
    by task and date, but a results file names no task, so it is matched by run
    stamp instead: a results file travels only if some packed sample came from
    the same invocation. Results left over from superseded runs are dropped,
    since they describe answers that are not in this submission.
    """
    zip_dir = os.path.join(dst_dir, LOGS_DIR_NAME)
    os.makedirs(zip_dir, exist_ok=True)

    for path in packed_files:
        shutil.copy2(path, zip_dir)

    stamps = {run_stamp(path) for path in packed_files}
    results = glob.glob(os.path.join(outputs_dir, f"{RESULTS_PREFIX}*.json*"))
    kept = [path for path in results if run_stamp(path) in stamps]
    for path in kept:
        shutil.copy2(path, zip_dir)
    if len(kept) < len(results):
        print(f"  dropped {len(results) - len(kept)} results file(s) from "
              f"superseded runs")

    zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)
    shutil.rmtree(zip_dir)
    print(f"Logs to add with public submission stored at {zip_path} "
          f"({len(packed_files)} samples + {len(kept)} results from "
          f"{len(stamps)} run(s))")
    return zip_path


def create_submission(outputs_dir: str, dst_dir: str) -> str:
    if not os.path.isdir(outputs_dir):
        raise ValueError(f"{outputs_dir} is not a directory")
    os.makedirs(dst_dir, exist_ok=True)

    packed_files, total, missing = [], 0, []
    print(f"Packing {len(TASKS)} tasks from {outputs_dir}")
    for task, dst_name in TASKS.items():
        try:
            packed_files.append(latest_samples_file(outputs_dir, task))
            total += convert_task(task, dst_name, outputs_dir, dst_dir)
        except FileNotFoundError as exc:
            print(f"  {task:<16} -- skipped: {exc}")
            missing.append(task)

    print(f"\n{len(TASKS) - len(missing)}/{len(TASKS)} tasks packed, {total} answers")
    if len(missing) == len(TASKS):
        # Almost always a path pointed one level above the model's own
        # subdirectory; building an empty archive would only hide that.
        raise FileNotFoundError(
            f"no task logs found in {outputs_dir}. lm-eval writes into a "
            f"per-model subdirectory — point --outputs_dir at it, or pass the "
            f"run's --model_args so it can be resolved. "
            f"Subdirectories here: {sorted(os.listdir(outputs_dir))}")
    if missing:
        print(f"MISSING: {', '.join(missing)} — the submission is incomplete")

    pack_logs(packed_files, outputs_dir, dst_dir)
    zip_path = shutil.make_archive(dst_dir, "zip", dst_dir)
    print("Submission stored at", zip_path)
    return zip_path


def preprocess_outputs_dir(outputs_dir: str, model_args: str) -> str:
    """Resolve the per-model subdirectory lm-eval writes into.

    The caller either points at that directory directly, or points one level up
    and passes the same ``model_args`` the run used.
    """
    if model_args:
        model_name = GeneralConfigTracker._get_model_name(model_args)
        return os.path.join(outputs_dir, sanitize_model_name(model_name))
    return outputs_dir


def truncate_prompts(rows: List[dict]) -> List[dict]:
    """Replace each request text with its sha256.

    Unused: none of the twelve tasks puts test answers into a prompt. Kept
    because ruTiE in ``log_to_submission.py`` does, and a future dialogue task
    scored on more than its last turn would need the same treatment before its
    logs are published.
    """
    import hashlib

    for row in rows:
        for key, argument in row["arguments"].items():
            text = argument["arg_0"]
            if not isinstance(text, str):
                text = text[0]
            row["arguments"][key]["arg_0"] = hashlib.sha256(
                text.encode()).hexdigest()
    return rows


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, required=True,
                        help="lm-evaluation-harness outputs")
    parser.add_argument("--dst_dir", type=str, default="submission_v2/",
                        help="dir to save files for submission")
    parser.add_argument("--model_args", type=str, default="",
                        help="the run's model_args, used to find the model's "
                             "subdirectory under --outputs_dir")
    return parser.parse_known_args()[0]


def main():
    args = get_args()
    create_submission(
        preprocess_outputs_dir(args.outputs_dir, args.model_args), args.dst_dir)


if __name__ == "__main__":
    main()
