#!/usr/bin/env python3
"""Pack lm-eval logs for MERA Reason leaderboard submission.

Expected archive layout:
  mmred.json, t_math.json, luzitania.json, ruaime.json
  logs_public.zip  (results_*.json and samples_*.json* for the four benchmarks)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime
from typing import Iterable

INPUT_DATE_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"
SAMPLES_PREFIX = "samples_"
RESULTS_PREFIX = "results_"

REASONING_TASKS = {
    "mmred": {
        "submission_name": "mmred",
        "sample_globs": ["samples_mmred*.json*"],
    },
    "tmath": {
        "submission_name": "t_math",
        "sample_globs": ["samples_tmath*.json*"],
    },
    "luzitania": {
        "submission_name": "luzitania",
        "sample_globs": ["samples_luzitania*.json*"],
    },
    "ruaime": {
        "submission_name": "ruaime",
        "sample_globs": ["samples_ruaime*.json*"],
    },
}

LOGS_PUBLIC_GLOBS = [
    "results_mmred*.json*",
    "results_tmath*.json*",
    "results_t_math*.json*",
    "results_luzitania*.json*",
    "results_ruaime*.json*",
    "samples_mmred*.json*",
    "samples_tmath*.json*",
    "samples_t_math*.json*",
    "samples_luzitania*.json*",
    "samples_ruaime*.json*",
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def extract_date(file_name: str) -> datetime:
    extract_str_date = file_name.split(".json")[0].split("_")[-1]
    return datetime.strptime(extract_str_date, INPUT_DATE_FORMAT)


def latest_files(paths: Iterable[str]) -> list[str]:
    unique = sorted(set(paths))
    if not unique:
        return []
    by_task: dict[str, list[str]] = {}
    for path in unique:
        base = os.path.basename(path)
        task_key = re.sub(r"\.json.*$", "", base)
        task_key = re.sub(r"^samples_", "", task_key)
        task_key = re.sub(r"^results_", "", task_key)
        by_task.setdefault(task_key, []).append(path)
    return [sorted(group, key=lambda p: extract_date(os.path.basename(p)), reverse=True)[0] for group in by_task.values()]


def find_sample_files(outputs_dir: str, sample_globs: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in sample_globs:
        files.extend(glob.glob(os.path.join(outputs_dir, pattern)))
    return latest_files(files)


def doc_id(doc: dict) -> int:
    meta = doc["doc"]["meta"]
    return int(meta["id"])


def samples_to_submission(sample_paths: list[str]) -> dict:
    rows: dict[int, dict] = {}
    for path in sample_paths:
        for entry in load_jsonl(path):
            doc = entry["doc"]
            prediction = entry["filtered_resps"][0]
            rows[doc_id(entry)] = {
                "outputs": prediction,
                "meta": {"id": int(doc["meta"]["id"])},
            }
    return {"data": {"test": [rows[key] for key in sorted(rows)]}}


def get_model_name(model_args: str) -> str:
    prefixes = ["peft=", "delta=", "pretrained=", "model=", "path=", "engine="]
    for prefix in prefixes:
        if prefix in model_args:
            return model_args.split(prefix, 1)[1].split(",", 1)[0]
    return ""


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[\"<>:/|\\?*\[\]]+", "__", model_name)


def preprocess_outputs_dir(outputs_dir: str, model_args: str) -> str:
    if not model_args:
        return outputs_dir
    model_name = get_model_name(model_args)
    if not model_name:
        return outputs_dir
    return os.path.join(outputs_dir, sanitize_model_name(model_name))


def pack_logs_public(outputs_dir: str, dst_dir: str) -> str:
    zip_dir = os.path.join(dst_dir, "logs_public")
    os.makedirs(zip_dir, exist_ok=True)
    copied = 0
    for pattern in LOGS_PUBLIC_GLOBS:
        for file_path in glob.glob(os.path.join(outputs_dir, pattern)):
            shutil.copy2(file_path, zip_dir)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(
            f"No reasoning benchmark logs found in {outputs_dir}. "
            "Expected results_*/samples_* files for mmred, tmath, luzitania, ruaime."
        )
    zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)
    shutil.rmtree(zip_dir)
    return zip_path


def create_submission(outputs_dir: str, dst_dir: str, tasks: list[str]) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for task in tasks:
        spec = REASONING_TASKS[task]
        sample_paths = find_sample_files(outputs_dir, spec["sample_globs"])
        if not sample_paths:
            missing.append(task)
            print(f"[skip] no samples for {task}")
            continue
        submission = samples_to_submission(sample_paths)
        out_path = os.path.join(dst_dir, f"{spec['submission_name']}.json")
        save_json(submission, out_path)
        print(f"[ok] {task}: {len(submission['data']['test'])} predictions -> {out_path}")

    if missing:
        raise FileNotFoundError(
            "Missing sample logs for: " + ", ".join(missing)
        )

    print("Packing logs for public submission...")
    logs_zip = pack_logs_public(outputs_dir, dst_dir)
    print("logs_public stored at", logs_zip)

    zip_path = shutil.make_archive(dst_dir, "zip", dst_dir)
    print("Submission stored at", zip_path)
    return zip_path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", required=True, help="lm-eval output directory")
    parser.add_argument(
        "--dst_dir",
        default="submission/",
        help="Directory for submission JSON files and zip archive",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="Same model_args string passed to lm_eval (to locate model subfolder)",
    )
    parser.add_argument(
        "--tasks",
        default="mmred,tmath,luzitania,ruaime",
        help="Comma-separated subset of reasoning tasks to pack",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = set(tasks) - set(REASONING_TASKS)
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown))}")

    outputs_dir = preprocess_outputs_dir(args.outputs_dir, args.model_args)
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    create_submission(outputs_dir, args.dst_dir, tasks)


if __name__ == "__main__":
    main()
