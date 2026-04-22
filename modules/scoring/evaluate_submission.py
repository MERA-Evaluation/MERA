import sys
from pathlib import Path

_SCORING_ROOT = Path(__file__).resolve().parent
_scoring_root_str = str(_SCORING_ROOT)
if _scoring_root_str not in sys.path:
    sys.path.insert(0, _scoring_root_str)

from src.worker import Worker
from src.utils import load_yaml, save_json
import json
import argparse


def evaluate_submissions(args):
    config = build_config(args)
    worker = Worker(conf=config, no_load_models=False)
    errors = worker.load()
    if len(errors):
        worker.log(errors)
        worker.log("Evaluate with errors...")
    res = worker.evaluate(local_path=args.submission_path, remove_local_file=False)
    save_json(res, args.results_path)
    worker.log(f"Submission stored at: {args.results_path}")
    worker.log(f"Evaluation result: {json.dumps(res, ensure_ascii=False, indent=4)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/main.yaml",
        help="path to auth config",
    )
    parser.add_argument(
        "--submission_path",
        type=str,
        default="submission.zip",
        help="path to submission",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="submission_results.json",
        help="path to submission results",
    )
    parser.add_argument("--judge_backend", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--judge_base_url", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_batch_size", type=int, default=None)
    parser.add_argument("--judge_max_new_tokens", type=int, default=None)
    res = parser.parse_known_args()[0]
    return res


def build_config(args):
    config = load_yaml(args.config_path)
    if "judges" not in config:
        config["judges"] = {}
    if "pollux" not in config["judges"]:
        config["judges"]["pollux"] = {}
    pollux_conf = config["judges"]["pollux"]
    if args.judge_backend is not None:
        pollux_conf["backend"] = args.judge_backend
    if args.judge_model is not None:
        pollux_conf["model_name"] = args.judge_model
    if args.judge_base_url is not None:
        pollux_conf["base_url"] = args.judge_base_url
    if args.judge_api_key is not None:
        pollux_conf["api_key"] = args.judge_api_key
    if args.judge_batch_size is not None:
        pollux_conf["batch_size"] = args.judge_batch_size
    if args.judge_max_new_tokens is not None:
        pollux_conf["max_new_tokens"] = args.judge_max_new_tokens
    return config


def main():
    args = get_args()
    evaluate_submissions(args)


if __name__ == "__main__":
    main()
