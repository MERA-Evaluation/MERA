from src.utils import save_json, get_files_from_dir, load_json
from src.worker import Worker
import os
import argparse


def remove_outputs(dst_dir="../../lm-evaluation-harness/lm_eval/datasets/", conf_path="configs/main.yaml"):
    paths = [x for x in get_files_from_dir(dst_dir) if x.endswith("task.json")]
    no_tasks = []
    worker = Worker(conf_path, no_load_models=True)
    worker.load()
    for task_name, task_obj in worker.tasks.items():
        dst = None
        for task_path in paths:
            if task_name.lower() in task_path:
                dst = task_path
                print("Process", task_name, "into", dst)
                break
            dst_task_name = os.path.split(os.path.split(task_path)[0])[-1].lower()
            k = len(set(task_name.lower()).intersection(set(dst_task_name))) / max(len(dst_task_name), len(task_name))
            if 0.65 < k:
                dst = task_path
                print("Process", task_name, "resolved from", dst, dst_task_name, k)
                break
        if dst is None:
            # try resolve
            print("Can't find", task_name)
            no_tasks.append(task_name)
        else:
            if task_obj.task_conf.use_in_total:
                task = task_obj.remove_outputs()
            else:
                task = load_json(task_obj.task_conf.origin_repo_path)
            if task is None:
                print("No task at origin", task_name)
            else:
                save_json(task, dst)
        print("---------------------")
    print("Not refactored tasks", no_tasks)
    return no_tasks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dst_dir", type=str,
        default="../../lm-evaluation-harness/lm_eval/datasets/",
        help="path to stored datasets dir"
    )
    res = parser.parse_known_args()[0]
    return res


def main():
    args = get_args()
    remove_outputs(dst_dir=args.dst_dir)


if __name__ == "__main__":
    main()
