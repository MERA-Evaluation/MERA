# MERA

<p align="center">
  <picture>
    <img alt="MERA" src="docs/mera-logo.svg" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/MERA-Evaluation/MERA/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.2.0-blue">
    </a>

</p>

<h2 align="center">
    <p> MERA (Multimodal Evaluation for Russian-language Architectures) is a new open benchmark for the Russian language for evaluating SOTA models.
</p>
</h2>


## **🔔 CALL FOR NEW DATASETS: Help Expand Our Benchmark!**

We’re inviting contributions to grow our benchmark with diverse datasets for the new chapter of the MERA TEXT.

### 🎯 Submission Guidelines

The tests needs to be HARD for the latest models! MERA TEXT targets primarily the Russian language tests.

***💡 Suggested Tasks***
- Dialogue / Conversational Skills
- Reasoning Tasks
- Instruction Following / Alignment
- Creativity & Generation
- Reflection
- Safety & Ethics
- Evolving / Adaptation
- Empathy / Theory of Mind / Emotional Intelligence
- Advanced & Meta-Cognitive Abilities
etc.

**🚀 How to Contribute?**

0. Submit a **Pull Request** with the dataset description to this repository ([instruction](docs/how_to_add_dataset.md)).
1. Develop your dataset according to the text LLM evaluation criteria ([see requirements](docs/dataset_review.md)).
2. Format your dataset to our specifications ([format instruction](docs/dataset_formatting.md)) and upload it to the 🤗 Hugging Face Hub ([instruction](docs/dataset_hf.md](docs/dataset_formatting.md))). 
3.  Integrate your dataset into our codebase using the instructions above. Check that it works by running the baselines! ([instruction](docs/task_codebase.md)).

We will review your submission and, upon approval, add it to New MERA TEXT.

**❓ Questions?**
Open an [Issue](https://github.com/MERA-Evaluation/MERA/issues) or discuss in our [Community Forum](https://t.me/+XkBIbHFg8s5iNGIy).  
*Let’s build a more representative benchmark together!* ✨  

Feel free to email any questions & feedback regarding our work at mera@a-ai.ru. If you find any bugs or ideas for code improvement, please suggest the fixes via pull-requests and issues in this official MERA GitHub repo. We will be glad to get your feedback!


## About MERA

MERA benchmark brings together all industry and academic players in one place to study the capabilities of SOTA models, draw attention to AI problems, develop collaboration within the Russian Federation and in the international arena and create an independent unified system for measuring all current models. This repository is a customized version of original [**Language Model Evaluation Harness**](https://github.com/EleutherAI/lm-evaluation-harness) (**LM-Harness** `v0.4.8`).

Our contributions to this project are:

- Instruction-based tasks available on 🤗 HuggingFace [dataset card](https://huggingface.co/datasets/MERA-evaluation/MERA).
- Customized version of LM-Harness evaluation code for models (`v0.4.8`).
- Benchmark website with the [Leaderboard](https://mera.a-ai.ru/en/leaderboard) and the scoring submission system.
- Baselines of the open models and Human Benchmark.


The MERA benchmark includes 23 text tasks (15 base tasks + 8 diagnostic tasks). See the task-table for a complete list.

| Name | Task Name | Task Type | Test Size | N-shots | Metrics |
| --- | --- | --- | --- | --- | --- |
| MathLogicQA | mathlogicqa | Math, Logic | 1143 | 1 | Acc |
| MultiQ | multiq | Reasoning | 900 | 0 | EM / F1 |
| PARus | parus | Common Sense | 500 | 1 | Acc |
| RCB | rcb | NLI | 438 | 1 | Acc / F1_macro |
| ruModAr | rumodar | Math, Logic | 6000 | 0 | EM |
| ruMultiAr | rumultiar | Math | 1024 | 1 | EM |
| ruOpenBookQA | ruopenbookqa | World Knowledge | 400 | 1 | Acc / F1_macro |
| ruTiE | rutie | Reasoning, Dialogue Context, Memory | 430 | 1* | Acc |
| ruWorldTree | ruworldtree | World Knowledge | 525 | 1 | Acc / F1_macro |
| RWSD | rwsd | Reasoning | 260 | 1 | Acc |
| SimpleAr | simplear | Math | 1000 | 2 | EM |
| BPS | bps | Code, Math | 1000 | 1 | Acc |
| CheGeKa | chegeka | World Knowledge | 416 | 1 | EM / F1 |
| LCS | lcs | Code, Math | 500 | 1 | Acc |
| ruHumanEval | ruhumaneval | Code | 164 | 0 | Pass@k |
| ruCodeEval | rucodeeval | Code | 164 | 0 | Pass@k |
| ruMMLU | rummlu | Reasoning | 14012 | 1 | Acc |
| MaMuRAMu | mamuramu | Reasoning | 4248 | 1 | Acc |
| USE | use | Exam | 900 | 1 | Grade_norm |
| ruDetox | rudetox | Ethics | 800 | 1 | J(STA, SIM, FL) |
| ruEthics | ruethics | Ethics | 1935 | 0 | 5 MCC |
| ruHateSpeech | ruhatespeech | Ethics | 265 | 1 | Acc |
| ruHHH | ruhhh | Ethics | 178 | 0 | Acc |

*"artificial" few-shot that is meant to make the task work correct

Our aim is to evaluate all the models:

- in the same scenarios;
- using the same metrics;
- with the same adaptation strategy (e.g., prompting);
- provide an opportunity to make controlled and clear comparisons.

MERA is a collaborative project created in a union of industry and academia with the **support of all the companies**, that are creating the foundation models, to ensure fair and transparent leaderboards for the models evaluation.

*We express our gratitude to our team and partners:*

*SberDevices, Sber AI, Yandex, Skoltech AI, MTS AI, NRU HSE, Russian Academy of Sciences, etc.*

*Powered by [Aliance AI](https://a-ai.ru)*

## Contents

The repository has the following structure:

- [`benchmark_tasks`](benchmark_tasks) — the tasks for evaluation of language models.
- [`humanbenchmarks`](humanbenchmarks/README.md) — materials and code for human evaluation.
- [`modules`](modules/scoring/README.md) — the examples of scoring scripts that are used on the website for scoring your submission.
- [`lm-evaluation-harness`](https://github.com/artemorloff/lm-evaluation-harness/tree/update_release) — a framework for few-shot evaluation of language models.
- [`scripts`](scripts) — the scripts used for evaluation of language models.


## The process of submission is the following:
- to view the datasets use the [HuggingFace preview](https://huggingface.co/datasets/MERA-evaluation/MERA/viewer/ruethics);
- clone MERA benchmark [repository](https://github.com/MERA-Evaluation/MERA) with submodules using the following code:

```bash
git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git
```

If you have cloned the repository with no submodlues downloaded (empty directory), run this code to fix it from the root directory:

```bash
git pull --all --rebase --recurse-submodules
```

- to get submission files use [shell script](MODEL_SCORING.md) and the provided customized **lm-harness** code (the actual model is not required for submission and evaluation), [see documentation for evaluation parameters](MODEL_SCORING_PARAMETERS.md).
- run your model on the all datasets using the code of lm-eval: the result of the code is the archive in ZIP format for the submission;
- register on the website;
- upload the submission file (ZIP) via the platform interface for the automatic assessment.

*Note that, the evaluation result is then displayed in the user's account and is kept **private**. Those who want to make their submission results public could use the *''Publish''* function. After validation of the submission is approved, the model's overall score will be shown publicly.*
*The parameters of the generation, prompts and few-shot/zero-shot are fixed. You can vary them for your own purposes. If you want to submit your results on the public leaderboard check that these parameters are the same and please add the logs (packed in submission file by default). We have to be sure that the scenarios for the models evaluation are the same and reproducible.*

We provide the [sample submission](modules/scoring/examples) for you to check the format.

The process of the whole MERA evaluation is described on the Figure:

![evaluation setup](docs/mera.png)

## MERA Open Reasoning Leaderboard

[MERA Reason](https://huggingface.co/spaces/MERA-evaluation/MERA_Reason) is an open Hugging Face leaderboard for evaluating **reasoning** capabilities of language models in Russian. Unlike the main MERA website submission flow, results are uploaded directly to the Space and appear on the leaderboard immediately after validation.

The reasoning track includes four benchmarks:

| Benchmark | Task code | Type | Test size | N-shots | Primary metric |
| --- | --- | --- | --- | --- | --- |
| ruAIME | `ruaime` | Mathematics (AIME) | 724 | 2 | Exact match |
| T-math | `tmath` | Olympiad mathematics | 331 | 0 | Exact match |
| Luzitania | `luzitania` | Logic, multi-step reasoning | 251 | 0 | Exact match |
| MMReD | `mmred` | Long-context dense reasoning | 750 | 0 | `em.dc_aggregate` |

Dataset cards: [ruAIME](https://huggingface.co/datasets/MERA-evaluation/ruAIME), [T-math](https://huggingface.co/datasets/MERA-evaluation/T-math), [Luzitania](https://huggingface.co/datasets/MERA-evaluation/Luzitania), [MMReD](https://huggingface.co/datasets/MERA-evaluation/MMReD).

### Install

Follow the [installation instructions](MODEL_SCORING.md#install) for `lm-evaluation-harness` from this repository (clone with submodules, then `pip install -e .` inside `lm-evaluation-harness/`).

### Run the reasoning benchmark

Use the helper script [`scripts/run_reasoning_benchmark.sh`](scripts/run_reasoning_benchmark.sh). It runs all four tasks sequentially, logs samples, computes metrics locally, and packs a submission ZIP.

**Hugging Face model (local inference):**

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_FOLDER="$PWD/mera_results/my-model-reasoning/" \
MERA_MODEL_STRING="pretrained=org/my-model,dtype=auto" \
MERA_COMMON_SETUP="--model hf --device cuda --batch_size=1 --log_samples --seed 1234 --verbosity ERROR --apply_chat_template --fewshot_as_multiturn" \
bash scripts/run_reasoning_benchmark.sh
```

**Model served via OpenAI-compatible API (e.g. vLLM):**

```bash
OUTPUT_FOLDER="$PWD/mera_results/my-model-reasoning/" \
MERA_MODEL_STRING="model=org/my-model,num_concurrent=16,timeout=3000,max_retries=5,base_url=http://0.0.0.0:8088/v1/chat/completions" \
bash scripts/run_reasoning_benchmark.sh
```

Useful environment variables:

- `OUTPUT_FOLDER` — directory for lm-eval logs and the packed submission (default: `./mera_results/`).
- `MERA_MODEL_STRING` — same as `--model_args` for `lm_eval` (required).
- `MERA_COMMON_SETUP` — extra `lm_eval` flags (default: `local-completions` with chat template and multi-turn few-shot).
- `TASKS` — space-separated subset of tasks (default: `mmred tmath luzitania ruaime`).
- `LIMIT` — optional sample limit for debugging (e.g. `LIMIT=10`).
- `PACK_SUBMISSION=0` — skip automatic ZIP packing after evaluation.

For long reasoning generations, override `GENERATION_KWARGS` if needed (task YAMLs define per-benchmark limits, e.g. Luzitania up to 65K tokens). See [`MODEL_SCORING.md`](MODEL_SCORING.md) and task configs in [`benchmark_tasks/`](benchmark_tasks/) for details.

### Pack submission manually

If you ran tasks separately or need to re-pack existing logs:

```bash
python scripts/log_to_reasoning_submission.py \
    --outputs_dir "$PWD/mera_results/my-model-reasoning/" \
    --model_args "pretrained=org/my-model,dtype=auto" \
    --tasks "mmred,tmath,luzitania,ruaime"
```

Or use [`scripts/pack_reasoning_submission.sh`](scripts/pack_reasoning_submission.sh) as a template.

The resulting ZIP archive must contain:

- `ruaime.json`, `t_math.json`, `luzitania.json`, `mmred.json` — prediction files in MERA submission format (`data.test[].outputs` + `meta.id`).
- `logs_public.zip` — archive with `results_*.json` (and optionally `samples_*.json*`) from lm-eval.

**Important:** leaderboard scores are extracted **only** from `results_*.json` inside `logs_public`. Sample logs alone are not sufficient — run evaluation without `--predict_only` so that metrics are computed locally, or ensure `results_*.json` files are present before packing.

### Submit to the leaderboard

1. Open [MERA Reason](https://huggingface.co/spaces/MERA-evaluation/MERA_Reason) and go to the **Submit** tab.
2. Enter the **model name** (e.g. `org/model`) and **team name**.
3. Upload the ZIP archive produced by `run_reasoning_benchmark.sh` or `log_to_reasoning_submission.py`.
4. After validation, scores appear on the **Leaderboard** and **Datasets** tabs.

For reproducible public results, keep generation parameters, prompts, and few-shot settings aligned with the task definitions in `benchmark_tasks/`. Include full `logs_public` in the submission archive.

------------------------------------

📌 It’s the first text version of the benchmark. We are to expand and develop it in the future with new tasks and multimodality.

Feel free to ask any questions regarding our work, write on email mera@a-ai.ru. If you have ideas and new tasks feel free to suggest them, **it’s important!** If you see any bugs, or you know how to make the code better please suggest the fixes via pull-requests and issues in this official github 🤗. We will be glad to get the feedback in any way.


## Cite as

```
@inproceedings{fenogenova-etal-2024-mera,
    title = "{MERA}: A Comprehensive {LLM} Evaluation in {R}ussian",
    author = "Fenogenova, Alena  and
      Chervyakov, Artem  and
      Martynov, Nikita  and
      Kozlova, Anastasia  and
      Tikhonova, Maria  and
      Akhmetgareeva, Albina  and
      Emelyanov, Anton  and
      Shevelev, Denis  and
      Lebedev, Pavel  and
      Sinev, Leonid  and
      Isaeva, Ulyana  and
      Kolomeytseva, Katerina  and
      Moskovskiy, Daniil  and
      Goncharova, Elizaveta  and
      Savushkin, Nikita  and
      Mikhailova, Polina  and
      Minaeva, Anastasia  and
      Dimitrov, Denis  and
      Panchenko, Alexander  and
      Markov, Sergey",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.534",
    doi = "10.18653/v1/2024.acl-long.534",
    pages = "9920--9948",
}
```
