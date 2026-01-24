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

## 🚀 About

**MERA** benchmark brings together all industry and academic players in one place to study the capabilities of SOTA models. Built on top of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.8), it enables researchers and practitioners to:

- **Compare models** on identical tasks and metrics
- **Reproduce results** with fixed prompts and few-shot settings
- **Submit** standardized ZIP archives for leaderboard integration

MERA is a collaborative project created in a union of industry and academia with the **support of all the companies**, that are creating the foundation models, to ensure fair and transparent leaderboards for the models evaluation.




## 🔍 Datasets Overview

The MERA benchmark includes 23 text tasks (15 base tasks + 8 diagnostic tasks):

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


## 🛠 Getting Started <a name="evaluation"></a>

First, you need to clone the MERA repository and load the submodule:

```bash
### Go to the folder where the repository will be cloned ###
mkdir mera
cd mera

### Clone & install core libs ###
git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA.git
cd MERA
```

If you have cloned the repository with no submodules downloaded (empty directory), run this code to fix it from the root directory:

```bash
git pull --all --rebase --recurse-submodules
```

Now, you can choose one of two evaluation regimes, depending on whether you want to obtain the metrics for public tasks locally or intend to use our remote scoring via the website.

### Remote Scoring

**Remote Scoring** (default): quick setup for cloud-based scoring — install only core dependencies, run the evaluation, and submit the resulting ZIP archive to our website to get the score.

> You will not get the metrics even for public datasets (for each dataset, you will see a "bypass" placeholder instead of actual metrics) in the terminal.

```bash
### Install lm-eval ###
cd lm-evaluation-harness
pip install -e .

### Install additional libs for models evaluation [Optional] ###
# vLLM engine
pip install -e ".[vllm]"
# API scoring
pip install -e ".[api]"

### Go to MERA folder ###
cd ../
```

```bash
### Run evaluation and pack logs ###
bash scripts/run_evaluation.sh \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,tensor_parallel_size=1" \
    --output_path "./results/Qwen2.5-0.5B-Instruct"
```

## 📁 Repository Structure

```text
MERA/
├── benchmark_tasks/                # Code for each task
├── docs/                           # Additional documentation and design notes
│   ├── dataset_cards/              # Task descriptions (EN/RU)
│   ├── dataset_formatting.md       # Dataset formatting requirements
│   ├── dataset_hf.md               # How to add new datasets on the MERA HuggingFace page
│   ├── dataset_review.md           # General dataset requirements
│   ├── human_baseline.md           # Human baseline documentation
│   └── task_codebase.md            # How to add a new task to the codebase
├── humanbenchmarks/                # Materials and code for human evaluation
├── modules/                        # Scoring scripts examples
├── lm-evaluation-harness/          # Submodule (codebase)
└── scripts/                        # Helpers: add tasks, run evaluations, and scoring
```


## 💪 How to Join the Leaderboard

Follow these steps to see your model on the Leaderboard:

1. **Run Remote Scoring**
   Evaluate the benchmark in the **Remote Scoring** regime (see [🛠 Getting Started](#evaluation) above).
   > You'll end up with a logs folder **and** a ready-to-submit zip archive like `Qwen2.5-0.5B-Instruct_submission.zip`.

2. **Submit on the website**
   Head over to [Create Submission](https://mera.a-ai.ru/ru/text/submits/create), upload the archive, and move on to the form.

3. **Fill in Model Details**
   Provide accurate information about the model and evaluation. These details are crucial for reproducibility—if something is missing, administrators may ping you (or your Submission might be rejected).

4. **Wait for Scoring** ⏳
   Scoring usually wraps up in **~10-15 minutes**.

5. **Publish your result**
   Once scoring finishes, click **"Submit for moderation"**. After approval, your model goes **Public** and appears on the [Leaderboard](https://mera.a-ai.ru/ru/text/leaderboard).

Good luck, and happy benchmarking! 🎉


## 🤝 Contributing

We are interested in improving the MERA and invite the community to contribute to the development of new domain-specific tasks and the project's codebase.

### Steps to Add a New Task:
0) Develop a dataset (on the contributor's side)
1) Convert the dataset to MERA format ([guide](docs/dataset_formatting.md))
2) Write evaluation code using lm-harness ([guide](docs/task_codebase.md))
3) Benchmark state-of-the-art baseline models on the dataset
4) Final moderation, and your dataset is officially added!

Feel free to email any questions & feedback regarding our work at mera@a-ai.ru.


## 📝 License

Distributed under the MIT License. See LICENSE for details.
