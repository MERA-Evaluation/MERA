# MERA with Language Model Evaluation Harness

MERA: Multimodal Evaluation for Russian-language Architectures

The Language Model Evaluation Harness support for the MERA benchmark datasets.

## Overview

This project provides a unified framework to test generative language models on MERA benchmark and its evaluation tasks.

## Install

To install `lm-eval` from the repository main branch, run:

```bash
cd lm-evaluation-harness
pip install -e .
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
cd lm-evaluation-harness
pip install -e ".[auto-gptq]"
```

To use vLLM, do 
```bash
cd lm-evaluation-harness
pip install lm_eval[vllm]
```

These commands are to be run from `lm-evaluation-harness` directory of this repository.

## MERA Benchmark:

### Run full benchmark with bash script

#### Running HF models

Sample command to run benchmark with `ai-forever/rugpt3large_based_on_gpt2` (`AutoModelForCausalLM` and `AutoModelForSeq2SeqLM` classes compatible)
model from Huggingface Hub:

```shell
CUDA_VISIBLE_DEVICES=0 \
MERA_FOLDER="$PWD/mera_results/rugpt3large_760m_defaults" \
MERA_MODEL_STRING="pretrained=ai-forever/rugpt3large_based_on_gpt2,dtype=auto" \
bash scripts/run_benchmark.sh
```

Running HuggingFace models with Multi-GPU:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
MERA_FOLDER="$PWD/mera_results/Qwen1.5-32B-Chat_defaults" \
MERA_MODEL_STRING="pretrained=Qwen/Qwen1.5-32B-Chat,dtype=auto,parallelize=True" \
bash scripts/run_benchmark.sh
```

Running HuggingFace models with vLLM:
```shell
CUDA_VISIBLE_DEVICES=1 \
MERA_FOLDER="$PWD/mera_results/Llama-3.1-8B-Instruct" \
MERA_MODEL_STRING="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.45" \
MERA_COMMON_SETUP="--model vllm --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR" \
bash scripts/run_benchmark.sh
```

Running HuggingFace models with vLLM with Multi-GPU:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
MERA_FOLDER="$PWD/mera_results/Llama-3.1-8B-Instruct" \
MERA_MODEL_STRING="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.45" \
MERA_COMMON_SETUP="--model vllm --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR" \
bash scripts/run_benchmark.sh
```

Running OpenAI models with support of logits (e.g. davinci-002) - put your API key instead of TOKEN:
```shell
OPENAI_API_KEY="TOKEN" \
MERA_FOLDER="$PWD/mera_results/davinci-002" \
MERA_MODEL_STRING="model=davinci-002" \
MERA_COMMON_SETUP="--model openai-completions --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity INFO --apply_chat_template" \
bash scripts/run_benchmark_gen.sh
```

Running OpenAI Chat models (e.g. gpt-4o):
```shell
OPENAI_API_KEY="TOKEN" \
MERA_FOLDER="$PWD/mera_results/gpt-4o-2024-05-13" \
MERA_MODEL_STRING="model=gpt-4o-2024-05-13" \
MERA_COMMON_SETUP="--model openai-chat-completions --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity INFO --apply_chat_template" \
bash scripts/run_benchmark_gen.sh
```

Running models by requesting the server that imitates OpenAI API (e.g. vllm.entrypoints.openai.api_server):
```shell
OPENAI_API_KEY="" \
MERA_FOLDER="$PWD/mera_results/Meta-Llama-3.1-405B-Instruct" \
MERA_MODEL_STRING="model=meta-llama/Meta-Llama-3.1-405B-Instruct,num_concurrent=128,base_url=http://localhost:8000/v1/completions,tokenizer_backend=huggingface,tokenizer=meta-llama/Meta-Llama-3.1-405B-Instruct" \
MERA_COMMON_SETUP="--model openai-completions --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity INFO" \
bash scripts/run_benchmark.sh
```

Parameters of shell scripts:
- `CUDA_VISIBLE_DEVICES` to set cuda device visibility, 
- `MERA_FOLDER` for path to store outputs,
- `MERA_MODEL_STRING` to setup `model_args` parameter of `lm-evaluation-harness`'s `lm_eval` module,
- `MERA_COMMON_SETUP` to change default parameters for model inferencing with `lm_eval` (defaults are `--model hf --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR`).

If you want to select only generative versions of tasks (all originally generative tasks and generative versions
of loglikelihood tasks), use `scripts/run_benchmark_gen.sh` script. To run all existing tasks
execute `scripts/run_benchmark_all.sh`. This way two separate submissions will be created: one for regular
MERA tasks (loglikelihood and generative tasks), one for generative MERA tasks only.

#### System prompt usage

If you want to pass some system prompt to use it for all tasks of MERA benchmark, you are supposed to pass this prompt as environmental variable `SYSTEM_PROMPT` along with other variables for any shell script. 
Pay attention that `SYSTEM_PROMPT` is passed in singular quotes to preserve all special symbols.

Example of running HuggingFace model on vLLM engine with special tokens, chat template and multi-turn enabled and system prompt stated with generative setup:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
SYSTEM_PROMPT='You are a helpful assistant' \
MERA_FOLDER="$PWD/mera_results/Llama-3.1-8B-Instruct" \
MERA_MODEL_STRING="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.45,add_bos_token=True" \
MERA_COMMON_SETUP="--model vllm --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR --apply_chat_template --fewshot_as_multiturn" \
bash scripts/run_benchmark_gen.sh
```

### Run specific bencmark manually (ruMMLU example)

Running specific benchmark available with `lm_eval` module.

Example:
```shell
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,max_length=11500 \
--device cuda --output_path="$PWD/mera_results/Mistral-7B-v0.1_defaults" --batch_size=1 --include_path=./benchmark_tasks \
--predict_only --log_samples --seed 1234 --tasks rummlu --num_fewshot=5
```

Example (openai-chat-completions with some server that acts like OpenAI API):
```shell
OPENAI_API_KEY="" lm_eval --model openai-chat-completions \
--model_args model="NousResearch/Meta-Llama-3.1-70B-Instruct,num_concurrent=128,base_url=http://localhost:8000/v1/chat/completions" \
--batch_size=2 --predict_only --log_samples --seed 1234 --verbosity INFO --output_path="./mera_results/Meta-Llama-3.1-70B-Instruct" \
--include_path=./benchmark_tasks --tasks rummlu --limit 256
```

#### Notes on `lm_eval` settings

1. Use `CUDA_VISIBLE_DEVICES` to set cuda device visibility (setting `--device cuda:3` works inconsisitently).

2. `MERA_MODEL_STRING` acts like `--model_args MERA_MODEL_STRING` arguements

`--model_args` is for comma separated parameters of `from_pretrained` method of autoclass. One should be aware of
hardware requirements to run big models and limit maximum input length of models with parameter `max_length`
to avoid out-of-memory errors during run. Also `model_args` may depend on the architecture you have chosen. For example:
- `pretrained=Qwen/Qwen1.5-32B-Chat,dtype=auto,parallelize=True,add_bos_token=True` (HF models)
- `pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,tensor_parallel_size=2,data_parallel_size=3,gpu_memory_utilization=0.7` (vLLM models)

Commonly used parameters of `model_args` or `MERA_MODEL_STRING`:
- `max_length=VALUE` - restricts model context length to some predefined value (for vLLM works also as `max_model_len`)
- `add_bos_token=True` - enables using special tokens be the tokenizer of the chosen model (always True for seq2seq models and all models of gemma family)
- `dtype=VALUE` - the precision of the weights of the model. E.g. `bfloat16`, `float32`.


3. `MERA_COMMON_SETUP` parameters:

3.1 `--model NAME` parameter defines the architecture to be used to run the model
- `--model hf` is used for models compatible with transformers' `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` class;
- `--model vllm` is used for vLLM compatible models;
- `--model nemo_lm` is used for NVIDIA NeMo compatible models;
- `--model openai-chat-completions` is used for OpenAI Chat models and other APIs that act the same as OpenAI one;
- other available architectures: `anthropic-completions`, `dummy`, `gguf`, `ggml`, `gigachat_llms`, `mamba_ssm`, `sparseml`, `neuronx`, `local-completions`, `local-chat-completions`, `openai-completions`, `openvino`, `textsynth`.

3.2 `--batch_size VALUE` is set to define the size of each batch while inferencing the model
`--batch_size=1` is set to use batch size of 1 to maximize benchmark results reproducibility.
`--batch_size=auto` may be set to determine batch size for run automatically based on tasks and inputs maximum value
to start search down is set with `--max_batch_size`. Bigger batches may speed up running whole MERA benchmark,
but results may become irreproducible, so it is not default suggestion. At the same time, for vLLM auto defining batch_size may reduce the time needed to complete computations.

3.3 `--output_path` is path to the directory (will be created) to store data for submission preparation and full logs.

3.4 `--predict_only` important to use this key always, it allows to run on datasets without proper replies provided - no metrics are computed, `bypass` flag and `999` value are used instead (as long as some tasks have closed test sets, no metrics for these tasks can be computed locally).

3.5 `--log_samples` turns on samples logging, should always be set to make the archive of logs to submit it on site.

3.6 `--apply_chat_template` turns on applying chat templates of your model to all requests (see more in [**documentation**](https://huggingface.co/docs/transformers/main/chat_templating)).
Be careful! Bare `--apply_chat_template` implies that the tokenizer has only one chat template. If the tokenizer of your model has more than one chat template ordered in a dictionary, provide the key for the relevant one, like
`--apply_chat_template default` or `--apply_chat_template rag`.

3.7 `--fewshot_as_multiturn` is used to turn fewshots in multi-turn conversation (requires using `--apply_chat_template`).

3.8 `--system_instruction` contains a string that will be used as system prompt for one or more passed tasks (if the chat template of the model does not take into account system prompt, it will be omitted, so that no system prompt is passed to the model; without `--apply_chat_template` system prompt is added to the beginning of each request to the model). Be careful! Some models have chat templates the do not support system prompts and may throw the error.

3.9 Use `--tasks` to provide comma separated list of tasks to run (available options are: `bps`, `chegeka`, `lcs`,
`mathlogicqa`, `multiq`, `parus`, `rcb`, `rudetox`, `ruethics`, `ruhatespeech`, `ruhhh`, `ruhumaneval`, `rummlu`,
`rumodar`, `rumultiar`, `ruopenbookqa`, `rutie`, `ruworldtree`, `rwsd`, `simplear`, `use`).
Avoiding this argument will run all tasks with same provided settings.

3.10 `--num_fewshot` sets fewshot count. MERA supposes to run tasks with the following fewshot count:
* `--num_fewshot=0` (zeroshot) with `multiq`, `rumodar`,`ruethics`, `ruhhh`, `ruhumaneval`, `rucodeeval`;
* `--num_fewshot=1` with `bps`, `lcs`, `chegeka`, `mathlogicqa`, `parus`, `rcb`, `rudetox`, `ruhatespeech`, `rummlu`, `ruworldtree`, `ruopenbookqa`, `rumultiar`, `use`, `rwsd`, `mamuramu`, `rutie`;
* `--num_fewshot=2` with `simplear`.

3.11 `--verbosity VALUE` is meant to define the level of logging the details of the lm-eval running. We recommend using either `INFO` or `ERROR` values.

3.12 `--seed VALUE` sets random, numpy, torch and fewshot seed equal to some value. Alternatively, you can pass `seed 1,2,3,4` four values to define each seed manually.

More information about specific flags and parameters can be found in [lm-evaluation-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs).


### Convert lm-eval logs to submission
Bash script above runs submission zip packing routine. Here is the way to run packing manually.

For converting run

```shell
python scripts/log_to_submission.py --outputs_dir="$PWD/mera_results/rugpt3large_760m_defaults" --dst_dir="$PWD/mera_results/rugpt3large_760m_defaults_submission" --model_args="pretrained=ai-forever/rugpt3large_based_on_gpt2,dtype=auto"
```

Cmd arguments:

* `--outputs_dir` — path to directory with outputs (`MERA_FOLDER` from bash script above)
* `--dst_dir` — directory for store submission zip
* `--logs_public_submit` (`--no-logs_public_submit`) — pack logs for public submission in separate file (true by default)
* `--model_args` — string containing the same info that was passed in `MERA_MODEL_STRING`
* `--gen` — indicates that only generative tasks are to be packed in archive (false by default)

Be careful! After running the model the results will be stored in subdirectory of `MERA_FOLDER`.
Do not use the same `MERA_FOLDER` for running the same model twice (this way only the latest results
will be packed) or different models (this way two or more subdirectories will be created and you will
have to pass `--model_args` to determine which subfolder is to be packed). If you are not using `--model_args`
argument, make sure you provided the full path (including the subdirectory) in `--outputs_dir` argument.
For example above it will be as follows:

```shell
python scripts/log_to_submission.py --outputs_dir="$PWD/mera_results/rugpt3large_760m_defaults/ai-forever__rugpt3large_based_on_gpt2/" --dst_dir="$PWD/mera_results/rugpt3large_760m_defaults_submission"
```
