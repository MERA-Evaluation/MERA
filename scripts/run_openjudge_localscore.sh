#!/bin/bash

MERA_COMMON_SETUP_default="--model hf --device cuda --batch_size=1 --log_samples --seed 1234 --verbosity ERROR"
MERA_COMMON_SETUP="${MERA_COMMON_SETUP:-$MERA_COMMON_SETUP_default}"
GENERATION_KWARGS="${GENERATION_KWARGS:-do_sample=False}"

if [[ -n "${MERA_OPENJUDGE_TASKS}" ]]; then
  TASKS="${MERA_OPENJUDGE_TASKS}"
else
  mapfile -t DISCOVERED_TASKS < <(rg -l "custom_openjudge_localscore_task.yaml" benchmark_tasks -g "*_localscore.yaml" | sed 's#.*[\\/]##' | sed 's#\.yaml$##')
  TASKS="${DISCOVERED_TASKS[*]}"
fi

if [[ -z "${TASKS}" ]]; then
  echo "No openjudge localscore tasks found. Set MERA_OPENJUDGE_TASKS or add *_localscore.yaml tasks that include custom_openjudge_localscore_task.yaml."
  exit 1
fi

for cur_task in ${TASKS}
do
  printf "task: %s \n" "$cur_task"
  if test -z "${SYSTEM_PROMPT}"
  then
    HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=$PWD \
    lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
    --gen_kwargs="${GENERATION_KWARGS}" --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} \
    --include_path=./benchmark_tasks
  else
    PROCESSED_SYSTEM=$(printf "%b" "$SYSTEM_PROMPT")
    HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=$PWD \
    lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
    --gen_kwargs="${GENERATION_KWARGS}" --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} \
    --system_instruction="${PROCESSED_SYSTEM}" --include_path=./benchmark_tasks
  fi
done

rm -r "${MERA_FOLDER}/ds_cache"
