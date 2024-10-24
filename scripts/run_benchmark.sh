#!/bin/bash

# internal MERA_COMMON_SETUP will be assigned 'default' value if external MERA_COMMON_SETUP not set or null.
# The value of external MERA_COMMON_SETUP remains untouched.
MERA_COMMON_SETUP_default="--model hf --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR"
MERA_COMMON_SETUP="${MERA_COMMON_SETUP:-$MERA_COMMON_SETUP_default}"
RUHUMANEVAL_GEN_KWARGS="${RUHUMANEVAL_GEN_KWARGS:-temperature=0.6,do_sample=True}"
GENERATION_KWARGS="${GENERATION_KWARGS:-do_sample=False}"

FEWSHOTS=(
 2
 1
 0
 1
)

TASKS=(
"simplear"
"bps lcs chegeka mathlogicqa parus rcb rudetox ruhatespeech rummlu ruworldtree ruopenbookqa rumultiar use rwsd mamuramu"
"multiq rumodar ruethics ruhhh ruhumaneval rucodeeval"
"rutie"
)

for fewshot_idx in "${!FEWSHOTS[@]}"
do
  for cur_task in ${TASKS[$fewshot_idx]}
  do
    printf "task: %s \n" "$cur_task"
    if [[ "$cur_task" == "ruhumaneval" || "$cur_task" == "rucodeeval" ]]; then
        GEN_KWARGS=${RUHUMANEVAL_GEN_KWARGS}; else
        GEN_KWARGS=${GENERATION_KWARGS}; fi

    if test -z "${SYSTEM_PROMPT}"
    then
        HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  PYTHONPATH=$PWD \
        lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
        --num_fewshot=${FEWSHOTS[$fewshot_idx]} --gen_kwargs="${GEN_KWARGS}" \
        --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} \
        --include_path=./benchmark_tasks
    else
        PROCESSED_SYSTEM=$(printf "%b" "$SYSTEM_PROMPT")
        HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  PYTHONPATH=$PWD \
        lm_eval --model hf --model_args "${MERA_MODEL_STRING}" --tasks $cur_task \
        --num_fewshot=${FEWSHOTS[$fewshot_idx]} --gen_kwargs="${GEN_KWARGS}" \
        --output_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} --system_instruction="${PROCESSED_SYSTEM}" \
        --include_path=./benchmark_tasks
    fi

  done
done

# Try to save regular submission
HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
python scripts/log_to_submission.py --outputs_dir "${MERA_FOLDER}" --dst_dir "${MERA_FOLDER}_submission" --model_args ${MERA_MODEL_STRING}

# Remove datasets cache folder
rm -r "${MERA_FOLDER}/ds_cache"
