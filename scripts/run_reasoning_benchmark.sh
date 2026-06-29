#!/usr/bin/env bash
# Run MERA Reason (mmred, tmath, luzitania, ruaime) and pack submission zip.
#
#   MERA_MODEL_STRING="model=org/my-model,base_url=http://0.0.0.0:8088/v1/completions,..." \
#   bash scripts/run_reasoning_benchmark.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_FOLDER="${OUTPUT_FOLDER:-$PWD/mera_results/}"
MERA_MODEL_STRING="${MERA_MODEL_STRING:?Set MERA_MODEL_STRING}"
MERA_COMMON_SETUP="${MERA_COMMON_SETUP:---model local-completions --batch_size=1 --log_samples --seed 1234 --verbosity ERROR --apply_chat_template --fewshot_as_multiturn}"
GENERATION_KWARGS="${GENERATION_KWARGS:-do_sample=False,until=[\"<|im_end|>\",\"<|eot_id|>\",\"</s>\"],max_gen_toks=1024}"
TASKS="${TASKS:-mmred tmath luzitania ruaime}"
PACK_SUBMISSION="${PACK_SUBMISSION:-1}"

mkdir -p "$OUTPUT_FOLDER"

for task in $TASKS; do
  case "$task" in
    ruaime) num_fewshot=2 ;;
    *) num_fewshot=0 ;;
  esac

  printf '\n===== task: %s =====\n' "$task"
  HF_DATASETS_CACHE="${OUTPUT_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false \
  HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTHONPATH="$PWD" lm_eval --model_args "${MERA_MODEL_STRING}" --tasks "$task" \
    --num_fewshot="$num_fewshot" --gen_kwargs="${GENERATION_KWARGS}" \
    --output_path="${OUTPUT_FOLDER}" ${MERA_COMMON_SETUP} \
    --include_path=./benchmark_tasks ${LIMIT:+--limit "$LIMIT"}
done

if [[ "$PACK_SUBMISSION" == "1" ]]; then
  python scripts/log_to_reasoning_submission.py \
    --outputs_dir "${OUTPUT_FOLDER}" \
    --model_args "${MERA_MODEL_STRING}" \
    --tasks "$(echo "$TASKS" | tr ' ' ',')"
fi

printf '\nDone. Logs: %s\n' "$OUTPUT_FOLDER"
