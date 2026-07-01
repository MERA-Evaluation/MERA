TASKS="mmred tmath luzitania ruaime"
OUTPUT_PATH="mera_results/"
PORT=8088
MERA_MODEL_STRING="model=$MODEL_NAME,num_concurrent=8,timeout=2000,max_retries=5,base_url=http://0.0.0.0:$PORT/v1/chat/completions"

python scripts/log_to_reasoning_submission.py \
    --outputs_dir "${OUTPUT_PATH}" \
    --model_args "${MERA_MODEL_STRING}" \
    --tasks "$(echo "$TASKS" | tr ' ' ',')"