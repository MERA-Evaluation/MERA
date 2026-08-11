TASKS="mmred tmath luzitania ruaime"
SUBMISSION_DIR="reasoning_submission/"
OUTPUT_PATH="mera_results/"
PORT=8088
MERA_MODEL_STRING="model=$MODEL_NAME,num_concurrent=16,timeout=3000,max_retries=5,base_url=http://0.0.0.0:$PORT/v1/chat/completions"

python scripts/log_to_reasoning_submission.py \
    --dst_dir "$SUBMISSION_DIR" \
    --outputs_dir "${OUTPUT_PATH}" \
    --model_args "${MERA_MODEL_STRING}" \
    --tasks "$(echo "$TASKS" | tr ' ' ',')"