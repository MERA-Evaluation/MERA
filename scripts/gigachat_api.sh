export HF_HOME=$HF_HOME
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PATH=$ENV_PATH/bin:$PATH
export GIGACHAT_TOKEN=$GIGACHAT_TOKEN
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE

echo "Запуск lm_eval для $MODEL_NAME и задачи $TASK_NAME"

cd $WORKING_DIR
    
lm_eval \
--model gigachat-completions \
--model_args "model=$MODEL_NAME,num_concurrent=8,timeout=1000,max_retries=5,base_url=$BASE_URL" \
--gen_kwargs "max_gen_toks=$MAX_GEN_TOKS,until=[]" \
--output_path="$OUTPUT_PATH/$MODEL_NAME" \
--device cpu \
--batch_size auto \
--include_path=./benchmark_tasks \
--apply_chat_template \
--fewshot_as_multiturn \
--num_fewshot $NUM_FEWSHOT \
--verbosity ERROR \
--log_samples \
--seed 1234 \
--tasks $TASK_NAME \
--use_cache "$CACHE_DIR/$MODEL_NAME/$TASK_NAME"