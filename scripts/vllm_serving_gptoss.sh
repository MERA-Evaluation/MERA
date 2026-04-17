vllm serve Qwen/Qwen3.5-9B \
--port 8000 \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--gpu-memory-utilization 0.9 \
--seed 1234 \
--trust-remote-code \
--dtype bfloat16 \
