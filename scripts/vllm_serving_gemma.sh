vllm serve google/gemma-4-26B-A4B-it \
--port 8000 \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--reasoning-parser gemma4 \
--gpu-memory-utilization 0.9 \
--seed 1234 \
--trust-remote-code \
--dtype bfloat16 \