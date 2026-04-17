vllm serve Qwen/Qwen3.5-9B --port 8000 --tensor-parallel-size 1 --max-model-len 32768 --reasoning-parser qwen3 --language-model-only
