vllm for dwani.ai

https://gemma-llm.readthedocs.io/en/latest/

- 4B -
    - vllm serve "google/gemma-3-4b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768
- 12 B
    - vllm serve "google/gemma-3-12b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768
- 27 B
    - vllm serve "google/gemma-3-27b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768

- https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm


- cd vllm
- uvicorn vllm-api:app --reload --port 7861 --host 0.0.0.0

pip install flashinfer --index-url https://flashinfer.ai/whl/cu126/torch2.6/

vllm serve google/gemma-3-4b-it \
    --served-model-name gemma-3-4b-it \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --trust-remote-code


curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gemma-3-4b-it", "messages": [{"role": "user", "content": "Who are you?"}]}'