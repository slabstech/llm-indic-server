vllm for dwani.ai

https://gemma-llm.readthedocs.io/en/latest/

- 4B -
    - vllm serve "google/gemma-3-4b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768
- 12 B
    - vllm serve "google/gemma-3-12b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768
- 27 B
    - vllm serve "google/gemma-3-27b-it" --port 8000 --tensor-parallel-size 1 --max-model-len 32768

- https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm