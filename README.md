# LLM Indic Server

## Overview
Large Language Model for Indic Langues



### Qwen/Qwen2.5-1.5B-Instruct - 3GB  - test
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```


curl -X 'POST' \
  'https://gaganyatri-llm-indic-server.hf.space/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "what is the capital of karnataka ?"
}'




## Table of Contents
- [Getting Started](#getting-started-development)
  - [For Production (Docker)](#for-production-docker)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [For Development (Local)](#for-development-local)
    - [Prerequisites](#prerequisites-1)
    - [Steps](#steps-1)
- [Downloading Translation Models](#downloading-translation-models)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Live Server](#live-server)
- [Evaluating Results](#evaluating-results)
- [Building Docker Image](#building-docker-image)
  - [Run the Docker Image](#run-the-docker-image)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Additional Resources](#additional-resources)
  - [Running Nemo Model](#running-nemo-model)
  - [Running with Transformers](#running-with-transformers)


## Getting Started - Development

### For Production (Docker)
- **Prerequisites**: Docker and Docker Compose
- **Steps**:
  1. **Start the server**:
  For GPU
  ```bash
  docker compose -f compose.yaml up -d
  ```
  For CPU only
  ```bash
  docker compose -f cpu-compose.yaml up -d
  ```
  2. **Update source and target languages**:
  Modify the `compose.yaml` file to set the desired language. Example configurations:
  - **Kannada**:
  ```yaml
  language: kn
  ```
  - **Hindi**:
  ```yaml
  language: hi
  ```

### For Development (Local)
- **Prerequisites**: Python 3.6+
- **Steps**:
  1. **Create a virtual environment**:
  ```bash
  python -m venv venv
  ```
  2. **Activate the virtual environment**:
  ```bash
  source venv/bin/activate
  ```
  On Windows, use:
  ```bash
  venv\Scripts\activate
  ```
  3. **Install dependencies**:
  - For GPU
      ```bash
      pip install -r requirements.txt
      ```
  - For CPU only
      ```
      pip install -r cpu-requirements.txt
      ```

## Downloading LLM Models
Models can be downloaded from  HuggingFace repository:

for t4 inference, test models between 2-4 GB vRAM.


### deepseek r1 1.5.b - 3.5 Gb
```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### DeepSeek-R1-Distill-Qwen-7B - 16 GB
```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 
```

### QWEN 2.5 VL 3B Instruct - 6GB
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
```

### Qwen/Qwen2.5-1.5B-Instruct - 3GB  - test
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```
### Qwen/Qwen2.5-3B-Instruct - 6GB
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```
## Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 - 2GB - test
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
```
### QWEN 2.5 VL 3B AWQ Instruct - 3GB  - test
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

### Qwen/Qwen2.5-0.5B-Instruct - 900 MB - CPU Model ?
```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```


#### Llam 3.2 1b - 2.5 GB
```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
```

#### Llam 3.2 3b - 6.5 GB
```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
```

runnig deepseek-r1 locally

https://github.com/deepseek-ai/DeepSeek-V3?tab=readme-ov-file#6-how-to-run-locally




https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ

## Running with FastAPI Server
Run the server using FastAPI with the desired model:
- for GPU
  ```bash
  python src/llm_api.py --port 7860 --language kn --host 0.0.0.0 --device gpu
  ```
- for CPU only
  ```bash
  python src/llm_api.py --port 7860 --language kn --host 0.0.0.0 --device cpu
  ```

## Evaluating Results
You can evaluate the ASR transcription results using `curl` commands. Below are examples for Kannada audio samples.
**Note**: GitHub doesn’t support audio playback in READMEs. Download the sample audio files and test them locally with the provided `curl` commands to verify transcription results.

### Kannada Transcription Examples

#### Sample 1: kannada_sample_1.wav
- **Audio File**: [samples/kannada_sample_1.wav](samples/kannada_sample_1.wav)
- **Command**:
```bash
curl -X 'POST' 'http://loca?language=kannada' -H 'accept: application/json'   -H 'Content-Type: multipa'Content-Type  multipart/form-data' -F 'file=@samples/kannada_sample_1.wav;type=audio/x-wav'
```
- **Expected Output**:
```ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು```
Translation: "What is the capital of Karnataka"


## Building Docker Image
Build the Docker image locally:
```bash
docker build -t slabstech/llm_indic_server -f Dockerfile .
```

### Run the Docker Image
```
docker run --gpus all -it --rm -p 7860:7860 slabstech/llm_indic_server
```


## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate


## References

## Additional Resources


### Running with Transformers
```bash
python hf_llama_3.py
```


- server-setup.sh - Use for container deployment on OlaKrutrim AI Pod