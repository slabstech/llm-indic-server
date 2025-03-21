# LLM Indic Server

## Overview
Large Language Model for Indic Langues


## Table of Contents
- [Getting Started](#getting-started-development)
- [Execute Sample LLM](#run-sample-code)
- [Other llms](#other-llms)
- [Running with FastAPI Server](#running-with-fastapi-server)

## Getting Started - Development

### For Development (Local)
- **Prerequisites**: Python 3.10
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
  - ```bash
    pip install -r requirements.txt
    ```

### Run Sample Code

```python
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(torch.bfloat16)


with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

outputs = tokenizer.batch_decode(outputs)
```

 #### Run code
  - ```python llm_code.py```


### Other llms 

- ```pip install -r indic-requirements.txt```

- ```python src/local/translate_llm.py```


### For server Development 
#### Running with FastAPI Server
Run the server using FastAPI with the desired model:
- ```bash
  python src/server/qwen_api.py --port 7860 --language kn --host 0.0.0.0
  ```

## Contact
- For any questions or issues, please open an issue on GitHub or contact us via email.
- For collaborations
  - Join the discord group - [invite link](https://discord.gg/WZMCerEZ2P) 
- For business queries, Email : info (at) slabstech (dot) com



<!-- 
### Downloading LLM Models
Models can be downloaded from  HuggingFace repository:


#### Qwen/Qwen2.5-1.5B-Instruct - 3GB model
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

#### Translation - Indic to English

```bash
huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M
```

#### Translation - Enlglish to Indic
```bash
huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M
```

### Local Gradio Development
- For Kannada LLM
  ```bash
  python src/ux/app_local_kannada.py 
  ```
- For English LLM
  ```bash 
  src/ux/app_local.py
  ```
-->

<!-- 
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
-->


<!-- 
## References

## Additional Resources


### Running with Transformers
```bash
python hf_llama_3.py
```


- server-setup.sh - Use for container deployment on OlaKrutrim AI Pod
-->
<!-- 

curl -X 'POST' \
  'https://gaganyatri-llm-indic-server.hf.space/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "what is the capital of karnataka ?"
}'


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


-->


## Citation

```bibtex
@article{gala2023indictrans,
title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vfT4YuzAYA},
note={}
}
```
