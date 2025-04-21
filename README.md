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
  python src/server/main.py --port 7860 --host 0.0.0.0
  ```

## Contact
- For any questions or issues, please open an issue on GitHub or contact us via email.
- For collaborations
  - Join the discord group - [invite link](https://discord.gg/WZMCerEZ2P) 
- For business queries, Email : info (at) slabstech (dot) com



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
```bibtex
@article{gemma_2025,
    title={Gemma 3},
    url={https://goo.gle/Gemma3Report},
    publisher={Kaggle},
    author={Gemma Team},
    year={2025}
}
```