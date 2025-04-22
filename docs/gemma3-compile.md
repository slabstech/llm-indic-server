Gemma3 - Torch Compile

python3.10 -m venv venv

 source venv/bin/activate

pip install git+https://github.com/huggingface/transformers.git@e5ac23081ec4021818a21d7442d396f31de8c30c

pip install accelerate pillow torchvision fastapi uvicorn python-multipart

- Reference
  - https://github.com/huggingface/transformers/pull/37447

  - https://huggingface.co/google/gemma-3-4b-it#running-the-model-on-a-singlemulti-gpu




- https://github.com/huggingface/transformers

- Function calling - https://www.philschmid.de/gemma-function-calling
- https://github.com/philschmid/gemini-samples/blob/main/examples/gemma-function-calling.ipynb