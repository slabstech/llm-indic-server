from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

from fastapi.responses import RedirectResponse

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = None
tokenizer = None

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

app = FastAPI()


@app.middleware("http")
async def lazy_load_model(request: Request, call_next):
    global model, tokenizer

    if model is None or tokenizer is None:
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    prompt = request.prompt
    messages = [
        {"role": "system", "content": "You are Dhwani, built for Indian languages. You are a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return ChatResponse(response=response)

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)