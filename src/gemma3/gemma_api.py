from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
from typing import List

# Enable TF32 for matrix multiplications
torch.set_float32_matmul_precision('high')

app = FastAPI(
    title="Gemma 3B Multi-Modal API",
    description="OpenAI-style API with Vision capabilities",
    version="1.0.0",
)

# Model initialization
MODEL_NAME = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
llm_model = torch.compile(llm_model, mode='reduce-overhead')

# Placeholder for vision model (replace with actual implementation)
vision_model = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3b-it"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200

class CompletionRequest(BaseModel):
    model: str = "gemma-3b-it"
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 200

class VisionRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "object": "text_completion",
            "choices": [{
                "text": response
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/vision/completions")
async def vision_completion(
    image: UploadFile = File(...),
    request: VisionRequest = None,
):
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Process image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Replace with actual vision model processing
        vision_output = "Image features placeholder"
        
        # Construct multimodal prompt
        combined_prompt = f"Image analysis: {vision_output}\nUser prompt: {request.prompt}"
        
        inputs = tokenizer(combined_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "object": "vision.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
