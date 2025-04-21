from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import io
from typing import List

# Enable TF32 and CUDA optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

app = FastAPI(
    title="Gemma 3B Multi-Modal API",
    description="CUDA-optimized API with Fast Image Processing",
    version="1.0.0",
)

# Model initialization
MODEL_ID = "google/gemma-3-4b-it"
device = "cuda"  # Force CUDA usage

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    use_fast=True  # Critical optimization
).to(device)  # Processor on GPU

class MessageContentItem(BaseModel):
    type: str
    text: str = None
    image: str = None  # URL/path for non-upload scenarios

class Message(BaseModel):
    role: str
    content: List[MessageContentItem]

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3b-it"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Convert messages to processor format
        hf_messages = []
        for msg in request.messages:
            content_items = []
            for item in msg.content:
                if item.type == "text":
                    content_items.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    content_items.append({"type": "image", "image": item.image})
            hf_messages.append({"role": msg.role, "content": content_items})

        # Process and generate
        inputs = processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True
            )
            generation = generation[0][input_len:]

        response = processor.decode(generation, skip_special_tokens=True)

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

@app.post("/v1/vision/completions")
async def vision_completion(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(200),
    temperature: float = Form(0.7)
):
    try:
        # Validate and process image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        # Create message with image tensor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process on GPU
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        input_len = inputs["input_ids"].shape[-1]

        # Optimized CUDA generation
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            generation = generation[0][input_len:]

        response = processor.decode(generation, skip_special_tokens=True)

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
