from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, BaseSettings, validator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configuration Settings
class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    log_level: str = "info"

settings = Settings()

# Setup logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0"
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Model and tokenizer as global variables
model = None
tokenizer = None

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Request and Response Models
class ChatRequest(BaseModel):
    prompt: str

    @validator('prompt')
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

class ChatResponse(BaseModel):
    response: str

# Startup event to initialize model
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        logger.info("Model and tokenizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.model_name}

# Home redirect
@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("100/minute")
async def chat(request: ChatRequest):
    logger.info(f"Received prompt: {request.prompt}")
    try:
        prompt = request.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            
        messages = [
            {"role": "system", "content": "You are Dhwani, built for Indian languages. You are a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=settings.max_tokens,
            do_sample=True,  # Add sampling for variety
            temperature=0.7  # Control creativity
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("Response generated successfully")
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",  # Note: file should be named 'main.py' or adjust this
        host=args.host,
        port=args.port,
        workers=4,
        log_level=settings.log_level,
        reload=False,
        timeout_keep_alive=30
    )