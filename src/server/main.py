from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from IndicTransToolkit import IndicProcessor

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
model_trans_indic_en = None
tokenizer_trans_indic_en = None
model_trans_en_indic = None
tokenizer_trans_en_indic = None

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Load the causal LM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    settings.model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

# Load the translation models and tokenizers
src_lang, tgt_lang = "eng_Latn", "kan_Knda"
model_name_trans_indic_en = "ai4bharat/indictrans2-indic-en-dist-200M"
model_name_trans_en_indic = "ai4bharat/indictrans2-en-indic-dist-200M"

tokenizer_trans_indic_en = AutoTokenizer.from_pretrained(model_name_trans_indic_en, trust_remote_code=True)
model_trans_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_indic_en,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

tokenizer_trans_en_indic = AutoTokenizer.from_pretrained(model_name_trans_en_indic, trust_remote_code=True)
model_trans_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_en_indic,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

logger.info("Models and tokenizers initialized successfully")

# Request and Response Models
class ChatRequest(BaseModel):
    prompt: str

    @field_validator('prompt')
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

class ChatResponse(BaseModel):
    response: str

# API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "your-secure-api-key":  # Replace with env var in production
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Translation function
def translate_text(text, src_lang, tgt_lang):
    if src_lang == "kan_Knda" and tgt_lang == "eng_Latn":
        tokenizer_trans = tokenizer_trans_indic_en
        model_trans = model_trans_indic_en
    elif src_lang == "eng_Latn" and tgt_lang == "kan_Knda":
        tokenizer_trans = tokenizer_trans_en_indic
        model_trans = model_trans_en_indic
    else:
        raise ValueError("Unsupported language pair")

    batch = ip.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    inputs = tokenizer_trans(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated_tokens = model_trans.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer_trans.as_target_tokenizer():
        generated_tokens = tokenizer_trans.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return translations[0]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.model_name}

# Home redirect
@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("100/minute")
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Received prompt: {chat_request.prompt}")
    try:
        prompt = chat_request.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Step 1: Translate Kannada prompt to English
        translated_prompt = translate_text(prompt, src_lang="kan_Knda", tgt_lang="eng_Latn")
        logger.info(f"Translated prompt to English: {translated_prompt}")

        # Step 2: Generate LLM response in English
        messages = [
            {"role": "system", "content": "You are Dhwani, a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
            {"role": "user", "content": translated_prompt}
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
            do_sample=True,
            temperature=0.7
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Generated English response: {response}")

        # Step 3: Translate English response to Kannada
        translated_response = translate_text(response, src_lang="eng_Latn", tgt_lang="kan_Knda")
        logger.info(f"Translated response to Kannada: {translated_response}")

        return ChatResponse(response=translated_response)
        
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

    uvicorn.run(app, host=args.host, port=args.port)