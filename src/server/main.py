import argparse
import io
import os
from time import time
from typing import List

import tempfile
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import requests

from logging_config import logger
from tts_config import SPEED, ResponseFormat, config as tts_config
from gemma_llm import LLMManager
#from auth import get_api_key, settings as auth_settings

# Supported language codes
SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya"
}

class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-27b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

llm_manager = LLMManager(settings.llm_model_name)

class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"  # Default to Kannada
    tgt_lang: str = "kan_Knda"  # Default to Kannada

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

    @field_validator("src_lang", "tgt_lang")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")
        return v

class ChatResponse(BaseModel):
    response: str

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

    @field_validator("src_lang", "tgt_lang")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")
        return v

class TranslationResponse(BaseModel):
    translations: List[str]

async def call_external_translation(sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    external_url = "https://gaganyatri-dhwani-server.hf.space/v1/translate"
    payload = {
        "sentences": sentences,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    }
    try:
        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=60
        )
        response.raise_for_status()
        translations = response.json().get("translations", [])
        if not translations or len(translations) != len(sentences):
            logger.warning(f"Unexpected response format: {response.json()}")
            raise ValueError("Invalid response from translation service")
        return translations
    except requests.Timeout:
        logger.error("Translation request timed out")
        raise HTTPException(status_code=504, detail="Translation service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/unload_all_models")
async def unload_all_models(
    #api_key: str = Depends(get_api_key)
    ):
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@app.post("/v1/load_all_models")
async def load_all_models(
    #api_key: str = Depends(get_api_key)
    ):
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    logger.info(f"Received translation request: {request.dict()}")
    try:
        translations = await call_external_translation(
            sentences=request.sentences,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang
        )
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest,
                #api_key: str = Depends(get_api_key)
                ):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    try:
        # Translate prompt to English if src_lang is not English
        if chat_request.src_lang != "eng_Latn":
            translated_prompt = await call_external_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn"
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt already in English, no translation needed")

        # Generate response in English
        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated English response: {response}")

        # Translate response to target language if tgt_lang is not English
        if chat_request.tgt_lang != "eng_Latn":
            translated_response = await call_external_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info("Response kept in English, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/visual_query/")
async def visual_query(
    file: UploadFile = File(...),
    query: str = Body(...),
    src_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    tgt_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    #api_key: str = Depends(get_api_key)
):
    try:
        image = Image.open(file.file)
        if image.size == (0, 0):
            raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")
        
        # Translate query to English if src_lang is not English
        if src_lang != "eng_Latn":
            translated_query = await call_external_translation(
                sentences=[query],
                src_lang=src_lang,
                tgt_lang="eng_Latn"
            )
            query_to_process = translated_query[0]
            logger.info(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.info("Query already in English, no translation needed")

        # Generate response in English
        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info(f"Generated English answer: {answer}")

        # Translate answer to target language if tgt_lang is not English
        if tgt_lang != "eng_Latn":
            translated_answer = await call_external_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang
            )
            final_answer = translated_answer[0]
            logger.info(f"Translated answer to {tgt_lang}: {final_answer}")
        else:
            final_answer = answer
            logger.info("Answer kept in English, no translation needed")

        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/chat_v2", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    src_lang: str = Form("kan_Knda"),
    tgt_lang: str = Form("kan_Knda"),
    #api_key: str = Depends(get_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language code. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")
    
    logger.info(f"Received prompt: {prompt}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, Image provided: {image is not None}")

    try:
        if image:
            image_data = await image.read()
            if not image_data:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            img = Image.open(io.BytesIO(image_data))
            
            # Translate prompt to English if src_lang is not English
            if src_lang != "eng_Latn":
                translated_prompt = await call_external_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn"
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")

            decoded = await llm_manager.chat_v2(img, prompt_to_process)
            logger.info(f"Generated English response: {decoded}")

            # Translate response to target language if tgt_lang is not English
            if tgt_lang != "eng_Latn":
                translated_response = await call_external_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")
        else:
            # Translate prompt to English if src_lang is not English
            if src_lang != "eng_Latn":
                translated_prompt = await call_external_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn"
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")
            
            decoded = await llm_manager.generate(prompt_to_process, settings.max_tokens)
            logger.info(f"Generated English response: {decoded}")
            
            # Translate response to target language if tgt_lang is not English
            if tgt_lang != "eng_Latn":
                translated_response = await call_external_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)