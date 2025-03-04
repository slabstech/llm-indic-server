import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
import os
import tempfile
import subprocess
import asyncio
import io
import logging
from logging.handlers import RotatingFileHandler
from time import time
from typing import List
import argparse
import uvicorn
import shutil


# Configure logging with log rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("transcription_api.log", maxBytes=10*1024*1024, backupCount=5), # 10MB per file, keep 5 backup files
        logging.StreamHandler() # This will also print logs to the console
    ]
)

class ASRModelManager:
    def __init__(self, default_language="kn", device_type="cuda"):
        self.default_language = default_language
        self.device_type = device_type
        self.model_language = {
            "kannada": "kn",
            "urdu": "ur"
        }
        self.config_models = {
            "as": "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large",
            "bn": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large",
            "te": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
            "ur": "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large"
        }
        self.model = self.load_model(self.default_language)

    def load_model(self, language_id="kn"):
        model_name = self.config_models.get(language_id, self.config_models["kn"])
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)

        device = torch.device(self.device_type if torch.cuda.is_available() and self.device_type == "cuda" else "cpu")
        model.freeze() # inference mode
        model = model.to(device) # transfer model to device

        return model


app = FastAPI()
asr_manager = ASRModelManager()

# Define the response model
class TranscriptionResponse(BaseModel):
    text: str

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    start_time = time()
    try:
        # Check file extension

        try:
            # Transcribe the audio
            language_id = asr_manager.model_language.get(language, asr_manager.default_language)

            end_time = time()
            logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
            return JSONResponse(content={"text": joined_transcriptions})
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
        except Exception as e:
            logging.error(f"An error occurred during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
        finally:
            # Clean up temporary files
            for chunk_file_path in chunk_file_paths:
                if os.path.exists(chunk_file_path):
                    os.remove(chunk_file_path)
            asr_manager.cleanup()
    except HTTPException as e:
        logging.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on.")
    parser.add_argument("--language", type=str, default="kn", help="Default language for the ASR model.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    args = parser.parse_args()

    asr_manager = ASRModelManager(default_language=args.language, device_type=args.device)
    uvicorn.run(app, host=args.host, port=args.port)