from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import requests
from openai import OpenAI
from PIL import Image
import base64
import io
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Product Description Generator")

# Pydantic model for request body when using image URL
class ImagePromptRequest(BaseModel):
    image_url: Optional[str] = None
    prompt: str

# Set up OpenAI client for vLLM server
client = OpenAI(
    api_key="EMPTY",  # vLLM does not require authentication
    base_url="http://localhost:7862/v1"  # Adjust if vLLM server is hosted elsewhere
)

def process_image(image_source: str | bytes, is_url: bool = True) -> str:
    """Helper function to process image (from URL or bytes) and return base64 string."""
    try:
        if is_url:
            # Fetch image from URL
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        else:
            # Process uploaded file
            image = Image.open(io.BytesIO(image_source)).convert("RGB")
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        return img_b64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def generate_description(img_b64: str, prompt: str) -> str:
    """Helper function to generate product description using vLLM."""
    try:
        # Prepare messages for vLLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful product description generator."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]

        # Send request to vLLM server
        response = client.chat.completions.create(
            model="google/gemma-3-4b-it",  # Adjust to your deployed model
            messages=messages,
            max_tokens=256,
            temperature=0.8,
            top_p=1.0
        )

        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating description: {str(e)}")

@app.post("/generate-description/")
async def generate_description_from_url(request: ImagePromptRequest):
    """
    Generate product description from an image URL and prompt.
    """
    if not request.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")
    
    # Process image from URL
    img_b64 = process_image(request.image_url, is_url=True)
    
    # Generate description
    description = generate_description(img_b64, request.prompt)
    
    return {"description": description}

@app.post("/generate-description/upload/")
async def generate_description_from_file(prompt: str, file: UploadFile = File(...)):
    """
    Generate product description from an uploaded image file and prompt.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    # Read and process uploaded image
    image_bytes = await file.read()
    img_b64 = process_image(image_bytes, is_url=False)
    
    # Generate description
    description = generate_description(img_b64, prompt)
    
    return {"description": description}

# Run the server with: uvicorn filename:app --reload