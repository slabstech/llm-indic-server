import requests
from openai import OpenAI
from PIL import Image
import base64
import io

# 1. Set up the OpenAI client for your vLLM server
client = OpenAI(
    api_key="EMPTY",  # vLLM by default does not require authentication
    base_url="http://localhost:7862/v1"  # Change if your server is on a different host/port
)

# 2. Load and encode your image as base64
img_url = "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()

# 3. Prepare the messages in OpenAI chat format with image content
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
                "text": "Describe the following product: Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur. Category: Toys & Games | Toy Figures & Playsets | Action Figures"
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

# 4. Send the request to the vLLM server
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",  # or your deployed Gemma 3 vision model
    messages=messages,
    max_tokens=256,
    temperature=0.8,
    top_p=1.0
)

# 5. Print the model's response
print(response.choices[0].message.content)
