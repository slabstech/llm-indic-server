from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Enable TensorFloat32 for matrix multiplications (NVIDIA Ampere+ GPUs)
torch.set_float32_matmul_precision('high')  # or 'medium' for BF16-based mixed precision[^5]

app = FastAPI()

# Model initialization
MODEL_NAME = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Compile with recommended settings for TF32 optimization
model = torch.compile(model, mode='reduce-overhead')

@app.post("/generate")
async def generate_text(prompt: str, max_length: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7
        )

    return {
        "response": tokenizer.decode(outputs[0], skip_special_tokens=True)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
