import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re
from datetime import datetime

# Enable TensorFloat32 for performance
torch.set_float32_matmul_precision('high')

# Function Registry
def get_current_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a, b):
    """Adds two numbers and returns the result as a string."""
    return str(float(a) + float(b))

FUNCTION_REGISTRY = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

class LocalGemmaChat:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        """Initialize the chat system with the model and a clear system prompt."""
        self.model_id = model_id
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are a helpful assistant. You can call predefined functions such as "
                    "get_current_time() and add_numbers(a, b). To call a function, output a "
                    "Python code block with a direct call to the function, like "
                    "print(get_current_time()) or print(add_numbers(3, 5)). "
                    "Do not define new functions."
                )}]
            }
        ]

    def _extract_function_call(self, text):
        """
        Extract a function call from the text if it matches a registered function and 
        contains no function definitions.
        """
        code_match = re.search(r"```(.*?)```", text, re.DOTALL)
        if code_match:
            code_block = code_match.group(1).strip()
            lines = code_block.split('\n')
            for line in lines:
                if line.strip().startswith('def '):
                    return None, None
            print_match = re.search(r"print\((.+)\)", code_block)
            func_call = print_match.group(1).strip() if print_match else code_block.strip()
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)", func_call)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                args = [arg.strip().strip("\"'") for arg in args_str.split(",")] if args_str else []
                if func_name in FUNCTION_REGISTRY:
                    return func_name, args
        return None, None

    def _handle_function_call(self, func_name, args):
        """
        Execute the function call and return the result.
        """
        func = FUNCTION_REGISTRY.get(func_name)
        if not func:
            return f"[Function '{func_name}' not found.]"
        try:
            result = func(*args)
            return f"{result}"
        except Exception as e:
            return f"[Error calling '{func_name}': {e}]"

    def send_message(self, user_message, max_new_tokens=100):
        """
        Send a message to the model and process its response.
        """
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })

        inputs = self.processor.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, top_p=None, top_k=None
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        func_name, args = self._extract_function_call(decoded)
        if func_name:
            func_result = self._handle_function_call(func_name, args)
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"Called {func_name} with args {args}"}]
            })
            self.messages.append({
                "role": "system",
                "content": [{"type": "text", "text": f"Function result: {func_result}"}]
            })
            return func_result
        else:
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": decoded}]
            })
            return decoded

# FastAPI app
app = FastAPI()

# Pydantic model for request body (mimicking OpenAI's chat completion request)
class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str | None = None
    max_tokens: int | None = 100
    temperature: float | None = None  # Not used in this example, but included for compatibility

# Initialize the chat model
chat = LocalGemmaChat()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI API-compatible endpoint for chat completions.
    """
    try:
        # Extract the last user message from the messages list
        user_message = next(
            (msg["content"] for msg in request.messages if msg["role"] == "user"),
            None
        )
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in request.")

        # Get the response from the chat model
        response = chat.send_message(user_message, max_new_tokens=request.max_tokens or 100)

        # Format the response to match OpenAI API structure
        return {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": chat.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Placeholder; actual token counting not implemented
                "completion_tokens": 0,  # Placeholder
                "total_tokens": 0  # Placeholder
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)