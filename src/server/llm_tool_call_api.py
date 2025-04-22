import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import time
import uuid

# Enable TensorFloat32 for performance
torch.set_float32_matmul_precision('high')

# Function Registry
def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a, b):
    return str(float(a) + float(b))

FUNCTION_REGISTRY = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

class LocalGemmaChat:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.model_id = model_id
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.messages = []

    def _extract_function_call(self, text):
        code_match = re.search(r"``````", text, re.DOTALL)
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
        func = FUNCTION_REGISTRY.get(func_name)
        if not func:
            return f"[Function '{func_name}' not found.]"
        try:
            result = func(*args)
            return f"{result}"
        except Exception as e:
            return f"[Error calling '{func_name}': {e}]"

    def send_message(self, max_new_tokens=100):
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

# OpenAI-compatible request and response models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Any] = None

class Choice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# Lifespan event handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    print("Loading Gemma model...")
    chatbot = LocalGemmaChat(model_id="google/gemma-3-4b-it")
    print("Model loaded.")
    yield
    # Optional: cleanup logic here

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global chatbot

    # Prepare messages for the chat template
    # Always start with a system prompt
    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": (
            "You are a helpful assistant. You can call predefined functions such as "
            "get_current_time() and add_numbers(a, b). To call a function, output a "
            "Python code block with a direct call to the function, like "
            "print(get_current_time()) or print(add_numbers(3, 5)). "
            "Do not define new functions."
        )}]
    }

    # Convert OpenAI-style messages to the format expected by the processor
    chat_messages = [system_prompt]
    for msg in request.messages:
        chat_messages.append({
            "role": msg.role,
            "content": [{"type": "text", "text": msg.content}]
        })

    # Check alternation: after system, must be user, then assistant, etc.
    roles = [m["role"] for m in chat_messages]
    # Remove system for alternation check
    roles_no_system = [r for r in roles if r != "system"]
    for i in range(1, len(roles_no_system)):
        if roles_no_system[i] == roles_no_system[i-1]:
            raise HTTPException(status_code=400, detail="Conversation roles must alternate user/assistant/user/assistant/...")

    # Set messages for the chatbot
    chatbot.messages = chat_messages

    # Call model and get response
    response_text = chatbot.send_message(max_new_tokens=request.max_tokens)
    resp = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message={"role": "assistant", "content": response_text},
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_tool_call_api:app", host="0.0.0.0", port=7861, reload=True)
