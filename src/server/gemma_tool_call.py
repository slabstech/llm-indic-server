import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re

# Enable TensorFloat32 for performance
torch.set_float32_matmul_precision('high')

# Function Registry
def get_current_time():
    """Returns the current time as a string."""
    from datetime import datetime
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

if __name__ == "__main__":
    chat = LocalGemmaChat(model_id="google/gemma-3-4b-it")
    print("Welcome to Local Gemma 3 Chat with Function Calling! Type 'quit' to exit.\n")
    print("Example function calls: get_current_time(), add_numbers(3, 5)")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break
        response = chat.send_message(user_input)
        print(f"Gemma: {response}\n")