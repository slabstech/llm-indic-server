# chat_with_gemma_function_calling.py

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re

# Example function registry
def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_numbers(a, b):
    return str(float(a) + float(b))

# Map function names to actual Python functions
FUNCTION_REGISTRY = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

class LocalGemmaChat:
    """
    Chat interface for Gemma 3 models with robust function calling support.
    Handles code blocks and print-wrapped function calls.
    """

    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.model_id = model_id
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are a helpful assistant. You can call functions such as get_current_time() "
                    "and add_numbers(a, b). If you need to use a function, output it as a Python code block."
                )}]
            }
        ]

    def _extract_function_call(self, text):
        """
        Extract function call from model output.
        Handles code blocks and print-wrapped calls.
        Returns (function_name, args) or (None, None)
        """
        # Extract code block if present
        code_block = None
        code_match = re.search(r"``````", text)
        if code_match:
            code_block = code_match.group(1).strip()
        else:
            code_block = text.strip()

        # Look for print-wrapped function call
        print_match = re.match(r"print\((.+)\)", code_block)
        if print_match:
            func_call = print_match.group(1)
        else:
            func_call = code_block

        # Now extract function name and arguments
        match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)", func_call)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            # Split args by comma, strip whitespace and quotes
            args = [arg.strip().strip("\"'") for arg in args_str.split(",")] if args_str else []
            return func_name, args
        return None, None

    def _handle_function_call(self, func_name, args):
        """
        Execute the function if registered and return its result as a string.
        """
        func = FUNCTION_REGISTRY.get(func_name)
        if not func:
            return f"[Function '{func_name}' not found.]"
        try:
            result = func(*args)
            return f"[Function '{func_name}' result: {result}]"
        except Exception as e:
            return f"[Error calling '{func_name}': {e}]"

    def send_message(self, user_message, max_new_tokens=100):
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })

        # Prepare input for the model
        inputs = self.processor.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        # Check for function call in the model's response
        func_name, args = self._extract_function_call(decoded)
        if func_name:
            # Function call detected, execute it
            func_result = self._handle_function_call(func_name, args)
            # Append function result to conversation
            self.messages.append({
                "role": "function",
                "name": func_name,
                "content": [{"type": "text", "text": func_result}]
            })
            # Optionally, let the model generate a follow-up answer with the function result
            followup_inputs = self.processor.apply_chat_template(
                self.messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            followup_input_len = followup_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                followup_generation = self.model.generate(
                    **followup_inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
                followup_generation = followup_generation[0][followup_input_len:]
            followup_decoded = self.processor.decode(followup_generation, skip_special_tokens=True)
            # Append assistant's follow-up reply
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": followup_decoded}]
            })
            return f"{decoded}\n\n{func_result}\n\n{followup_decoded}"
        else:
            # No function call, just append assistant reply
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
