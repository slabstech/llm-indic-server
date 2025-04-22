import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import re

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
        self.processor = AutoProcessor.from_pretrained(model_id)
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
        
        Args:
            text (str): The model's output text.
        
        Returns:
            tuple: (function_name, args) if a valid call is found, else (None, None).
        """
        # Look for a code block
        code_match = re.search(r"```(.*?)```", text, re.DOTALL)
        if code_match:
            code_block = code_match.group(1).strip()
            # Check for function definitions
            lines = code_block.split('\n')
            for line in lines:
                if line.strip().startswith('def '):
                    # Contains function definition, treat as plain text
                    return None, None
            # No function definitions, look for a function call
            print_match = re.search(r"print\((.+)\)", code_block)
            if print_match:
                func_call = print_match.group(1).strip()
            else:
                func_call = code_block.strip()
            # Extract function name and arguments
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)", func_call)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                args = [arg.strip().strip("\"'") for arg in args_str.split(",")] if args_str else []
                if func_name in FUNCTION_REGISTRY:
                    return func_name, args
        # No code block or no registered function call found
        return None, None

    def _handle_function_call(self, func_name, args):
        """
        Execute the function call and return the result.
        
        Args:
            func_name (str): Name of the function to call.
            args (list): List of arguments for the function.
        
        Returns:
            str: The function result or an error message.
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
        """
        Send a message to the model and process its response.
        
        Args:
            user_message (str): The user's input message.
            max_new_tokens (int): Maximum tokens to generate.
        
        Returns:
            str: The assistant's response.
        """
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })

        # Generate model response
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

        # Try to extract and handle a function call
        func_name, args = self._extract_function_call(decoded)
        if func_name:
            func_result = self._handle_function_call(func_name, args)
            self.messages.append({
                "role": "function",
                "name": func_name,
                "content": [{"type": "text", "text": func_result}]
            })
            # Generate a follow-up response
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
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": followup_decoded}]
            })
            return f"{decoded}\n\n{func_result}\n\n{followup_decoded}"
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