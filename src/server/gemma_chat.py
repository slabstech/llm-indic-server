# chat_with_gemma.py

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

class LocalGemmaChat:
    """
    A class to replicate the basic chat interface of the Google GenAI SDK
    using Hugging Face Transformers and a local Gemma 3 model.
    """

    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.model_id = model_id
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Maintain conversation history as a list of message dicts
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }
        ]

    def send_message(self, user_message, max_new_tokens=100):
        """
        Mimics the .send_message() method of the GenAI SDK.
        Appends the user message to history, generates, and returns the assistant's reply.
        """
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })
        # Prepare input
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
        # Append assistant's reply to history
        self.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": decoded}]
        })
        return decoded

if __name__ == "__main__":
    # Example usage: simple chat loop
    chat = LocalGemmaChat(model_id="google/gemma-3-4b-it")
    print("Welcome to Local Gemma 3 Chat! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break
        response = chat.send_message(user_input)
        print(f"Gemma: {response}")
