from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from datetime import datetime

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# 1. Define the function schema as per the documentation
function_definitions = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time in ISO format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# 2. Compose the prompt following Gemma's function calling structure
setup = (
    "You have access to functions. If you decide to invoke any of the function(s), "
    "you MUST put it in the format of {\"name\": function name, \"parameters\": dictionary of argument name and its value} "
    "You SHOULD NOT include any other text in the response if you call a function."
)

function_block = str(function_definitions)

user_query = "What is the current date and time?"

prompt = (
    f"{setup}\n"
    f"{function_block}\n"
    f"{user_query}"
)

# 3. Tokenize and prepare the input
inputs = processor(
    prompt, return_tensors="pt", return_dict=True
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# 4. Generate the model's response
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print("Model output:", decoded)

# 5. Detect and execute the function call if present
import json

try:
    response = json.loads(decoded)
    if (
        isinstance(response, dict)
        and response.get("name") == "get_current_datetime"
    ):
        # Call the function in Python
        result = datetime.now().isoformat()
        print("Function call detected: get_current_datetime()")
        print("Current date and time:", result)
    else:
        print("No function call detected. Model response:", decoded)
except Exception as e:
    print("Could not parse model output as JSON. Model response:", decoded)
