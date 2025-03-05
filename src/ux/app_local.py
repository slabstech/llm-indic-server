# file: src/local/hf_qwen_2_5_1_5_b_gradio.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are Dhwani, built for Indian languages. You are a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

'''
# Create the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.outputs.Textbox(label="Response"),
    title="Dhwani AI- Text Query",
    description="Enter a prompt and get a response from Dhwani AI."
)

# Launch the Gradio interface
iface.launch()

'''
# Create the Gradio interface
with gr.Blocks(title="Indian language Query") as demo:
    gr.Markdown("# Dhwani AI- Text Query / ಧ್ವಾನಿ AI- ಪಠ್ಯ ಪ್ರಶ್ನೆ")
   

    #gr.Markdown("Enter your query and get a response Dhwani AI")
    #gr.Markdown("ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ ಮತ್ತು ಧ್ವಾನಿ AI ನಿಂದ ಉತ್ತರವನ್ನು ಪಡೆಯಿರಿ")

    query_input = gr.Textbox(label="Enter your text query / ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ", lines=2, placeholder="ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು ?")
    submit_button = gr.Button("Submit")
    dhwani_output = gr.Textbox(label="Answer / ಉತ್ತರ", interactive=False)

    submit_button.click(
        fn=generate_response,
        inputs=query_input,
        outputs=dhwani_output
    )

# Launch the interface with share=True
demo.launch(share=False)