import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from IndicTransToolkit import IndicProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device != "cpu" else torch.float32
model_name_llm = "Qwen/Qwen2.5-1.5B-Instruct"
# Model and tokenizer initialization
model = AutoModelForCausalLM.from_pretrained(
    model_name_llm,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_llm)
ip = IndicProcessor(inference=True)
src_lang, tgt_lang = "eng_Latn", "kan_Knda"
model_name_trans_indic_en = "ai4bharat/indictrans2-indic-en-dist-200M"
model_name_trans_en_indic = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer_trans_indic_en = AutoTokenizer.from_pretrained(model_name_trans_indic_en, trust_remote_code=True)
model_trans_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_indic_en,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
tokenizer_trans_en_indic = AutoTokenizer.from_pretrained(model_name_trans_en_indic, trust_remote_code=True)
model_trans_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_en_indic,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)


# Utility functions
def translate_text(text, src_lang, tgt_lang):
    if src_lang == "kan_Knda" and tgt_lang == "eng_Latn":
        tokenizer_trans = tokenizer_trans_indic_en
        model_trans = model_trans_indic_en
    elif src_lang == "eng_Latn" and tgt_lang == "kan_Knda":
        tokenizer_trans = tokenizer_trans_en_indic
        model_trans = model_trans_en_indic
    else:
        raise ValueError("Unsupported language pair")

    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer_trans(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated_tokens = model_trans.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer_trans.as_target_tokenizer():
        generated_tokens = tokenizer_trans.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return translations[0]

kannada_prompt = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"


translated_prompt = translate_text(kannada_prompt, src_lang="kan_Knda", tgt_lang="eng_Latn")
print(f"Translated prompt to English: {translated_prompt}")



messages = [
    {"role": "system", "content": "You are Dhwani, a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
    {"role": "user", "content": translated_prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Generated English response: {response}")



translated_response = translate_text(response, src_lang="eng_Latn", tgt_lang="kan_Knda")
print(f"Translated response to Kannada: {translated_response}")
