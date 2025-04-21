import torch
from logging_config import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from fastapi import HTTPException
from io import BytesIO

class LLMManager:
    def __init__(self, model_name: str, device: str = "cuda"):
        if not torch.cuda.is_available():
            logger.error("CUDA is not available on this system.")
            raise RuntimeError("CUDA is required but not available.")
        
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16  # Use bfloat16 for CUDA
        self.model = None
        self.is_loaded = False
        self.processor = None
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            logger.info(f"GPU memory allocated after unload: {torch.cuda.memory_allocated()}")
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=self.torch_dtype
                ).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device} with bfloat16 precision")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
            logger.info(f"Decoded input: {self.processor.decode(inputs_vlm['input_ids'][0])}")
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in max 2 lines."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})

        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Vision query response: {decoded}")
        return decoded

    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})

        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat_v2 response: {decoded}")
        return decoded