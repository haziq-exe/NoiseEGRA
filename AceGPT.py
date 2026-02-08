from random import seed
from .EGRA_functions import EGRA, GaussianLogitsProcessor
from transformers import LogitsProcessor, LogitsProcessorList
import torch

class AceGPT(EGRA):
    def __init__(self):
        super().__init__(model="FreedomIntelligence/AceGPT-v2-8B-Chat")


    def apply_chat_template(self, messages, add_generation_prompt=False):
        chat_text = ""
        for message in messages:
            if message["role"] == "system":
                chat_text += f"<system>{message['content']}\n"
            elif message["role"] == "user":
                chat_text += f"<user>{message['content']}\n"
            elif message["role"] == "assistant":
                chat_text += f"<assistant>{message['content']}\n"

        if add_generation_prompt:
            chat_text += "<assistant>"
        
        return chat_text
    
    def generate(self, prompt, max_new_tokens=100, do_sample=True, temperature=1, seed=None):
        """
        For some reason AceGPT doesn't have a built in apply_chat_template function so need to implement custom one.
        """
        if seed is not None:
          torch.manual_seed(seed)

        chat_text = self.apply_chat_template(prompt, add_generation_prompt=True)
        device = next(iter(self.model.hf_device_map.values()))
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text
    
    def generate_with_embedding_noise(self, prompt, max_new_tokens = 100, temperature = 1.0, top_p = 0.9, seed = None,
                                    embed_noise_std = 0.01,logits_noise_std = 0.5, logits_noise_decay = 0.9):

        """
        Generates text by:
        1) Adding Gaussian noise to input embeddings
        2) Adding Gaussian noise to logits during sampling
        """

        if seed is not None:
            torch.manual_seed(seed)

        # Tokenize
        chat_text = self.apply_chat_template(prompt, add_generation_prompt=True)
        device = next(iter(self.model.hf_device_map.values()))
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Get embeddings
        with torch.no_grad():
            embed_layer = self.model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids)

            # Add Gaussian noise to embeddings
            if embed_noise_std > 0:
                noise = torch.randn_like(inputs_embeds) * embed_noise_std
                inputs_embeds = inputs_embeds + noise

        # Logits warper
        processor = GaussianLogitsProcessor(
            sigma=logits_noise_std,
            decay=logits_noise_decay,
            prompt_length=input_ids.shape[1],
        )
        logits_processor = LogitsProcessorList([processor])

        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)