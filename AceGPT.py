from random import seed
from .EGRA_functions import EGRA
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