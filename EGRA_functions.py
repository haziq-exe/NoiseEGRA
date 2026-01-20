from transformers import AutoModelForCausalLM, AutoTokenizer

class EGRA:
    def __init__(self, model):
        self.model = AutoModelForCausalLM.from_pretrained(model) 
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def zero_shot(self, prompt, max_new_tokens=100, do_sample=True):
        """
        prompt should always be a list of dicts of the form [ {"role" : "system", "content" : system_prompt},
                                              {"role" : "user", "content" : user_prompt}  ]
        """
        chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

        return outputs