from .EGRA_functions import EGRA

class AceGPT(EGRA):
    def __init__(self, use_AENI=False):
        super().__init__(model="FreedomIntelligence/AceGPT-v2-8B-Chat", use_AENI=use_AENI)


    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **tokenizer_kwargs):
        chat_text = ""
        for message in messages:
            if message["role"] == "system":
                chat_text += f"<System>:{message['content']} "
            elif message["role"] == "user":
                chat_text += f"<User>:{message['content']} "
            elif message["role"] == "assistant":
                chat_text += f"<Assistant>:{message['content']} "

        if add_generation_prompt:
            chat_text += "<Assistant>:"

        if not tokenize:
            return chat_text

        tokenizer_kwargs = dict(tokenizer_kwargs)
        tokenizer_kwargs.setdefault("add_special_tokens", False)
        encoded = self.tokenizer(chat_text, **tokenizer_kwargs)
        return encoded["input_ids"]