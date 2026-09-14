"""A 64-wide randomly initialised model, for tests that need a real forward pass.

Small enough to build in a second on CPU, with a tokeniser that needs no
download. The decoder returns a fixed compliant-looking sentence, so anything
that scores text sees something well formed.
"""

import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from noiseegra.EGRA_functions import EGRA


class Tok:
    pad_token_id = eos_token_id = bos_token_id = 1

    def _i(self, t):
        return [2 + (b % 250) for b in t.encode()][:400]

    def __call__(self, t, return_tensors=None, add_special_tokens=True, **k):
        ids = ([1] if add_special_tokens else []) + self._i(t)
        if return_tensors == "pt":
            return BatchEncoding({"input_ids": torch.tensor([ids]),
                                  "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        return {"input_ids": ids}

    def apply_chat_template(self, m, tokenize=False, add_generation_prompt=False, **k):
        t = "".join(f"<{x['role']}>{x['content']}" for x in m)
        return t + "<a>" if add_generation_prompt else t

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("word" for _ in ids) + '. "Hi," she says.'


class Tiny(EGRA):
    def __init__(self, **kw):
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
            pad_token_id=0, bos_token_id=1, eos_token_id=1)).eval()
        self.model.generation_config.pad_token_id = 0
        self.model.generation_config.eos_token_id = None
        self.tokenizer = Tok()
        self.device = "cpu"
