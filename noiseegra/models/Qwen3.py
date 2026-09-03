from ..EGRA_functions import EGRA


class Qwen3(EGRA):
    """Qwen3-8B-Instruct wrapper.

    Standard dense ``Qwen3ForCausalLM``: 36 blocks, hidden size 4096, text-only,
    Apache 2.0. float16 is safe on this model, so it runs natively on a T4 (no
    bfloat16 emulation, unlike Jais).

    Qwen3 has a reasoning mode that is on by default and emits a ``<think>``
    block before the answer. We disable it -- a visible chain of thought would
    contaminate the story text and every constraint measured on it.
    """

    def __init__(self, dtype=None, model: str = "Qwen/Qwen3-8B"):
        super().__init__(model=model, dtype=dtype)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
