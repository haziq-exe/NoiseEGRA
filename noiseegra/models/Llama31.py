from ..EGRA_functions import EGRA


class Llama31(EGRA):
    """Llama-3.1-8B-Instruct. Standard 32-block LlamaForCausalLM, float16-safe.

    Gated on Hugging Face: accept the licence on the model page and log in first.
    """

    def __init__(self, dtype=None, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__(model=model, dtype=dtype)
