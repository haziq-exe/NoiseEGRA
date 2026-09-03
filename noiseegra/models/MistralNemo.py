from ..EGRA_functions import EGRA


class MistralNemo(EGRA):
    """Mistral-Nemo-Instruct-2407. 40 blocks, hidden 5120, ~12B, Apache 2.0.

    About 24 GB in float16, so it fits across two T4s with room for the KV cache
    but not much else. The largest of the English options here.
    """

    def __init__(self, dtype=None, model: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        super().__init__(model=model, dtype=dtype)
