from ..EGRA_functions import EGRA
import torch


class Jais(EGRA):
    """Jais-2-8B-Chat wrapper. Uses bfloat16 on CUDA when supported, else float16."""

    def __init__(self):
        print("May need to upgrade transformers library to latest for this to run")
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        super().__init__(model="inceptionai/Jais-2-8B-Chat", dtype=dtype)
