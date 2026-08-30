from ..EGRA_functions import EGRA
import torch


def preferred_dtype():
    """bfloat16 only on Ampere (sm_80) and newer.

    ``torch.cuda.is_bf16_supported()`` reports True on older cards such as the
    T4 (sm_75) because recent PyTorch counts software emulation, which is
    unusably slow. Compute capability is the reliable test.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


class Jais(EGRA):
    """Jais-2-8B-Chat wrapper. Uses bfloat16 on Ampere+ GPUs, else float16."""

    def __init__(self, dtype=None):
        print("May need to upgrade transformers library to latest for this to run")
        super().__init__(model="inceptionai/Jais-2-8B-Chat", dtype=dtype or preferred_dtype())
