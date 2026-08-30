from ..EGRA_functions import EGRA
import torch


class Jais(EGRA):
    """Jais-2-8B-Chat wrapper.

    Jais must run in bfloat16 -- this is a property of the model, not a
    preference. Its residual stream carries activations about two orders of
    magnitude larger than the other models in this study (the RMS-calibrated
    noise std is 5.25 for Jais against 0.036 for ALLaM and 0.042 for Fanar).
    float16 tops out at 65504, so intermediate values overflow to inf/NaN.
    bfloat16 has float32's exponent range and does not.

    On pre-Ampere GPUs (T4 sm_75, P100 sm_60 -- both of Kaggle's options)
    bfloat16 has no hardware support and is emulated, which is slower. That is
    the correct trade here: float16 does not make Jais slower, it makes it
    wrong. Pass ``dtype=`` explicitly only if you know what you are doing.
    """

    def __init__(self, dtype=None):
        print("May need to upgrade transformers library to latest for this to run")
        if dtype is None:
            dtype = torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
                print(
                    "[Jais] Pre-Ampere GPU: bfloat16 is emulated here and generation "
                    "will be slower. Jais needs bfloat16 regardless -- float16 "
                    "overflows at this model's activation scale."
                )
        super().__init__(model="inceptionai/Jais-2-8B-Chat", dtype=dtype)
