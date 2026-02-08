from .EGRA_functions import EGRA
import torch

class Jais(EGRA):
    def __init__(self):
        print("May need to upgrade transformers library to latest for this to run")
        super().__init__(model="inceptionai/Jais-2-8B-Chat")
        self.model = self.model.to(torch.bfloat16)