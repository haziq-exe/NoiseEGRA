from .EGRA_functions import EGRA

class Jais(EGRA):
    def __init__(self):
        super().__init__(model="inceptionai/Jais-2-70B-Chat")