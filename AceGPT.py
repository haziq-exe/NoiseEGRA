from .EGRA_functions import EGRA

class AceGPT(EGRA):
    def __init__(self):
        super().__init__(model="i/FreedomIntelligence/AceGPT-v2-70B-Chat")