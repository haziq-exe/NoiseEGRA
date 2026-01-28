from .EGRA_functions import EGRA

class Allam(EGRA):
    def __init__(self):
        super().__init__(model="humain-ai/ALLaM-7B-Instruct-preview")