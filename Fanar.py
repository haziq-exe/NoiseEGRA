from .EGRA_functions import EGRA

class Fanar(EGRA):
    def __init__(self):
        super().__init__(model="QCRI/Fanar-1-9B-Instruct")