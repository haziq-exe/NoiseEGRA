from .EGRA_functions import EGRA

class Jais(EGRA):
    def __init__(self):
        super().__init__(model="inceptionai/Jais-2-70B-Chat")


    def zero_shot(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=True):
        return super().zero_shot(output_file, num_stories, max_new_tokens, do_sample, include_sys)

    
    def CoT_selfReflection(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=True):
        return super().CoT_selfReflection(output_file, num_stories, max_new_tokens, do_sample, include_sys)