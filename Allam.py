from EGRA_functions import EGRA

class Allam(EGRA):
    def __init__(self):
        super().__init__(model="humain-ai/ALLaM-7B-Instruct-preview")


    def zero_shot(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=False):
        return super().zero_shot(output_file, num_stories, max_new_tokens, do_sample, include_sys)

    
    def CoT_selfReflection(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=False):
        return super().CoT_selfReflection(output_file, num_stories, max_new_tokens, do_sample, include_sys)