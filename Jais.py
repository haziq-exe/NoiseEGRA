from datasets import Datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from EGRA_functions import EGRA
import csv
from pathlib import Path

class Jais(EGRA):
    def __init__(self):
        super.__init__(model="inceptionai/Jais-2-70B-Chat")


    def zero_shot(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=True):
        
        output_csv = Path(output_file)

        for _ in range(num_stories):
            prompt = [{"role" : "user" , "content" : "Generate a short arabic story."}] 
            output = super().zero_shot(prompt, max_new_tokens, do_sample, include_sys)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(output)
    
    def CoT_selfReflection(self, output_file="example.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=True):
        
        output_csv = Path(output_file)

        for _ in range(num_stories):
            prompt = [{"role" : "user" , "content" : "Generate a short arabic story."}] 
            output = super().zero_shot(prompt, max_new_tokens, do_sample, include_sys)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(output)


        return super().CoT_selfReflection(prompt, max_new_tokens, do_sample)