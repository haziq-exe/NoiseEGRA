from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from pathlib import Path

# Need to update all the prompts to the actual prompts we will use


class EGRA:
    def __init__(self, model):
        self.model = AutoModelForCausalLM.from_pretrained(model) 
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def generate(self, prompt, max_new_tokens=100, do_sample=True):
        """
        prompt should always be a list of dicts of the form [ {"role" : "system", "content" : system_prompt},
                                              {"role" : "user", "content" : user_prompt}  ]
        """

        chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)

        return outputs

    def zero_shot(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, do_sample=True, include_sys=True):

        output_csv = Path(output_file)
        prompt = [{"role" : "user" , "content" : "Generate a short arabic story."}] 

        if include_sys: #Allam shouldn't have system prompt
            prompt.insert(0, [{"role" : "system" , "content" : "You are children's story generator in Arabic."}])

        for _ in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, include_sys)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(output)
    
    def CoT_selfReflection(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, do_sample=True, include_sys=True):
        prompt = []
        output_csv = Path(output_file)

        if include_sys:
            prompt.append({"role" : "system" , "content" : "You are children's story generator in Arabic."})

        prompt.append({"role" : "user", "content" : "Example CoT"})
        prompt.append({"role" : "assistant", "content" : "Example reasoning"})

        prompt.append([{"role" : "user" , "content" : "Generate a short arabic story."}])

        for _ in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, include_sys)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(output)