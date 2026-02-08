from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
from pathlib import Path
from transformers import LogitsProcessor, LogitsProcessorList
from . import prompts


class EGRA:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def generate(self, prompt, max_new_tokens=100, do_sample=True, temperature=1.0, seed=None):
        """
        prompt should always be a list of dicts of the form [ {"role" : "system", "content" : system_prompt},
                                              {"role" : "user", "content" : user_prompt}  ]
        """

        if seed is not None:
          torch.manual_seed(seed)

        chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = next(iter(self.model.hf_device_map.values()))
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text

    def zero_shot(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, do_sample=True, include_sys=True, temperature=1.0, seed=None, print_output=False):

        output_csv = Path(output_file)
        if not include_sys:
            prompt = [{"role" : "user" , "content" : prompts.SYS_ZERO_SHOT + "\n\n\n" + prompts.PROMPT_ZERO_SHOT}] 
        else:
            prompt = [{"role" : "system" , "content" : prompts.SYS_ZERO_SHOT}]
            prompt.append({"role" : "user" , "content" : prompts.PROMPT_ZERO_SHOT})

        for _ in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed)
            if print_output:
                print(output)
            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])
    
    def CoT_selfReflection(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, do_sample=True, include_sys=True, temperature=1.0, seed=None, print_output=False):
        prompt = []
        output_csv = Path(output_file)

        if include_sys:
            prompt.append({"role" : "system" , "content" : prompts.SYS_COT})

        prompt.append({"role" : "user", "content" : prompts.USER_COT_EXAMPLE})
        prompt.append({"role" : "assistant", "content" : prompts.ASSISTANT_COT_EXAMPLE})

        prompt.append({"role" : "user" , "content" : prompts.PROMPT_COT})

        for _ in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed)
            if print_output:
                print(output)
            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])

    def generate_with_embedding_noise(self, prompt, max_new_tokens = 100, temperature = 1.0, top_p = 0.9, seed = None,
                                      embed_noise_std = 0.01,logits_noise_std = 0.5, logits_noise_decay = 0.9):

      """
      Generates text by:
      1) Adding Gaussian noise to input embeddings
      2) Adding Gaussian noise to logits during sampling
      """

      if seed is not None:
          torch.manual_seed(seed)

      # Tokenize
      chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
      device = next(iter(self.model.hf_device_map.values()))
      inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
      input_ids = inputs["input_ids"]
      attention_mask = inputs.get("attention_mask", None)

      # Get embeddings
      with torch.no_grad():
          embed_layer = self.model.get_input_embeddings()
          inputs_embeds = embed_layer(input_ids)

          # Add Gaussian noise to embeddings
          if embed_noise_std > 0:
              noise = torch.randn_like(inputs_embeds) * embed_noise_std
              inputs_embeds = inputs_embeds + noise

      # Logits warper
      processor = GaussianLogitsProcessor(
          sigma=logits_noise_std,
          decay=logits_noise_decay,
          prompt_length=input_ids.shape[1],
      )
      logits_processor = LogitsProcessorList([processor])

      # Generate
      outputs = self.model.generate(
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
          do_sample=True,
          temperature=temperature,
          top_p=top_p,
          max_new_tokens=max_new_tokens,
          logits_processor=logits_processor,
          eos_token_id=self.tokenizer.eos_token_id,
      )

      return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

  
    def embedding_noise(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, include_sys=True, temperature=1.0,
                        embed_noise_std = 0.01,logits_noise_std = 0.5, logits_noise_decay = 0.9, seed=None, print_output=False, top_p=0.9):
      
      output_csv = Path(output_file)

      if not include_sys:
        prompt = [{"role" : "user" , "content" : prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}] 
      else:
        prompt = [{"role" : "system" , "content" : prompts.SYS_NOISE}]
        prompt.append({"role" : "user" , "content" : prompts.NOISE_1})

      for _ in range(num_stories):
        output = self.generate_with_embedding_noise(prompt, max_new_tokens, temperature=temperature, embed_noise_std=embed_noise_std,logits_noise_std=logits_noise_std, logits_noise_decay=logits_noise_decay, seed=seed, top_p=top_p)

        if print_output:
            print("----- FIRST STAGE OUTPUT-----\n")
            print(output)
    
        prompt.append({"role" : "assistant" , "content" : output})
        prompt.append({"role" : "user" , "content" : prompts.NOISE_2})

        output = self.generate(prompt, max_new_tokens, temperature=temperature, seed=seed)

        if print_output:
            print("----- SECOND STAGE OUTPUT-----\n")
            print(output)

        with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([output])
    
    def twoStage_zero_shot(self, output_file="example_file.csv", num_stories=1 ,max_new_tokens=100, do_sample=True, include_sys=True, temperature=1.0, seed=None, print_output=False):

        output_csv = Path(output_file)
        if not include_sys:
            prompt = [{"role" : "user" , "content" : prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}] 
        else:
            prompt = [{"role" : "system" , "content" : prompts.SYS_NOISE}]
            prompt.append({"role" : "user" , "content" : prompts.NOISE_1})

        for _ in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed)
            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role" : "assistant", "content" : output})
            prompt.append({"role" : "user", "content" : prompts.NOISE_2})

            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed)

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])



class GaussianLogitsProcessor(LogitsProcessor):
    """
    Adds zero-mean Gaussian noise to logits at each decoding step.
    Noise decays exponentially over time.
    """
    def __init__(self, sigma: float = 0.5, decay: float = 0.9, prompt_length: int = 0):
        self.sigma = sigma
        self.decay = decay
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[-1] - self.prompt_length
        if step <= 0:
            return scores

        std = self.sigma * (self.decay ** (step - 1))
        noise = torch.randn_like(scores) * std
        return scores + noise