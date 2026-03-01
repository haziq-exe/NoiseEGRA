from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
from pathlib import Path
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Callable, Dict, List, Optional, Sequence
from . import prompts
import math


class EGRA:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def _get_transformer_blocks(self):
        candidate_paths = [
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("transformer", "blocks"),
            ("gpt_neox", "layers"),
            ("decoder", "layers"),
        ]

        for path in candidate_paths:
            current = self.model
            for attr in path:
                if not hasattr(current, attr):
                    current = None
                    break
                current = getattr(current, attr)

            if isinstance(current, torch.nn.ModuleList) and len(current) > 0:
                return current
            if isinstance(current, (list, tuple)) and len(current) > 0 and all(isinstance(m, torch.nn.Module) for m in current):
                return current

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                if any(tag in name for tag in ("layers", "decoder.layers", "transformer.h", "gpt_neox.layers", "blocks")):
                    return module

        raise ValueError("Could not locate transformer blocks for this model architecture.")

    def _normalize_layer_index(self, layer_idx, total_layers):
        if not isinstance(layer_idx, int):
            raise TypeError("layer index must be an int.")

        if layer_idx < 0:
            layer_idx += total_layers

        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(f"layer index {layer_idx} is out of range for {total_layers} layers.")

        return layer_idx

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

        for x in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed+(128*x))
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

        for x in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed+(128*x))
            if print_output:
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

        for x in range(num_stories):
            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed+(128*x))
            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role" : "assistant", "content" : output})
            prompt.append({"role" : "user", "content" : prompts.NOISE_2})

            output = self.generate(prompt, max_new_tokens, do_sample, temperature=temperature, seed=seed+(128*x))

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])
    

    def generate_with_embedding_noise(
        self, prompt, embed_noise_std, hidden_noise_std, hidden_noise_decay, hidden_layers, logits_noise_std = 0.0, logits_noise_decay = 0.0,
        max_new_tokens = 100, temperature = 1.0,seed = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
        
        chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = next(iter(self.model.hf_device_map.values()))
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
    
        
        handles = []
        
        try:
            
            if embed_noise_std and embed_noise_std > 0:
                embed_layer = self.model.get_input_embeddings()
                noise_applied = {"done": False}
                
                def embedding_hook(module, input, output):
                    if not noise_applied["done"]:
                        # adding only on first forward pass to the prompt
                        with torch.no_grad():
                            noise = torch.randn_like(output) * embed_noise_std
                            output.add_(noise)
                        noise_applied["done"] = True
                    return output
                
                handle = embed_layer.register_forward_hook(embedding_hook)
                handles.append(handle)
    

            if hidden_noise_std and hidden_noise_std > 0 and hidden_layers:
                named = list(self.model.named_modules())
                for idx in hidden_layers:
                    str_idx = str(idx)
                    candidates = [(name, module) for name, module in named if name and str_idx in name.split('.')]
                    if not candidates:
                        continue
                    chosen_name, chosen_module = max(candidates, key=lambda nm: len(nm[0]))
    
                    def make_hook(std, decay):
                        state = {"step": 0}
                    
                        def hook(module, input, output):
                            with torch.no_grad():
                                if isinstance(output, torch.Tensor):
                                    target = output
                                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                                    target = output[0]
                                else:
                                    return None
                    
                                if target.dim() != 3:
                                    return None
                    
                                cur_std = std * (decay ** state["step"])
                                state["step"] += 1
                    
                                if cur_std <= 0:
                                    return None
                    
                                noise = torch.randn_like(target[:, -1:, :]) * cur_std
                                target[:, -1:, :].add_(noise)
                    
                            return None
                    
                        return hook
    
                    handle = chosen_module.register_forward_hook(make_hook(hidden_noise_std, hidden_noise_decay))
                    handles.append(handle)
    
            
            logits_processor = None
            if logits_noise_std and logits_noise_std > 0:
                processor = GaussianLogitsProcessor(
                    sigma=logits_noise_std,
                    decay=logits_noise_decay,
                    prompt_length=prompt_len,
                )
                logits_processor = LogitsProcessorList([processor])
    
            
            gen_kwargs = {
                **inputs,
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
            
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
    
            outputs = self.model.generate(**gen_kwargs)
            
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
    
        
        generated_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_with_residual_stream_noise(self, prompt, residual_layers, residual_noise_std, 
            residual_noise_decay=1.0, max_noise_tokens=250,
            logits_noise_std=0.0, logits_noise_decay=0.0, max_new_tokens=100, temperature=1.0, seed=None,
        ):
        """
        Injects Gaussian noise into the residual stream (block output) at selected transformer layers.
        Noise is applied on each generated token and decays over decoding steps.
        """
    
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
        if residual_noise_std <= 0:
            raise ValueError("residual_noise_std must be > 0.")
        if residual_noise_decay < 0:
            raise ValueError("residual_noise_decay must be >= 0.")
        if not isinstance(residual_layers, (list, tuple)) or len(residual_layers) == 0:
            raise ValueError("residual_layers must be a non-empty list/tuple of layer indices.")
    
        chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = next(iter(self.model.hf_device_map.values()))
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
    
        blocks = self._get_transformer_blocks()
        normalized_layers = sorted({self._normalize_layer_index(idx, len(blocks)) for idx in residual_layers})
    
        handles = []
        model_handle = None
    
        # Shared state across ALL residual hooks + model forward calls
        shared = {
            "forward_calls": 0,      # counts model forward invocations during generate()
            "t": 0,                  # decoding step index (0 for first generated token)
            "cur_t": 0,              # step index for *this* forward call
            "is_prefill": True,      # first forward is prompt prefill
        }
    
        try:
            # Advance step counter exactly ONCE per forward call (not per layer)
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    # Prefill pass
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    # Decode pass: set cur_t for this pass, then increment t for next pass
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1
    
            model_handle = self.model.register_forward_pre_hook(model_pre_hook)
    
            def residual_hook(module, input, output):
                # Skip prompt prefill for ALL layers
                if shared["is_prefill"]:
                    return None
    
                with torch.no_grad():
                    if isinstance(output, torch.Tensor):
                        target = output
                    elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                        target = output[0]
                    else:
                        return None
    
                    if target.dim() != 3:
                        return None
    
                    t = shared["cur_t"]
    
                    if max_noise_tokens:
                        T = max_noise_tokens
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
                        cur_std = residual_noise_std * cosine_decay
                    else:
                        cur_std = residual_noise_std * (residual_noise_decay ** t)
    
                    if cur_std <= 0:
                        return None
    
                    noise = torch.randn_like(target[:, -1:, :]) * cur_std
                    target[:, -1:, :].add_(noise)
    
                return None
    
            for layer_idx in normalized_layers:
                handles.append(blocks[layer_idx].register_forward_hook(residual_hook))
    
            logits_processor = None
            if logits_noise_std and logits_noise_std > 0:
                processor = GaussianLogitsProcessor(
                    sigma=logits_noise_std,
                    decay=logits_noise_decay,
                    prompt_length=prompt_len,
                )
                logits_processor = LogitsProcessorList([processor])
    
            gen_kwargs = {
                **inputs,
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
    
            outputs = self.model.generate(**gen_kwargs)
    
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            if model_handle is not None:
                try:
                    model_handle.remove()
                except Exception:
                    pass
    
        generated_ids = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def twoStage_residual_noise(
        self, residual_noise_std, residual_noise_decay, residual_layers, logits_noise_std = 0.0, logits_noise_decay = 0.0,
        output_file="example_file.csv", num_stories=1, max_new_tokens=100, include_sys=True, temperature=1.0, seed=None, print_output=False,
    ):
        output_csv = Path(output_file)

        if not include_sys:
            prompt = [{"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}]
        else:
            prompt = [{"role": "system", "content": prompts.SYS_NOISE}]
            prompt.append({"role": "user", "content": prompts.NOISE_1})

        for x in range(num_stories):
            output = self.generate_with_residual_stream_noise(
                prompt,
                residual_layers=residual_layers,
                residual_noise_std=residual_noise_std,
                residual_noise_decay=residual_noise_decay,
                logits_noise_std=logits_noise_std,
                logits_noise_decay=logits_noise_decay,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=(seed + (128 * x)) if seed is not None else None,
            )

            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role": "assistant", "content": output})
            prompt.append({"role": "user", "content": prompts.NOISE_2})

            output = self.generate(prompt, max_new_tokens, temperature=temperature, seed=(seed + (128 * x)) if seed is not None else None)

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])
                

    def twoStage_embedding_noise(
        self, embed_noise_std, hidden_noise_std, hidden_layers, logits_noise_std = 0.0, logits_noise_decay = 0.0,
        output_file="example_file.csv", num_stories=1, max_new_tokens=100, include_sys=True, temperature=1.0, seed=None, print_output=False,
    ):
        output_csv = Path(output_file)

        if not include_sys:
            prompt = [{"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}]
        else:
            prompt = [{"role": "system", "content": prompts.SYS_NOISE}]
            prompt.append({"role": "user", "content": prompts.NOISE_1})

        for x in range(num_stories):
            output = self.generate_with_embedding_noise(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=(seed + (128 * x)) if seed is not None else None,
                embed_noise_std=embed_noise_std,
                logits_noise_std=logits_noise_std,
                logits_noise_decay=logits_noise_decay,
                hidden_noise_std=hidden_noise_std,
                hidden_layers=hidden_layers,
            )

            if print_output:
                print("----- FIRST STAGE OUTPUT-----\n")
                print(output)

            prompt.append({"role": "assistant", "content": output})
            prompt.append({"role": "user", "content": prompts.NOISE_2})

            output = self.generate(prompt, max_new_tokens, temperature=temperature, seed=(seed + (128 * x)) if seed is not None else None)

            if print_output:
                print("----- SECOND STAGE OUTPUT-----\n")
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
        
        if std <= 0 or self.sigma <= 0:
            return scores
            
        noise = torch.randn_like(scores) * std
        return scores + noise
