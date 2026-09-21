from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch
import csv
from pathlib import Path
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Callable, Dict, List, Optional, Sequence
from . import prompts
from .decoders import truncation_warper
import math


def _cosine_noise_decay(t: int, max_noise_tokens: int) -> float:
    if max_noise_tokens <= 0:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * min(t, max_noise_tokens) / max_noise_tokens))



_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_OPEN_THINK = re.compile(r"^\s*<think>.*$", re.S)


def strip_reasoning(text: str) -> str:
    """Remove a reasoning block from generated text.

    Qwen3 and other reasoning models emit ``<think> ... </think>`` before the
    answer and their chat templates leave that on by default. Left in, it is
    scored as if it were the story: the word count, the sentence count and the
    readability grade all describe the model's deliberation rather than what it
    wrote. An unclosed block means the model spent the whole budget thinking and
    never produced a story, which is a failed generation and is returned empty
    rather than as a page of reasoning.
    """
    if "<think>" not in text:
        return text
    out = _THINK.sub("", text)
    return "" if _OPEN_THINK.match(out) else out.strip()


class EGRA:
    def __init__(self, model, use_AENI=False, dtype=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # float16 on a CPU-only machine is slow and some operators are not
        # implemented for it at all, so the default follows the device rather
        # than being a constant. A GPU-quota outage is a reason to run on CPU,
        # not a reason to run wrong.
        if dtype is not None:
            model_dtype = dtype
        elif torch.cuda.is_available():
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        if use_AENI:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=model_dtype, device_map="auto", attn_implementation="eager"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=model_dtype, device_map="auto"
            )
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

    def _input_device(self):
        """Device to place input ids on.

        Prefers the accelerate device map (set when the model is loaded with
        ``device_map=...``) and falls back to the first parameter's device, so
        models placed manually or run on CPU work too.
        """
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            return next(iter(device_map.values()))
        return next(self.model.parameters()).device

    def _sampling_kwargs(self, do_sample=True, temperature=1.0, top_p=None, top_k=None,
                         typical_p=None, min_p=None, eta_cutoff=None,
                         penalty_alpha=None):
        """Decoding settings, including the published truncation schemes.

        Beyond top-k (Fan et al., ACL 2018) and nucleus (Holtzman et al., ICLR
        2020), the comparisons a reviewer will expect are locally typical
        sampling (Meister et al., TACL 2023), eta-sampling (Hewitt et al., EMNLP
        Findings 2022), min-p (Nguyen et al., ICLR 2025) and contrastive search
        (Su et al., NeurIPS 2022).

        Top-k, nucleus and contrastive search are keyword arguments the
        generation stack still reads. The other three are not: a generation
        config accepts any attribute set on it, so a key the installed version
        has stopped looking at is stored, ignored, and never reported. Those
        three are therefore applied as an explicit processor
        (`noiseegra.decoders`), and this function leaves them out of the keyword
        arguments entirely. The caller then asks for temperature 1.0, because
        the processor applies the temperature itself -- see that module.

        Contrastive search is not sampling: `penalty_alpha` with `top_k` selects
        deterministically, penalising candidates similar to what has already been
        written. It still produces different stories here because each story is
        seeded differently and the prompt is perturbed, but it is the one arm
        whose diversity does not come from the sampling distribution.
        """
        kwargs = {
            "do_sample": do_sample,
        }
        if penalty_alpha is not None:
            # Contrastive search: deterministic, so `do_sample` must be off.
            kwargs["do_sample"] = False
            kwargs["penalty_alpha"] = penalty_alpha
            kwargs["top_k"] = int(top_k or 4)
            return kwargs
        if do_sample:
            # The processor scales by the temperature, so asking for it again
            # here would apply it twice.
            handled_here = (typical_p is None and min_p is None
                            and eta_cutoff is None)
            kwargs["temperature"] = temperature if handled_here else 1.0
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
        return kwargs

    # Reasoning models emit a <think> block before the answer. Qwen3's chat
    # template leaves that switched on by default, so the model may spend the whole
    # token budget reasoning and never reach the story -- which costs generation
    # time and puts the reasoning text into what is scored.
    enable_thinking = None       # None leaves the template's own default alone

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """
        Central chat-template entry point for all EGRA generation methods.
        Subclasses can override this to support tokenizers/models without a
        built-in chat template implementation.
        """
        kwargs = {}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = bool(self.enable_thinking)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        except TypeError:
            # The tokeniser's template does not take the argument; the model has
            # no reasoning mode to switch off and the default is what we want.
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

    # ---- runaway generations ------------------------------------------- #

    def _word_budget_stopper(self, prompt_len: int, max_words, every: int = 16):
        """Stop a generation once it has written far more than the task allows.

        A perturbation strong enough to buy diversity is often strong enough to
        stop the model terminating: the per-token-noise arm wrote 159 words and 78
        sentences against a 65-word, eight-sentence rule and ran to the token cap
        every single time, which made it about six times slower than every other
        condition for output that was already scored as failed. Waiting for the cap
        buys nothing -- a story twice over the word limit has broken the length
        rule and cannot un-break it by continuing.

        The text is decoded every ``every`` steps rather than every step, because
        decoding is the expensive part and a few tokens of overshoot do not matter.
        Returns None when there is no budget, so the normal path is untouched.
        """
        if not max_words or max_words <= 0:
            return None
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

        tokenizer = self.tokenizer

        class _WordBudget(StoppingCriteria):
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, scores, **kwargs):
                self.calls += 1
                if self.calls % every:
                    return False
                new = input_ids[0][prompt_len:]
                if new.numel() < max_words:      # cannot be over the budget yet
                    return False
                text = tokenizer.decode(new, skip_special_tokens=True)
                return len(text.split()) > max_words

        return StoppingCriteriaList([_WordBudget()])

    def generate(self, prompt, max_new_tokens=100, do_sample=True, temperature=1.0,
                 top_p=None, top_k=None, seed=None, max_words=None, entropy_out=None,
                 typical_p=None, min_p=None, eta_cutoff=None, penalty_alpha=None):
        """
        prompt should always be a list of dicts of the form [ {"role" : "system", "content" : system_prompt},
                                              {"role" : "user", "content" : user_prompt}  ]
        """

        if seed is not None:
          torch.manual_seed(seed)

        chat_text = self.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        stopper = self._word_budget_stopper(inputs["input_ids"].shape[-1], max_words)
        # See generate_with_orthogonal_steering: this reads the model's own
        # uncertainty off the raw scores, so raising the temperature leaves it
        # unchanged and the two kinds of intervention can be told apart.
        probe_kwargs = {}
        if entropy_out is not None:
            from .entropy_gate import EntropyProbe
            probe_state = {}
            probe_kwargs["logits_processor"] = LogitsProcessorList(
                [EntropyProbe(probe_state, keep_history=True)])
            entropy_out.append(probe_state)
        _kw = self._sampling_kwargs(
            do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
            typical_p=typical_p, min_p=min_p, eta_cutoff=eta_cutoff,
            penalty_alpha=penalty_alpha,
        )
        # Locally typical, eta and min-p are applied here rather than asked for
        # by name; `_sampling_kwargs` explains why, and leaves them out of the
        # keyword arguments so nothing is applied twice.
        truncator = truncation_warper(
            temperature=temperature, typical_p=typical_p, min_p=min_p,
            eta_cutoff=eta_cutoff) if do_sample and penalty_alpha is None else None
        if truncator is not None:
            existing = probe_kwargs.get("logits_processor") or LogitsProcessorList()
            probe_kwargs["logits_processor"] = LogitsProcessorList(
                list(existing) + [truncator])
        # Say in the log what decoding each condition is actually doing, once
        # per distinct setting. Six conditions differing only in these three
        # settings once came back byte-identical to plain nucleus sampling
        # across 200 stories each, and nothing in the log said so.
        #
        # Once per PROCESS is not enough: one process runs every condition of a
        # suite in turn, so the first arm would report and the seven arms whose
        # decoding is in question would not -- which is exactly the case this
        # line exists to cover.
        said = getattr(type(self), "_said_decoding", None)
        if not isinstance(said, set):
            said = set()
            type(self)._said_decoding = said
        applied = ("none" if truncator is None else
                   "typical_p=%s min_p=%s eta_cutoff=%s at temperature %s"
                   % (typical_p, min_p, eta_cutoff, temperature))
        key = (tuple(sorted(_kw.items())), applied)
        say = key not in said
        if say:
            said.add(key)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **probe_kwargs,
            **({"stopping_criteria": stopper} if stopper is not None else {}),
            **_kw,
        )

        # Reported AFTER generating, with the number of times the stack actually
        # invoked the processor. Saying which settings were passed is not
        # evidence that any of them were used: a plain object with the right
        # __call__ is accepted by one version of the generation stack and
        # dropped by the next, which is how five decoding conditions came back
        # byte-identical to plain nucleus sampling twice over. A count of zero
        # here is the fault, visible in the log, at the first story.
        if say:
            import transformers
            ran = "n/a" if truncator is None else str(truncator.calls)
            print(f"decoding: transformers {transformers.__version__}; "
                  f"passed to generate {_kw}; processor: {applied}; "
                  f"it ran {ran} times over {max_new_tokens} tokens",
                  flush=True)
            if truncator is not None and truncator.calls == 0:
                raise RuntimeError(
                    "the decoding setting was built and handed to generate, and "
                    "the generation stack never called it. Every story in this "
                    "condition would be plain sampling under a run id claiming "
                    f"otherwise. Settings: {applied}")
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))

        return strip_reasoning(text)

    def zero_shot(
        self,
        output_file="example_file.csv",
        num_stories=1,
        max_new_tokens=100,
        do_sample=True,
        include_sys=True,
        temperature=1.0,
        top_p=None,
        top_k=None,
        seed=None,
        print_output=False,
    ):

        output_csv = Path(output_file)
        if not include_sys:
            prompt = [{"role" : "user" , "content" : prompts.SYS_ZERO_SHOT + "\n\n\n" + prompts.PROMPT_ZERO_SHOT}] 
        else:
            prompt = [{"role" : "system" , "content" : prompts.SYS_ZERO_SHOT}]
            prompt.append({"role" : "user" , "content" : prompts.PROMPT_ZERO_SHOT})

        for x in range(num_stories):
            story_seed = (seed + (128 * x)) if seed is not None else None
            output = self.generate(
                prompt,
                max_new_tokens,
                do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )
            if print_output:
                print(output)
            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])
    
    def CoT_selfReflection(
        self,
        output_file="example_file.csv",
        num_stories=1,
        max_new_tokens=100,
        do_sample=True,
        include_sys=True,
        temperature=1.0,
        top_p=None,
        top_k=None,
        seed=None,
        print_output=False,
    ):
        prompt = []
        output_csv = Path(output_file)

        if include_sys:
            prompt.append({"role" : "system" , "content" : prompts.SYS_COT})

        prompt.append({"role" : "user", "content" : prompts.USER_COT_EXAMPLE})
        prompt.append({"role" : "assistant", "content" : prompts.ASSISTANT_COT_EXAMPLE})

        prompt.append({"role" : "user" , "content" : prompts.PROMPT_COT})

        for x in range(num_stories):
            story_seed = (seed + (128 * x)) if seed is not None else None
            output = self.generate(
                prompt,
                max_new_tokens,
                do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )
            if print_output:
                print(output)
            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])

    def twoStage_zero_shot(
        self,
        output_file="example_file.csv",
        num_stories=1,
        max_new_tokens=100,
        do_sample=True,
        include_sys=True,
        temperature=1.0,
        top_p=None,
        top_k=None,
        seed=None,
        print_output=False,
    ):

        output_csv = Path(output_file)

        for x in range(num_stories):
            if not include_sys:
                prompt = [{"role" : "user" , "content" : prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}]
            else:
                prompt = [{"role" : "system" , "content" : prompts.SYS_NOISE}]
                prompt.append({"role" : "user" , "content" : prompts.NOISE_1})

            story_seed = (seed + (128 * x)) if seed is not None else None
            output = self.generate(
                prompt,
                max_new_tokens,
                do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )
            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role" : "assistant", "content" : output})
            prompt.append({"role" : "user", "content" : prompts.NOISE_2})

            output = self.generate(
                prompt,
                max_new_tokens,
                do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])
    

    def generate_with_attention_output_noise(
        self, prompt, attn_noise_std, attn_layers,
        logits_noise_std=0.0, logits_noise_decay=0.0,
        max_new_tokens=100, do_sample=True, temperature=1.0, top_p=None, top_k=None, seed=None,
        max_noise_tokens=200,
    ):
        """
        Injects Gaussian noise into the self-attention output at selected layers,
        specifically after the output projection (o_proj) but before the MLP
        and before the residual addition. This targets the attention branch's
        contribution to the residual stream in isolation.

        Injection site within each transformer block:
            x = input_layernorm(hidden_states)
            attn_out, attn_weights, past_kv = self_attn(x)   # <-- noise added here
            hidden_states = hidden_states + attn_out          # residual add (unmodified)
            hidden_states = hidden_states + mlp(post_attention_layernorm(hidden_states))

        The self_attn forward hook receives output as a tuple:
            (attn_output, attn_weights, past_key_value)
        We perturb output[0] (shape: B x 1 x D during decoding) and return
        the full tuple with the modified tensor so the KV cache is preserved.

        Args:
            attn_noise_std:  Base noise standard deviation (before cosine decay).
            attn_layers:     List of layer indices at which to inject noise.
            max_noise_tokens: Decay horizon T; noise reaches zero at token T.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if attn_noise_std <= 0:
            raise ValueError("attn_noise_std must be > 0.")
        if not isinstance(attn_layers, (list, tuple)) or len(attn_layers) == 0:
            raise ValueError("attn_layers must be a non-empty list/tuple of layer indices.")

        chat_text = self.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        blocks = self._get_transformer_blocks()
        normalized_layers = sorted({
            self._normalize_layer_index(idx, len(blocks)) for idx in attn_layers
        })

        handles = []
        model_handle = None

        shared = {
            "forward_calls": 0,
            "t": 0,
            "cur_t": 0,
            "is_prefill": True,
        }

        try:
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1

            model_handle = self.model.register_forward_pre_hook(model_pre_hook)

            def make_attn_hook(std):
                def hook(module, input, output):
                    # Skip prompt prefill — only perturb during decoding
                    if shared["is_prefill"]:
                        return None

                    # self_attn returns a tuple: (attn_output, attn_weights, past_key_value)
                    # attn_output is the result of softmax(QK)V @ W_o, shape (B, T, D).
                    # We must return the full tuple to preserve the KV cache —
                    # returning only a tensor would silently drop past_key_value.
                    if not isinstance(output, (tuple, list)) or len(output) == 0:
                        return None

                    attn_output = output[0]

                    if not isinstance(attn_output, torch.Tensor) or attn_output.dim() != 3:
                        return None

                    with torch.no_grad():
                        t = shared["cur_t"]
                        T = max_noise_tokens
                        cosine_decay = _cosine_noise_decay(t, max_noise_tokens)
                        cur_std = std * cosine_decay

                        if cur_std <= 0:
                            return None

                        # During decoding, attn_output shape is (B, 1, D).
                        # Perturb only the last token position to match residual stream
                        # noise convention and avoid touching any cached positions.
                        noise = torch.randn_like(attn_output[:, -1:, :]) * cur_std
                        attn_output[:, -1:, :].add_(noise)

                    # Return the full tuple with the modified attn_output in position 0.
                    # output[1:] contains attn_weights and past_key_value — untouched.
                    return (attn_output,) + tuple(output[1:])

                return hook

            for layer_idx in normalized_layers:
                # Hook on block.self_attn, not on block itself.
                # This fires after o_proj inside self_attn has run, but before
                # the MLP and before the residual addition in the decoder layer.
                attn_module = blocks[layer_idx].self_attn
                handles.append(attn_module.register_forward_hook(make_attn_hook(attn_noise_std)))

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
                "max_new_tokens": max_new_tokens,
                **self._sampling_kwargs(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
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
        return strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))


    def generate_with_residual_stream_noise(self, prompt, residual_layers, residual_noise_std, 
            residual_noise_decay=1.0, max_noise_tokens=200,
            disable_residual_noise_decay=False,
            logits_noise_std=0.0, logits_noise_decay=0.0, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=None, top_k=None, seed=None,
        ):
        """
        Injects Gaussian noise into the residual stream (block output) at selected transformer layers.
        Noise is applied on each generated token.
        If disable_residual_noise_decay is True, noise std stays constant across decode steps.
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
    
        chat_text = self.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = self._input_device()
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
    
                    if disable_residual_noise_decay:
                        cur_std = residual_noise_std
                    elif max_noise_tokens:
                        T = max_noise_tokens
                        cosine_decay = _cosine_noise_decay(t, max_noise_tokens)
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
                "max_new_tokens": max_new_tokens,
                **self._sampling_kwargs(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
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
        return strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))

    def generate_with_orthogonal_steering(
        self, prompt, plan,
        max_new_tokens=500, do_sample=True, temperature=1.0, top_p=None, top_k=None, seed=None,
        story_index=None,
        max_words=None,
        entropy_out=None,
        typical_p=None, min_p=None, eta_cutoff=None,
    ):
        """
        Constraint steering with direction-constrained noise, injected at the same
        site as residual stream noise (block output, last token, decode steps).

        At every targeted layer and decode step the block output is perturbed by

            delta = sum_c beta_c * f_c(t) * rho * s_c   +   epsilon

        where ``s_c`` are the per-constraint unit steering directions (optionally
        orthogonalised against each other), ``rho`` is the model's median block RMS
        so that ``beta`` is on the same dimensionless scale as the paper's ``alpha``,
        ``f_c`` is a per-constraint schedule, and ``epsilon`` is Gaussian noise
        restricted to the orthogonal complement of the protected subspace
        (``plan.noise_mode == "orth"``), confined to it (``"para"``), or
        unrestricted (``"iso"``, i.e. the published L-Res method).

        Two caveats worth stating in any write-up. First, orthogonality holds
        **at the injection site, at that step** -- once the perturbed state is
        written to the KV cache and read by later layers the two components mix,
        so this is a per-site invariant, not a global one. Second, in a
        ``D``-dimensional stream a random draw already places only ``k/D`` of its
        energy inside a ``k``-dimensional subspace, so ``"orth"`` differs from
        ``"iso"`` only in proportion to the protected rank; enlarge it via
        ``SteeringVectorSet.protect_extra`` if you want the arms to separate.

        Args:
            plan: a :class:`noiseegra.subspace.SteeringPlan`. Build it with
                ``SteeringPlan.build(...)`` from an extracted
                :class:`noiseegra.steering_vectors.SteeringVectorSet`.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # A fresh constant offset per story: drawn after seeding, so it is
        # reproducible, and held fixed for the whole generation.
        # ``story_index`` matters when the plan has laid the whole set of
        # perturbations out in advance: it says which of them this story takes.
        # Without it the plan falls back to drawing one independently.
        if hasattr(plan, "resample_offset"):
            try:
                plan.resample_offset(story_index=story_index)
            except TypeError:
                plan.resample_offset()

        chat_text = self.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]

        # Size this story's perturbation by how far it moves the model's own
        # predictions, now that the prompt it will be measured on is known. The
        # direction was drawn at random above; only its length is set here.
        if (getattr(plan, "offset_norm", "") == "fisher"
                and any(lp.offset is not None for lp in plan.layer_plans.values())):
            from .fisher_calibration import calibrate_offset

            if getattr(plan, "_fisher_cache", None) is None:
                plan._fisher_cache = {}
            target = getattr(plan, "_gamma_this_story", None) or plan.offset_gamma
            got = calibrate_offset(
                self, plan, input_ids, float(target),
                max_length=float(plan.rms_scale * math.sqrt(plan.dim)),
                cache=plan._fisher_cache,
            )
            if getattr(plan, "fisher_log", None) is None:
                plan.fisher_log = []
            plan.fisher_log.append(got)
            print(f"  [fisher] length {got['length']:.2f} moves the predictions "
                  f"{got['distance']:.2f} nucleus-units (asked {float(target):.2f})"
                  + ("" if got["reached"] else "  -- NOT REACHED at the largest length"),
                  flush=True)

        # A shadow copy of the story: a second row with the same words and the
        # same steering but no perturbation, against which the story is held
        # level along the protected directions at every steered layer. Set up
        # after the sizing above, which measures on one row.
        shadow = bool(getattr(plan, "shadow_protect", False)) and (
            float(getattr(plan, "offset_gamma", 0.0) or 0.0) > 0
            or float(getattr(plan, "noise_alpha", 0.0) or 0.0) > 0)
        if shadow:
            if (getattr(plan, "steer_mode", "constant") == "feedback"
                    or float(getattr(plan, "amplify_lambda", 1.0) or 1.0) != 1.0):
                raise NotImplementedError(
                    "the shadow copy supports constant and error-driven steering; "
                    "feedback steering and amplification read each row's own "
                    "state and would give the shadow a push of its own")
            inputs = {k: v.repeat(2, 1) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]

        blocks = self._get_transformer_blocks()
        normalized_layers = sorted({
            self._normalize_layer_index(idx, len(blocks)) for idx in plan.layers
        })
        unknown = [li for li in normalized_layers if li not in plan.layer_plans]
        if unknown:
            raise KeyError(
                f"plan has no per-layer geometry for layers {unknown}; "
                f"available: {sorted(plan.layer_plans)}"
            )

        handles = []
        model_handle = None

        # Same decode-step bookkeeping as the other noise methods: one increment
        # per model forward call, not per layer.
        shared = {"forward_calls": 0, "t": 0, "cur_t": 0, "is_prefill": True}

        # Entropy gate. The probe records the entropy of each step's next-token
        # distribution *after* the forward pass the hooks live in, so at step t the
        # hooks read the value from step t-1: "was the model uncertain about the
        # token it just emitted". Starts at +inf so the first steps are never
        # blocked for want of a measurement.
        gate_threshold = float(getattr(plan, "gate_threshold", 0.0) or 0.0)
        entropy_state = {"entropy": float("inf"), "history": [], "open": 0, "steps": 0}
        probes = []
        if gate_threshold > 0:
            from .entropy_gate import EntropyProbe

            probes.append(EntropyProbe(entropy_state))
        # Error-driven steering needs to see the story so far. A logits processor
        # is the only place in `generate` that does; it runs after the forward
        # pass, so what it measures is read by the hooks on the next step.
        if getattr(plan, "steer_mode", "constant") == "error":
            from .constraint_control import ConstraintProbe

            if plan.control_state is None:
                plan.control_state = {}
            plan.control_state["errors"] = {}
            probes.append(ConstraintProbe(
                self.tokenizer, plan.control_state,
                plan.controller, inputs["input_ids"].shape[-1],
            ))
        processors = None
        if probes:
            # LogitsProcessorList comes from the module-level import. Importing
            # it here instead made it a function-local name, so every later
            # reference in this function was unbound whenever `probes` was empty
            # -- which is every arm that has no constraint probe. The entropy
            # recorder added below is exactly such a reference, and it took a
            # Kaggle run to find out.
            processors = LogitsProcessorList(probes)

        try:
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1

            model_handle = self.model.register_forward_pre_hook(model_pre_hook)

            def make_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        target = output
                    elif (
                        isinstance(output, (tuple, list))
                        and len(output) > 0
                        and isinstance(output[0], torch.Tensor)
                    ):
                        target = output[0]
                    else:
                        return None

                    if target.dim() != 3:
                        return None

                    with torch.no_grad():
                        if shared["is_prefill"]:
                            # Per-token noise never touches the prompt (paper
                            # convention): it is redrawn every step, and the prompt
                            # is read once. Steering optionally does, CAA-style,
                            # across all prompt positions -- and so does the
                            # per-story offset, which is one fixed vector for the
                            # whole generation and so has a well-defined value here.
                            # Adding it at prefill shifts how the model reads the
                            # instruction before it writes a token, which changes
                            # the story without perturbing any decode step.
                            offset_here = bool(getattr(plan, "offset_prefill", False))
                            amp_here = bool(getattr(plan, "amplify_prefill", False))
                            if not plan.steer_prefill and not offset_here and not amp_here:
                                return None
                            delta = None
                            if plan.steer_prefill:
                                delta = plan.steering_only(
                                    layer_idx, 0, device=target.device,
                                )
                            # The perturbation's own band, when it differs from
                            # the push's. Without this check here the band is
                            # honoured at decode steps and ignored at the prompt,
                            # and since every run perturbs the prompt and leaves
                            # decoding alone, a band sweep produced three
                            # byte-identical arms and read as a clean null.
                            bands = getattr(plan, "offset_layers", None) or ()
                            taper = float(getattr(plan, "offset_taper", 1.0) or 1.0)
                            off = None
                            if offset_here and (not bands or layer_idx in bands):
                                off = plan.layer_plans[layer_idx].offset
                                if off is not None:
                                    off = off.to(target.device) * float(
                                        getattr(plan, "offset_prefill_gain", 1.0) or 1.0)
                                    if taper >= 1.0:
                                        delta = off if delta is None else delta + off
                                        off = None      # folded in; applied flat
                            if delta is not None:
                                # Leave the final prompt positions alone when
                                # asked: they are the chat template's own
                                # tokens, marking that the instruction has ended
                                # and the answer starts here. See
                                # SteeringPlan.prompt_tail_clear.
                                keep = int(getattr(plan, "prompt_tail_clear", 0) or 0)
                                head = int(getattr(plan, "prompt_head_clear", 0) or 0)
                                d = delta.to(target.dtype).view(1, 1, -1)
                                n = target.shape[1]
                                lo = head if 0 < head < n else 0
                                hi = n - keep if 0 < keep < n - lo else n
                                if hi > lo:
                                    target[:, lo:hi, :].add_(d)
                                else:
                                    target.add_(d)
                            if off is not None:
                                # The perturbation faded along the prompt: full
                                # strength where the instruction begins, `taper`
                                # of it at the last position it touches.
                                #
                                # Sparing the last positions outright separates
                                # two things that were measured together. The
                                # perturbation near the end of the prompt is what
                                # varies the wording -- and what makes the model
                                # answer the reader instead of telling a story.
                                # The perturbation early in the prompt is what
                                # varies what happens, and is safe: at two and a
                                # half stories out with 48 positions spared there
                                # is not one non-story in 200, and variety of what
                                # happens is still won. Cutting it at a position
                                # loses the wording with the failure; fading it
                                # keeps some of both.
                                keep = int(getattr(plan, "prompt_tail_clear", 0) or 0)
                                head = int(getattr(plan, "prompt_head_clear", 0) or 0)
                                n = target.shape[1]
                                lo = head if 0 < head < n else 0
                                hi = n - keep if 0 < keep < n - lo else n
                                if hi > lo:
                                    ramp = torch.linspace(
                                        1.0, taper, hi - lo,
                                        device=target.device, dtype=target.dtype,
                                    ).view(1, -1, 1)
                                    target[:, lo:hi, :].add_(
                                        off.to(target.dtype).view(1, 1, -1) * ramp)
                            if plan.steer_prefill and getattr(plan, "steer_mode", "constant") == "feedback":
                                for pos in range(target.shape[1]):
                                    fb = plan.feedback_delta(
                                        layer_idx, target[0, pos, :].float(), 0,
                                        device=target.device,
                                    )
                                    if fb is not None:
                                        target[0, pos, :].add_(fb.to(target.dtype))
                            if amp_here:
                                # Per position: each prompt position has its own
                                # deviation from the average, so this is not one
                                # vector added everywhere.
                                amp = plan.amplify_delta(
                                    layer_idx, target[0].float(), device=target.device
                                )
                                if amp is not None:
                                    target[0].add_(amp.to(target.dtype))
                            return None

                        # Gate the perturbation, never the steering: a gate sweep
                        # has to change where the perturbation lands without
                        # changing the constraint pressure.
                        gate_open = (gate_threshold <= 0
                                     or entropy_state["entropy"] >= gate_threshold)
                        if layer_idx == normalized_layers[0]:
                            entropy_state["steps"] += 1
                            entropy_state["open"] += int(gate_open)

                        delta = plan.delta_for(
                            layer_idx, shared["cur_t"], with_noise=gate_open,
                            with_offset=gate_open, device=target.device,
                        )
                        # Feedback steering reads this story's own position on each
                        # constraint axis, so it cannot be precomputed the way a
                        # constant push can.
                        fb = plan.feedback_delta(
                            layer_idx, target[0, -1, :].float(), shared["cur_t"],
                            device=target.device,
                        )
                        if fb is not None:
                            delta = fb if delta is None else delta + fb
                        ed = plan.error_delta(layer_idx, shared["cur_t"],
                                              device=target.device)
                        if ed is not None:
                            delta = ed if delta is None else delta + ed
                        if delta is not None:
                            target[:, -1:, :].add_(delta.to(target.dtype).view(1, 1, -1))
                        if gate_open:
                            amp = plan.amplify_delta(
                                layer_idx, target[0, -1, :].float(),
                                device=target.device,
                            )
                            if amp is not None:
                                target[:, -1:, :].add_(amp.to(target.dtype).view(1, 1, -1))

                    return None

                if not shadow:
                    return hook

                def shadowed(module, input, output):
                    if isinstance(output, torch.Tensor):
                        target = output
                    elif (isinstance(output, (tuple, list)) and len(output) > 0
                          and isinstance(output[0], torch.Tensor)):
                        target = output[0]
                    else:
                        return hook(module, input, output)
                    if target.dim() != 3 or target.shape[0] != 2:
                        return hook(module, input, output)
                    prefill = bool(shared["is_prefill"])
                    t_now = shared["cur_t"]
                    with torch.no_grad():
                        before = target[1].clone()
                        hook(module, input, output)
                        # The shadow gets the steering and nothing else.
                        target[1].copy_(before)
                        if prefill:
                            if plan.steer_prefill:
                                d = plan.steering_only(layer_idx, 0, device=target.device)
                                if d is not None:
                                    keep = int(getattr(plan, "prompt_tail_clear", 0) or 0)
                                    head = int(getattr(plan, "prompt_head_clear", 0) or 0)
                                    n = target.shape[1]
                                    lo = head if 0 < head < n else 0
                                    hi = n - keep if 0 < keep < n - lo else n
                                    dd = d.to(target.dtype).view(1, -1)
                                    if hi > lo:
                                        target[1, lo:hi, :].add_(dd)
                                    else:
                                        target[1].add_(dd)
                            rows = slice(None)
                        else:
                            d = plan.delta_for(layer_idx, t_now, with_noise=False,
                                               with_offset=False, device=target.device)
                            if getattr(plan, "steer_mode", "constant") == "error":
                                ed = plan.error_delta(layer_idx, t_now, device=target.device)
                                if ed is not None:
                                    d = ed if d is None else d + ed
                            if d is not None:
                                target[1, -1:, :].add_(d.to(target.dtype).view(1, -1))
                            rows = slice(-1, None)
                        # Hold the story level with the shadow along the
                        # protected directions. The difference between the two
                        # rows is everything the perturbation has done so far,
                        # including what earlier layers carried back into these
                        # directions; only that part is removed.
                        prot = plan.layer_plans[layer_idx].protect
                        if prot is not None:
                            prot = prot.to(device=target.device, dtype=torch.float32)
                            diff = (target[0, rows, :] - target[1, rows, :]).float()
                            target[0, rows, :].sub_(((diff @ prot) @ prot.t()).to(target.dtype))
                    return None

                return shadowed

            for layer_idx in normalized_layers:
                handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

            # The push and a truncation scheme are different interventions --
            # one reshapes the representation, the other the distribution over
            # the next token -- so they compose. Every method arm until now used
            # the checkpoint's own cut-offs, which meant the comparison was
            # against decoders the method was never combined with.
            gen_kwargs = self._sampling_kwargs(
                do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
                typical_p=typical_p, min_p=min_p, eta_cutoff=eta_cutoff,
            )
            # Locally typical, eta and min-p are not read from the keyword
            # arguments by the installed stack; they are applied as a processor.
            truncator = truncation_warper(
                temperature=temperature, typical_p=typical_p, min_p=min_p,
                eta_cutoff=eta_cutoff) if do_sample else None
            if truncator is not None:
                processors = LogitsProcessorList([*(processors or []), truncator])
            # `entropy_out` records the model's own next-token uncertainty, read
            # off the raw scores before temperature or any cut-off is applied.
            # That is the point: raising the temperature does not change this
            # number at all, because it rescales the scores afterwards. So it
            # separates an intervention that makes the model less certain from
            # one that moves it somewhere else while leaving it just as certain.
            if entropy_out is not None:
                from .entropy_gate import EntropyProbe
                probe_state = {}
                probe = EntropyProbe(probe_state, keep_history=True)
                processors = (LogitsProcessorList([*(processors or []), probe])
                              if processors is not None
                              else LogitsProcessorList([probe]))
                entropy_out.append(probe_state)
            if processors is not None:
                gen_kwargs["logits_processor"] = processors
            stopper = self._word_budget_stopper(inputs["input_ids"].shape[-1], max_words)
            if stopper is not None:
                gen_kwargs["stopping_criteria"] = stopper
            if shadow:
                # The shadow must write the story's words, not its own. Each row
                # samples independently, so after every step the shadow's new
                # token is overwritten with the story's before the next forward
                # reads it. Done here, after sampling, so the story's own sampling
                # is exactly what it would be without a shadow. The shadow is
                # finished when the story is.
                from transformers import StoppingCriteria, StoppingCriteriaList
                eos = self.model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, (list, tuple))
                          else ([] if eos is None else [eos]))

                class _ShadowCopy(StoppingCriteria):
                    def __call__(self_, ids, scores, **kwargs):
                        ids[1:, -1] = ids[0, -1]
                        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
                        done[1:] = int(ids[0, -1]) in eos
                        return done

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [*(gen_kwargs.get("stopping_criteria") or []), _ShadowCopy()])
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **gen_kwargs
            )

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
        text = strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        if shadow:
            other = outputs[1][input_ids.shape[-1]:]
            drift = int((generated_ids != other).sum())
            plan.shadow_drift = int(getattr(plan, "shadow_drift", 0) or 0) + drift
            if drift:
                print(f"  [shadow] the shadow copy left the story at {drift} of "
                      f"{generated_ids.numel()} positions -- its protection was "
                      "measured against the wrong text", flush=True)
        if gate_threshold > 0 and entropy_state["steps"]:
            self.last_gate_rate = entropy_state["open"] / entropy_state["steps"]
        else:
            self.last_gate_rate = 1.0
        return text

    @torch.no_grad()
    def generate_with_entropy_noise(
        self, prompt, attention_noise_std, attn_entropy_layers,
        entropy_calc: str = "max_weight",
        top_k_size: int = 10,
        logits_noise_std=0.0, logits_noise_decay=0.0,
        max_new_tokens=100, temperature=1.0, seed=None,
        max_noise_tokens=200,
    ):

        """
        Attention Entropy-Based Noise Injection (AENI).

        At each decode step, measures the peakedness of the post-softmax attention
        distribution at each targeted layer using one of four methods (controlled
        by `entropy_calc`), then injects Gaussian noise into the self-attention
        output (post o_proj, pre-MLP, pre-residual addition) scaled by:

            cur_std = attention_noise_std * entropy_scale * cosine_decay

        where entropy_scale ∈ [0, 1] is HIGH when attention is peaked (low entropy)
        and LOW when attention is diffuse (high entropy). `attention_noise_std` acts
        as the ceiling — calibrated per-model via RMS as with other noise methods.

        entropy_calc options
        --------------------
        "max_weight":
            entropy_scale = mean_heads( max_position(w) )
            The mean per-head maximum attention weight. Peaked → high max → more noise.
            Naturally in [0, 1]. Zero context-length dependency. No normalisation needed.
            Recommended as the simplest and most robust option.


        Args:
            attention_noise_std:  Noise std ceiling (scaled by entropy_scale at each step).
            attn_entropy_layers:  List of layer indices to target.
            entropy_calc:         One of "max_weight", "topk_entropy", "gini", "renyi2".
            top_k_size:           k for "topk_entropy" method (default 10).
            max_noise_tokens:     Cosine decay horizon T.
        """
        VALID_METHODS = ("max_weight", "topk_entropy", "gini", "renyi2")
        if entropy_calc not in VALID_METHODS:
            raise ValueError(
                f"entropy_calc must be one of {VALID_METHODS}, got '{entropy_calc}'."
            )

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        chat_text = self.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        blocks = self._get_transformer_blocks()
        normalized_layers = sorted({
            self._normalize_layer_index(idx, len(blocks)) for idx in attn_entropy_layers
        })

        handles = []
        model_handle = None

        shared = {
            "forward_calls": 0,
            "t": 0,
            "cur_t": 0,
            "is_prefill": True,
        }

        try:
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1

            model_handle = self.model.register_forward_pre_hook(model_pre_hook)

            def _compute_entropy_scale(w: torch.Tensor) -> float:
                """
                w: (B, num_heads, T_k) — float32, clamped ≥ 0, current token only.
                Returns entropy_scale in [0, 1]:
                    HIGH  = peaked attention  = more noise
                    LOW   = diffuse attention = less noise
                """

                # Mean of per-head maximum weight.
                # Peaked head → max near 1 → high scale.
                # Uniform → max near 1/T_k → low scale.
                # No normalisation, no context-length dependency.
                scale = w.max(dim=-1).values.mean().item()

                return float(scale)

            def make_attn_entropy_hook(layer_idx: int):
                def hook(module, input, output):
                    if shared["is_prefill"]:
                        return None

                    # self_attn returns: (attn_output, attn_weights, past_key_value)
                    if not isinstance(output, (tuple, list)) or len(output) < 2:
                        print(
                            f"[AENI] Layer {layer_idx}: unexpected self_attn output "
                            f"format — skipping noise injection."
                        )
                        return None

                    attn_output = output[0]
                    attn_weights = output[1]

                    if not isinstance(attn_output, torch.Tensor) or attn_output.dim() != 3:
                        return None

                    if attn_weights is None or not isinstance(attn_weights, torch.Tensor):
                        print(
                            f"[AENI] Layer {layer_idx}: attn_weights is None — "
                            f"eager mode not active, skipping noise injection."
                        )
                        return None

                    if attn_weights.dim() != 4:
                        print(
                            f"[AENI] Layer {layer_idx}: unexpected attn_weights shape "
                            f"{attn_weights.shape} — skipping noise injection."
                        )
                        return None

                    with torch.no_grad():
                        # Last query row only — what the current token attends to.
                        # Clamp + cast: guards against fp16/bf16 underflow producing
                        # tiny negative values after softmax (causes nan in log).
                        w = attn_weights[:, :, -1, :].float().clamp(min=0.0)  # (B, heads, T_k)

                        entropy_scale = _compute_entropy_scale(w)

                        t = shared["cur_t"]
                        T = max_noise_tokens
                        cosine_decay = _cosine_noise_decay(t, max_noise_tokens)

                        cur_std = attention_noise_std * entropy_scale  * cosine_decay

                        if cur_std <= 0:
                            return None

                        noise = torch.randn_like(attn_output[:, -1:, :]) * cur_std
                        attn_output[:, -1:, :].add_(noise)

                    # Return full tuple: modified attn_output at [0],
                    # attn_weights and past_key_value untouched at [1:].
                    return (attn_output,) + tuple(output[1:])

                return hook

            for layer_idx in normalized_layers:
                handles.append(
                    blocks[layer_idx].self_attn.register_forward_hook(
                        make_attn_entropy_hook(layer_idx)
                    )
                )

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
                "output_attentions": True,
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
        return strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))


    @torch.no_grad()
    def generate_with_residual_and_entropy_noise(
        self,
        prompt,
        residual_layers,
        residual_noise_std,
        attn_entropy_layers,
        attention_noise_std,
        entropy_calc: str = "max_weight",
        residual_noise_decay=0.0,
        max_noise_tokens=200,
        logits_noise_std=0.0,
        logits_noise_decay=0.0,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_p=None,
        top_k=None,
        seed=None,
        top_k_size=None,
        disable_residual_noise_decay=False,
    ):
        """
        Combined residual-stream and entropy-based attention noise generation.

        During decode steps only (prefill skipped):
            1) Add entropy-scaled noise to self-attention output at
               `attn_entropy_layers`.
            2) Add residual-stream noise to block output at `residual_layers`.

        Both schedules share the same decode-step counter and cosine decay
        horizon (`max_noise_tokens`).
        """
        VALID_METHODS = ("max_weight", "topk_entropy", "gini", "renyi2")
        if entropy_calc not in VALID_METHODS:
            raise ValueError(
                f"entropy_calc must be one of {VALID_METHODS}, got '{entropy_calc}'."
            )

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if residual_noise_std <= 0:
            raise ValueError("residual_noise_std must be > 0.")
        if attention_noise_std <= 0:
            raise ValueError("attention_noise_std must be > 0.")
        if residual_noise_decay < 0:
            raise ValueError("residual_noise_decay must be >= 0.")
        if not isinstance(residual_layers, (list, tuple)) or len(residual_layers) == 0:
            raise ValueError("residual_layers must be a non-empty list/tuple of layer indices.")
        if not isinstance(attn_entropy_layers, (list, tuple)) or len(attn_entropy_layers) == 0:
            raise ValueError("attn_entropy_layers must be a non-empty list/tuple of layer indices.")

        chat_text = self.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        blocks = self._get_transformer_blocks()
        normalized_residual_layers = sorted({
            self._normalize_layer_index(idx, len(blocks)) for idx in residual_layers
        })
        normalized_attn_layers = sorted({
            self._normalize_layer_index(idx, len(blocks)) for idx in attn_entropy_layers
        })

        handles = []
        model_handle = None

        shared = {
            "forward_calls": 0,
            "t": 0,
            "cur_t": 0,
            "is_prefill": True,
        }

        try:
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1

            model_handle = self.model.register_forward_pre_hook(model_pre_hook)

            def _compute_entropy_scale(w: torch.Tensor) -> float:
                # Mean per-head max attention weight.
                scale = w.max(dim=-1).values.mean().item()
                return float(scale)

            def attn_entropy_hook(module, input, output):
                if shared["is_prefill"]:
                    return None

                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    return None

                attn_output = output[0]
                attn_weights = output[1]

                if not isinstance(attn_output, torch.Tensor) or attn_output.dim() != 3:
                    return None

                if attn_weights is None or not isinstance(attn_weights, torch.Tensor):
                    return None

                if attn_weights.dim() != 4:
                    return None

                with torch.no_grad():
                    w = attn_weights[:, :, -1, :].float().clamp(min=0.0)
                    entropy_scale = _compute_entropy_scale(w)

                    t = shared["cur_t"]
                    if max_noise_tokens:
                        T = max_noise_tokens
                        cosine_decay = _cosine_noise_decay(t, max_noise_tokens)
                        cur_std = attention_noise_std * entropy_scale * cosine_decay
                    else:
                        cur_std = attention_noise_std * entropy_scale * (residual_noise_decay ** t)

                    if cur_std <= 0:
                        return None

                    noise = torch.randn_like(attn_output[:, -1:, :]) * cur_std
                    attn_output[:, -1:, :].add_(noise)

                return (attn_output,) + tuple(output[1:])

            def residual_hook(module, input, output):
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
                    if disable_residual_noise_decay:
                        cur_std = residual_noise_std
                    elif max_noise_tokens:
                        T = max_noise_tokens
                        cosine_decay = _cosine_noise_decay(t, max_noise_tokens)
                        cur_std = residual_noise_std * cosine_decay
                    else:
                        cur_std = residual_noise_std * (residual_noise_decay ** t)

                    if cur_std <= 0:
                        return None

                    noise = torch.randn_like(target[:, -1:, :]) * cur_std
                    target[:, -1:, :].add_(noise)

                return None

            for layer_idx in normalized_attn_layers:
                handles.append(
                    blocks[layer_idx].self_attn.register_forward_hook(attn_entropy_hook)
                )

            for layer_idx in normalized_residual_layers:
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
                "max_new_tokens": max_new_tokens,
                "output_attentions": True,
                **self._sampling_kwargs(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
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
        return strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))


    def generate_with_embedding_noise(
        self, prompt, embed_noise_std,
        logits_noise_std=0.0, logits_noise_decay=0.0,
        max_new_tokens=100, temperature=1.0, seed=None,
        max_noise_tokens=200,
    ):
        """
        Injects Gaussian noise into the token embedding lookup table output
        during decoding. This is the only embedding-level injection site that
        has no residual-stream equivalent — all hidden-layer embedding inputs
        are equivalent to the previous block's residual output.

        Injection site:
            model.get_input_embeddings()  (nn.Embedding)
            Output shape during decode: (B, 1, D)

        Noise is added only during decode steps (prefill is skipped), with the
        same cosine decay schedule used by all other noise methods.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if embed_noise_std <= 0:
            raise ValueError("embed_noise_std must be > 0.")

        chat_text = self.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        device = self._input_device()
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        handles = []
        model_handle = None

        shared = {
            "forward_calls": 0,
            "t": 0,
            "cur_t": 0,
            "is_prefill": True,
        }

        try:
            def model_pre_hook(module, inp):
                shared["forward_calls"] += 1
                if shared["forward_calls"] == 1:
                    shared["is_prefill"] = True
                    shared["cur_t"] = 0
                else:
                    shared["is_prefill"] = False
                    shared["cur_t"] = shared["t"]
                    shared["t"] += 1

            model_handle = self.model.register_forward_pre_hook(model_pre_hook)

            embed_layer = self.model.get_input_embeddings()

            def embed_hook(module, input, output):
                if shared["is_prefill"]:
                    return None

                if not isinstance(output, torch.Tensor) or output.dim() != 3:
                    return None

                with torch.no_grad():
                    t = shared["cur_t"]
                    cosine_decay = _cosine_noise_decay(t, max_noise_tokens)
                    cur_std = embed_noise_std * cosine_decay

                    if cur_std <= 0:
                        return None

                    noise = torch.randn_like(output) * cur_std
                    output.add_(noise)

                return output

            handles.append(embed_layer.register_forward_hook(embed_hook))

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
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
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
        return strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))

    def twoStage_residual_noise(
        self, residual_noise_std, residual_noise_decay, residual_layers, logits_noise_std = 0.0, logits_noise_decay = 0.0,
        disable_residual_noise_decay=False,
        output_file="example_file.csv", num_stories=1, max_new_tokens=100, do_sample=True, include_sys=True, temperature=1.0, top_p=None, top_k=None, seed=None, print_output=False,
    ):
        output_csv = Path(output_file)



        for x in range(num_stories):
            story_seed = (seed + (128 * x)) if seed is not None else None

            if not include_sys:
                prompt = [{"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}]
            else:
                prompt = [{"role": "system", "content": prompts.SYS_NOISE}]
                prompt.append({"role": "user", "content": prompts.NOISE_1})
                
            output = self.generate_with_residual_stream_noise(
                prompt,
                residual_layers=residual_layers,
                residual_noise_std=residual_noise_std,
                residual_noise_decay=residual_noise_decay,
                disable_residual_noise_decay=disable_residual_noise_decay,
                logits_noise_std=logits_noise_std,
                logits_noise_decay=logits_noise_decay,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )

            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role": "assistant", "content": output})
            prompt.append({"role": "user", "content": prompts.NOISE_2})

            output = self.generate(
                prompt,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([output])

    def Double_Residual(
        self,
        residual_noise_std_stage1,
        residual_noise_std_stage2,
        residual_layers,
        residual_noise_decay=0.0,
        logits_noise_std=0.0,
        logits_noise_decay=0.0,
        output_file="example_file.csv",
        num_stories=1,
        max_new_tokens=100,
        do_sample=True,
        include_sys=True,
        temperature=1.0,
        top_p=None,
        top_k=None,
        seed=None,
        print_output=False,
        max_noise_tokens=200,
        disable_residual_noise_decay=False,
    ):
        """
        Two-stage generation with residual-stream noise injected in both stages.
        Stage 1 and Stage 2 use independent residual noise std values.
        """
        output_csv = Path(output_file)

        for x in range(num_stories):
            story_seed = (seed + (128 * x)) if seed is not None else None

            if not include_sys:
                prompt = [{"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}]
            else:
                prompt = [{"role": "system", "content": prompts.SYS_NOISE}]
                prompt.append({"role": "user", "content": prompts.NOISE_1})

            output = self.generate_with_residual_stream_noise(
                prompt,
                residual_layers=residual_layers,
                residual_noise_std=residual_noise_std_stage1,
                residual_noise_decay=residual_noise_decay,
                max_noise_tokens=max_noise_tokens,
                disable_residual_noise_decay=disable_residual_noise_decay,
                logits_noise_std=logits_noise_std,
                logits_noise_decay=logits_noise_decay,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )

            if print_output:
                print("----- FIRST STAGE OUTPUT -----\n")
                print(output)

            prompt.append({"role": "assistant", "content": output})
            prompt.append({"role": "user", "content": prompts.NOISE_2})

            output = self.generate_with_residual_stream_noise(
                prompt,
                residual_layers=residual_layers,
                residual_noise_std=residual_noise_std_stage2,
                residual_noise_decay=residual_noise_decay,
                max_noise_tokens=max_noise_tokens,
                disable_residual_noise_decay=disable_residual_noise_decay,
                logits_noise_std=logits_noise_std,
                logits_noise_decay=logits_noise_decay,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=story_seed,
            )

            if print_output:
                print("----- SECOND STAGE OUTPUT -----\n")
                print(output)

            # with output_csv.open(mode="a", newline="", encoding="utf-8") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([output])


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
