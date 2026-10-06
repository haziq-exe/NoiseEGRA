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



def local_model_path(model_id: str) -> str:
    """A saved copy of ``model_id`` if one is attached, else ``model_id``.

    On Kaggle a model can be saved once as a notebook's output and attached to
    later runs (scripts/kaggle_harness.py cache-model), so its weights are read
    from disk instead of downloaded each time. ``EGRA_LOCAL_MODELS`` lists the
    folders to look in (os.pathsep-separated); a copy is a subfolder named after
    the model id with '/' as '__', found at most two levels down."""
    import os
    from pathlib import Path
    roots = [r for r in os.environ.get("EGRA_LOCAL_MODELS", "").split(os.pathsep) if r]
    name = str(model_id).replace("/", "__")
    for r in roots:
        for cand in [Path(r) / name, *Path(r).glob(f"*/{name}"), *Path(r).glob(f"*/*/{name}")]:
            if (cand / "config.json").is_file():
                print(f"loading {model_id} from the saved copy at {cand}", flush=True)
                return str(cand)
    return model_id


def _cache_tensors(pkv):
    """Each layer's (keys, values) in a generation cache, whatever its kind:
    a Cache with ``layers`` (transformers 4.56+), one with ``key_cache`` and
    ``value_cache`` lists, or the legacy tuple of pairs."""
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers if getattr(l, "keys", None) is not None]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(kv[0], kv[1]) for kv in pkv]

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
        model = local_model_path(model)
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
            # Multimodal checkpoints keep the text decoder under language_model;
            # tried first so a vision or audio tower's layers are never taken.
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
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
                 typical_p=None, min_p=None, eta_cutoff=None, penalty_alpha=None,
                 hidden_prefix=None):
        """
        ``hidden_prefix``: text placed at the start of the reply, then a paragraph
        break; the story is what the model writes after it, and the prefix is not
        returned.

        prompt should always be a list of dicts of the form [ {"role" : "system", "content" : system_prompt},
                                              {"role" : "user", "content" : user_prompt}  ]
        """

        if seed is not None:
          torch.manual_seed(seed)

        chat_text = self.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        if hidden_prefix:
            chat_text = chat_text + str(hidden_prefix) + "\n\n"
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
        num_beams=1, num_return_sequences=1, length_penalty=1.0,
        num_beam_groups=1, diversity_penalty=0.0,
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
        # The noise's target from the model itself, measured once per prompt
        # before any story and before any hook is attached: top-p's shift and
        # the top word's probability along the greedy continuation with the
        # noise off (see online_calibration.target_from_top_share), and, with
        # online_rule_start, the starting length at which random draws of the
        # noise already reach that target. Before the seed, because measuring
        # the start draws noise of its own.
        if getattr(plan, "offset_vocab", False) and getattr(plan, "_vocab_pool", None) is None:
            plan._vocab_ids, plan._vocab_pool = self._content_word_vectors()
            print(f"  [vocab] noise directions drawn from {len(plan._vocab_ids)} content "
                  "words' input embeddings", flush=True)
        rule = None
        online = float(getattr(plan, "offset_online", 0.0) or 0.0) > 0
        measured = bool(getattr(plan, "offset_measured", False))
        rule_on = float(getattr(plan, "online_rule_k", 0.0) or 0.0) > 0 and (online or measured)
        # The rule steering's own effect, once per prompt, sized to
        # plan.steer_effect when that is set (and only printed otherwise, in an
        # arm that also measures the noise). First, because the noise's target
        # and start are measured along the steered passage, so the steering has
        # to be at its final strength before they are. Any arm with steering can
        # ask for it, steering alone included.
        want = float(getattr(plan, "steer_effect", 0.0) or 0.0)
        share = float(getattr(plan, "steer_share", 0.0) or 0.0)
        if (want > 0 or share > 0 or rule_on) and float(getattr(plan, "steer_budget", 0.0) or 0.0) > 0:
            scache = getattr(plan, "_steer_cache", None)
            if scache is None:
                scache = plan._steer_cache = {}
            ids = self.tokenizer(self.apply_chat_template(prompt, tokenize=False,
                                                          add_generation_prompt=True),
                                 return_tensors="pt")["input_ids"].to(self._input_device())
            skey = tuple(int(i) for i in ids.view(-1).tolist())
            if skey not in scache:
                from .online_calibration import budget_for_effect
                sz = budget_for_effect(self, plan, ids, want if want > 0 else None,
                                       share=share if share > 0 else None)
                smsg = (f"  [steer] unsteered top word {sz['top_prob']:.3f}; budget "
                        f"{sz['given']:g} moves the predictions {sz['at_given']:.4f} per step")
                if want > 0 or share > 0:
                    plan.steer_budget = sz["budget"]
                    smsg += (f"; sized to {sz['effect']:.4g}"
                             + (f" (share {share:g})" if share > 0 else "")
                             + f": budget {sz['budget']:.3f} "
                             f"moves them {sz['moves']:.4f}"
                             + ("" if sz["reached"] else "  -- NOT REACHED"))
                scache[skey] = sz
                print(smsg, flush=True)
        if rule_on:
            from .online_calibration import (_reference, measure_for_rule,
                                             start_for_target, target_from_top_share)
            cache = getattr(plan, "_rule_cache", None)
            if cache is None:
                cache = plan._rule_cache = {}
            ids = self.tokenizer(self.apply_chat_template(prompt, tokenize=False,
                                                          add_generation_prompt=True),
                                 return_tensors="pt")["input_ids"].to(self._input_device())
            key = tuple(int(i) for i in ids.view(-1).tolist())
            if key not in cache:
                ref = _reference(self, plan, ids, 48)
                hz = int(getattr(plan, "horizon_rank", 0) or 0)
                if hz > 0:
                    # Where the noise points, before it is sized: the start is
                    # measured with draws from this basis.
                    from .horizon import horizon_basis
                    hb, hd = horizon_basis(self, plan, ids, ref[0], rank=hz)
                    for l, v in hb.items():
                        if l in plan.layer_plans:
                            plan.layer_plans[l].horizon_basis = v
                    print(f"  [horizon] rank {hz} at {int(hd['layers'])} layers: "
                          f"{hd['gain']:.1f}x the far-ahead movement per unit of "
                          f"next-word movement of a random direction "
                          f"(lowest layer {hd['gain_min']:.1f}x)", flush=True)
                m = measure_for_rule(self, plan, ids, reference=ref)
                seen = ""
                if getattr(plan, "rule_unsteered", False):
                    # The unsteered top word, measured with the steering's own
                    # sizing above (same passage length, steering off). With no
                    # steering in the arm the steered reading already is that.
                    sc = (getattr(plan, "_steer_cache", None) or {}).get(key)
                    if sc is not None:
                        seen = f" (steered {m['top_prob']:.3f})"
                        m["steered_top_prob"] = m["top_prob"]
                        m["top_prob"] = float(sc["top_prob"])
                m["target"] = target_from_top_share(m["unit"], m["top_prob"],
                                                    float(plan.online_rule_k))
                if float(getattr(plan, "margin_scale", 0.0) or 0.0) > 0:
                    from .online_calibration import reference_margin
                    m["ref_margin"] = reference_margin(ref[1])
                    m["margin_norm"] = 1.0
                    if getattr(plan, "margin_neutral", False):
                        from .online_calibration import neutral_scale
                        m["margin_norm"] = neutral_scale(
                            ref[1], m["ref_margin"], power=float(plan.margin_scale),
                            lo=0.25, hi=float(getattr(plan, "margin_cap", 4.0) or 4.0),
                            word_start=(self._word_start_mask()
                                        if getattr(plan, "margin_words", False) else None))
                    print(f"  [margin] median top-two gap on the greedy continuation "
                          f"{m['ref_margin']:.3f} nats; average scale there "
                          f"{m['margin_norm']:.3f}"
                          + (" (divided out)" if getattr(plan, "margin_neutral", False) else ""),
                          flush=True)
                msg = (f"  [rule] top-p moves {m['unit']:.4f} per step, top word "
                       f"{m['top_prob']:.3f}{seen}: noise target {m['target']:.3f} "
                       f"(k {float(plan.online_rule_k):g})")
                if getattr(plan, "online_rule_start", False) or measured:
                    # Always the length at which the whole noise -- prompt and
                    # writing -- reaches the target, so a length means the same
                    # thing in an arm that writes with the noise off, or reads
                    # the prompt without it. Measured without the prompt's share
                    # the length comes out a third longer (14.5 against 11.1 on
                    # Qwen3-1.7B), more than the writing takes: the stories
                    # overshoot and the controller sits at its floor.
                    # Likewise unfaded along the prompt: a fade changes where
                    # the prompt's noise lands, not the length it is measured at.
                    decode, prefill = plan.offset_decode, plan.offset_prefill
                    taper = getattr(plan, "offset_taper", 1.0)
                    plan.offset_decode, plan.offset_prefill = True, True
                    plan.offset_taper = 1.0
                    dgain = getattr(plan, "daydream_gain", 1.0)
                    if getattr(plan, "daydream_free", False):
                        plan.daydream_gain = 1.0
                    try:
                        st = start_for_target(self, plan, ids, m["target"], m["unit"],
                                              reference=ref)
                        m.update(st)
                        from .online_calibration import noise_divergence
                        dv = noise_divergence(self, plan, ids, ref[0], st["start"])
                    finally:
                        plan.offset_decode, plan.offset_prefill = decode, prefill
                        plan.offset_taper = taper
                        plan.daydream_gain = dgain
                    msg += (f"; greedy text departs at token {dv['first']:.0f} of "
                            f"{dv['tokens']:.0f} (median of 4 draws, "
                            f"{dv['departed']:.0%} depart)")
                    msg += (f"; starting length {st['start']:.2f} "
                            f"({st['fraction']:.3f} of the norm) moves it "
                            f"{st['moves']:.3f}"
                            + ("" if st["reached"] else "  -- NOT REACHED at the norm"))
                from . import online_calibration as _oc
                if _oc.PLAN_KAPPA > 0 and _oc.PLAN_ALT_PROMPT is not None:
                    # Sized where the plan lives: the start replaced by the
                    # length that moves the plan-layer state a share of the
                    # distance between two story requests, held fixed (the
                    # controller, which reads the next-word distribution, is
                    # locked at 1x).
                    alt = self.tokenizer(self.apply_chat_template(
                        _oc.PLAN_ALT_PROMPT, tokenize=False, add_generation_prompt=True),
                        return_tensors="pt")["input_ids"].to(self._input_device())
                    ps = _oc.plan_space_start(self, plan, ids, alt, _oc.PLAN_KAPPA)
                    m["start"] = ps["start"]
                    plan.online_min_gain = plan.online_max_gain = 1.0
                    msg += (f"\n  [plan size] two requests' plan states {ps['ref']:.2f} apart "
                            f"at layer {_oc._plan_layer(self)}; length {ps['start']:.2f} "
                            f"({ps['fraction']:.3f} of the norm) moves the plan state "
                            f"{ps['moves']:.2f} (asked {ps['goal']:.2f}), held fixed"
                            + ("" if ps["reached"] else "  -- NOT REACHED at the norm"))
                cache[key] = m
                print(msg, flush=True)
            rule = cache[key]

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
        pick = getattr(plan, "_vocab_pick", None)
        if pick is not None and story_index is not None and int(story_index) < 3:
            words = [self.tokenizer.decode([plan._vocab_ids[int(i)]]).strip() for i in pick]
            print(f"  [vocab] story {story_index}: {', '.join(words)}", flush=True)

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

        # The rule's target, and with it the starting length when that was
        # measured too, applied to this story's own draw.
        if rule is not None:
            if online:
                plan.offset_online = float(rule["target"])
            if "start" in rule:
                from .online_calibration import scale_offsets
                scale_offsets(plan, float(rule["start"]))

        # Random noise at another place in the architecture, drawn for this
        # story from its seed and sized like the offset. Attached once the
        # steering hooks are, removed with them.
        arch = None
        mech = str(getattr(plan, "arch_mechanism", "") or "")
        if mech and float(getattr(plan, "arch_size", 0.0) or 0.0) > 0:
            from .arch_noise import ArchNoise, calibrate as arch_calibrate

            if getattr(plan, "_arch_cache", None) is None:
                plan._arch_cache = {}
            story_seed = (seed if seed is not None
                          else int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
            arch = ArchNoise(self, plan, mech, seed=story_seed,
                             n_prompt=int(input_ids.shape[-1]))
            got = arch_calibrate(arch, input_ids, float(plan.arch_size),
                                 cache=plan._arch_cache)
            # Sized on the first sentence's draw; later sentences get draws of
            # their own at the same knob.
            arch.per_sentence = bool(getattr(plan, "arch_per_sentence", False))
            if getattr(plan, "arch_log", None) is None:
                plan.arch_log = []
            plan.arch_log.append(got)
            print(f"  [{mech}] knob {got['knob']:.4f} moves the predictions "
                  f"{got['distance']:.2f} nucleus-units (asked {plan.arch_size:.2f})"
                  + ("" if got["reached"] else "  -- NOT REACHED at the largest knob"),
                  flush=True)

        # A shadow copy of the story: a second row with the same words and the
        # same steering but no perturbation. With shadow_protect the story is
        # held level with it along the protected directions at every steered
        # layer; with offset_online it is only measured against, to size the
        # noise while the story is written. Set up after the sizing above, which
        # measures on one row.
        protect = bool(getattr(plan, "shadow_protect", False))
        online = (float(getattr(plan, "offset_online", 0.0) or 0.0) > 0
                  and any(lp.offset is not None for lp in plan.layer_plans.values()))
        shadow = (protect or online) and (
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

        # An amplified random direction (noiseegra.hidden_context): grown on the
        # prompt alone, before any hook is attached, then added at the first
        # plan layer to every position after the first, under the steering.
        from . import hidden_context as _hc
        if _hc.AMPLIFY is not None:
            rad, steps = _hc.AMPLIFY[0], _hc.AMPLIFY[1]
            lock = float(_hc.AMPLIFY[2]) if len(_hc.AMPLIFY) > 2 else 0.0
            src = normalized_layers[0]
            tgt = max(src + 1, int(round(2 * len(blocks) / 3)))
            theta, a0, a1 = _hc.amplified_direction(
                self, input_ids[:1], int(seed if seed is not None else (story_index or 0)),
                rad, steps, src, tgt, lock=lock)
            plan._amplify_log = (a0, a1)
            amp_state = {"prefill": True}

            def amp_hook(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                with torch.no_grad():
                    v = theta.to(device=t.device, dtype=t.dtype)
                    if amp_state["prefill"]:
                        t[:, 1:] += v
                        amp_state["prefill"] = False
                    else:
                        t += v
                return None

            handles.append(blocks[src].register_forward_hook(amp_hook))
            if story_index is not None and int(story_index) < 3:
                kl = getattr(self, "_amplify_kl", None) if lock else None
                print(f"  [amplified {rad:g}x, {steps} steps"
                      + (f", reply type locked at {lock:g}" if lock else "") + f"] story {story_index}: "
                      f"layer {src} -> {tgt}; moves the planning state {a0:.3f} -> {a1:.3f} of its size"
                      + (f"; first-word change {kl[0]:.3f} -> {kl[-1]:.3f} nats" if kl else ""),
                      flush=True)

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
        sizer = None
        tilt = None
        guard = None
        anchor = None
        fb_mode = str(getattr(plan, "feedback_mode", "") or "")
        if fb_mode and float(getattr(plan, "feedback_eta", 0.0) or 0.0) <= 0:
            fb_mode = ""
        fb_stats = {"steps": 0, "cos_sum": 0.0}
        guard_alpha = float(getattr(plan, "guard_alpha", 0.0) or 0.0)
        guard_fix = guard_alpha > 0 and float(getattr(plan, "correct_eta", 0.0) or 0.0) > 0
        cache_box = {}
        debt_eta = float(getattr(plan, "debt_eta", 0.0) or 0.0)
        debt = None
        pulse_every = int(getattr(plan, "pulse_every", 0) or 0)
        pulser = None
        avoid_eta = float(getattr(plan, "avoid_eta", 0.0) or 0.0)
        avoid = None
        avoid_start = ({l: lp.offset.detach().float().clone()
                        for l, lp in plan.layer_plans.items() if lp.offset is not None}
                       if avoid_eta > 0 else {})
        fit_base = None
        if (fb_mode or guard_alpha > 0 or debt_eta > 0 or avoid_eta > 0
                or float(getattr(plan, "anchor_p1", 0.0) or 0.0) > 0) and not shadow:
            raise ValueError("the guard and the feedback read the noise-free shadow row; "
                             "they need the noise sized while writing (offset_online)")
        if avoid_eta > 0 and any(lp.offset_traj is not None for lp in plan.layer_plans.values()):
            raise ValueError("turning away from the default turns a constant per-story "
                             "direction; run it with no drift (noise_beta None)")
        if fb_mode and any(lp.offset_traj is not None for lp in plan.layer_plans.values()):
            raise ValueError("feedback turns a constant per-story direction; "
                             "run it with no drift (noise_beta None)")
        start_dirs = {l: lp.offset.detach().float().clone()
                      for l, lp in plan.layer_plans.items()
                      if fb_mode and lp.offset is not None}

        try:
            def model_pre_hook(module, inp):
                # A replay of the current step (the direction correction's
                # gradient) is not a new step.
                if shared.get("replay"):
                    return None
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
                            from .subspace import prompt_taper
                            taper = prompt_taper(plan)
                            off = None
                            if offset_here and (not bands or layer_idx in bands):
                                off = plan.layer_plans[layer_idx].offset
                                if off is not None:
                                    pg = getattr(plan, "offset_prefill_gain", 1.0)
                                    # 0 is no prompt noise (the fit can ask for it).
                                    off = off.to(target.device) * (1.0 if pg is None else float(pg))
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
                        if layer_idx == normalized_layers[0] and not shared.get("replay"):
                            entropy_state["steps"] += 1
                            entropy_state["open"] += int(gate_open)
                            # The controller may change the gain after this
                            # forward; a replay of this step needs this one.
                            shared["gain_used"] = getattr(plan, "online_gain", 1.0)

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
                        # directions; only that part is removed. A shadow that
                        # is there only to measure leaves the story alone.
                        prot = plan.layer_plans[layer_idx].protect if protect else None
                        if prot is not None:
                            prot = prot.to(device=target.device, dtype=torch.float32)
                            diff = (target[0, rows, :] - target[1, rows, :]).float()
                            target[0, rows, :].sub_(((diff @ prot) @ prot.t()).to(target.dtype))
                    return None

                return shadowed

            for layer_idx in normalized_layers:
                handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
            if arch is not None:
                arch.attach()

            # B-Trans: a Gaussian offset on every hidden-size norm layer, drawn
            # now from this story's seed and held for the whole story.
            if float(getattr(plan, "btrans_sigma", 0.0) or 0.0) > 0:
                from .dynamic_noise import attach_btrans
                handles.extend(attach_btrans(self.model, float(plan.btrans_sigma)))

            # The noise's direction turned each step by the displacement it
            # caused downstream: the story's state minus its shadow's at the
            # feedback layer, read after the steered layers have acted.
            if fb_mode:
                from .dynamic_noise import feedback_update
                fb_layer = self._normalize_layer_index(
                    int(getattr(plan, "feedback_layer", 20)), len(blocks))

                def fb_hook(module, input, output):
                    if shared.get("replay") or shared["is_prefill"]:
                        return None
                    t_ = output if isinstance(output, torch.Tensor) else (
                        output[0] if isinstance(output, (tuple, list)) and len(output) > 0
                        and isinstance(output[0], torch.Tensor) else None)
                    if t_ is None or t_.dim() != 3 or t_.shape[0] != 2:
                        return None
                    with torch.no_grad():
                        cos = feedback_update(plan, (t_[0, -1] - t_[1, -1]).float(),
                                              fb_mode, float(plan.feedback_eta))
                    fb_stats["steps"] += 1
                    fb_stats["cos_sum"] += cos
                    return None

                handles.append(blocks[fb_layer].register_forward_hook(fb_hook))

            # The generation's own cache, for replaying the current step with a
            # gradient when the guard corrects the noise's direction.
            if guard_fix or debt_eta > 0 or avoid_eta > 0 or pulse_every > 0:
                def grab_cache(module, args, kwargs):
                    if not shared.get("replay"):
                        cache_box["pkv"] = kwargs.get("past_key_values")
                    return None

                handles.append(self.model.register_forward_pre_hook(grab_cache, with_kwargs=True))

            # Which token this forward reads, for noise placed at sentence ends.
            plan._at_boundary = False
            if float(getattr(plan, "boundary_gain", 0.0) or 0.0) > 0:
                bset = self._boundary_ids()

                def see_token(module, args, kwargs):
                    if shared.get("replay"):
                        return None
                    ids = kwargs.get("input_ids", args[0] if args else None)
                    plan._at_boundary = bool(
                        ids is not None and ids.shape[-1] == 1
                        and not shared.get("is_prefill") and int(ids[0, -1]) in bset)
                    return None

                handles.append(self.model.register_forward_pre_hook(see_token, with_kwargs=True))

            # Forgetting the noise: at the first step the writing noise has
            # faded to nothing, the story's cached keys and values -- its memory
            # of the prompt, or of everything so far -- are overwritten with the
            # shadow row's, which read the same tokens with the steering and no
            # noise. From then on the noise survives only in the words it chose.
            forget = str(getattr(plan, "prompt_forget", "") or "")
            plan._forgot_at = None
            if forget:
                if not shadow:
                    raise ValueError("forgetting the noise copies the noise-free shadow "
                                     "row's memory; it needs the online sizing")
                n_prompt_f = int(inputs["input_ids"].shape[-1])

                def forget_noise(module, args, kwargs):
                    if (shared.get("replay") or shared.get("is_prefill")
                            or plan._forgot_at is not None):
                        return None
                    if plan.envelope_at(shared["cur_t"]) > 1e-6:
                        return None
                    pkv = kwargs.get("past_key_values")
                    if pkv is None:
                        return None
                    for k, v in _cache_tensors(pkv):
                        n = n_prompt_f if forget == "prompt" else k.shape[-2]
                        k[0, ..., :n, :].copy_(k[1, ..., :n, :])
                        v[0, ..., :n, :].copy_(v[1, ..., :n, :])
                    plan._forgot_at = int(shared["cur_t"])
                    return None

                handles.append(self.model.register_forward_pre_hook(forget_noise,
                                                                    with_kwargs=True))

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
            if arch is not None and arch.per_sentence:
                from .arch_noise import SentenceCounter
                processors = LogitsProcessorList(
                    [*(processors or []), SentenceCounter(self.tokenizer, arch)])
            # A hidden daydream: its words, and the paragraph break that ends
            # it, are written first and cut from the text afterwards.
            dd = int(getattr(plan, "daydream_steps", 0) or 0)
            dd_text = str(getattr(plan, "daydream_break", "") or "\n\n")
            dd_break = (self.tokenizer(dd_text, add_special_tokens=False)["input_ids"]
                        if dd > 0 else [])
            # With a boundary the daydream runs on past its length until it ends
            # a sentence or a line (at most dd_extra words more), so the story
            # does not start inside one of its sentences.
            dd_extra = 24 if (dd > 0 and getattr(plan, "daydream_boundary", False)) else 0
            dd_state = {"start": None}
            dd_hidden = dd + len(dd_break) if dd > 0 else 0
            plan._daydream_shift = dd_hidden
            dd_keep = bool(getattr(plan, "daydream_keep", False))
            if online and shadow:
                # First in the list, so it reads the model's own scores before
                # anything else has touched them.
                from .online_calibration import OnlineSizer
                absolute = None
                if getattr(plan, "online_absolute", False) and rule is not None:
                    absolute = float(rule["target"]) * float(rule["unit"])
                sizer = OnlineSizer(plan, float(plan.offset_online),
                                    bounds=(float(getattr(plan, "online_min_gain", 0.25) or 0.25),
                                            float(getattr(plan, "online_max_gain", 2.5) or 2.5)),
                                    absolute=absolute,
                                    hold=(int(getattr(plan, "offset_envelope_steps", 0) or 0)
                                          if str(getattr(plan, "offset_envelope", "flat")) == "rise"
                                          else dd_hidden + dd_extra))
                processors = LogitsProcessorList([sizer, *(processors or [])])
                ms = float(getattr(plan, "margin_scale", 0.0) or 0.0)
                if ms > 0 and rule is not None and "ref_margin" in rule:
                    # Right after the controller, which sizes the noise from
                    # its raw push; the push on the decision is then equalised.
                    from .online_calibration import MarginScaler
                    scaler = MarginScaler(float(rule["ref_margin"]), power=ms,
                                          hi=float(getattr(plan, "margin_cap", 4.0) or 4.0),
                                          word_start=(self._word_start_mask()
                                                      if getattr(plan, "margin_words", False)
                                                      else None),
                                          norm=float(rule.get("margin_norm", 1.0)),
                                          steps=int(getattr(plan, "margin_steps", 0) or 0),
                                          content=(self._content_mask()
                                                   if getattr(plan, "margin_content", False)
                                                   else None))
                    processors = LogitsProcessorList([sizer, scaler, *list(processors)[1:]])
                if guard_alpha > 0:
                    # After the controller, which must read the story's raw
                    # scores; before any cut-off.
                    from .dynamic_noise import CleanGuard, correct_offsets, prefix_cache

                    def on_violation(ids, token):
                        # Not on the first token: it comes from the prompt's
                        # forward pass, which a one-position replay does not
                        # reproduce (the prompt is perturbed differently).
                        if (not guard_fix or cache_box.get("pkv") is None
                                or shared["is_prefill"]):
                            return False
                        grads = self._token_gradients(
                            plan, blocks, normalized_layers, shared, cache_box["pkv"],
                            ids, token, prefix_cache)
                        if grads is None:
                            return False
                        return correct_offsets(plan, grads, float(plan.correct_eta),
                                               shared["cur_t"])

                    guard = CleanGuard(guard_alpha, on_violation)
                    processors = LogitsProcessorList([sizer, guard, *list(processors)[1:]])
                if debt_eta > 0:
                    # After the controller (raw scores); it only reads them.
                    from .dynamic_noise import RuleDebt, correct_offsets, debt_token_sets, prefix_cache
                    if getattr(self, "_debt_sets", None) is None:
                        self._debt_sets = debt_token_sets(self.tokenizer)

                    def on_suppressed(ids, tokens):
                        if cache_box.get("pkv") is None or shared["is_prefill"]:
                            return False
                        grads = self._token_gradients(
                            plan, blocks, normalized_layers, shared, cache_box["pkv"],
                            ids, tokens, prefix_cache)
                        if grads is None:
                            return False
                        # Remove the part of the noise that lowers the owed words.
                        return correct_offsets(
                            plan, {l: (None if g is None else -g) for l, g in grads.items()},
                            debt_eta, shared["cur_t"])

                    debt = RuleDebt(self.tokenizer, self._debt_sets,
                                    float(getattr(plan, "debt_tau", 0.3)), on_suppressed,
                                    active=lambda: plan.envelope_at(shared["cur_t"]) > 0.05)
                    processors = LogitsProcessorList([sizer, debt, *list(processors)[1:]])
                if pulse_every > 0:
                    # A new segment: a fresh direction, the context read again
                    # with it, the envelope restarted. Last, after everything
                    # that reads this step's scores.
                    from .dynamic_noise import PulseTrigger

                    def on_pulse(ids):
                        plan._in_pulse = True
                        try:
                            plan.resample_offset()
                        finally:
                            plan._in_pulse = False
                        plan.pulse_start = int(shared["cur_t"]) + 1
                        if cache_box.get("pkv") is None:
                            return
                        if str(getattr(plan, "pulse_reread", "context")) == "story":
                            self._reread_story(ids, int(inputs["input_ids"].shape[-1]),
                                               float(getattr(plan, "pulse_gain", 1.0)),
                                               plan, shared, cache_box)
                        else:
                            self._reread_prompt(
                                {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                                shared, cache_box)

                    pulser = PulseTrigger(self.tokenizer, pulse_every, on_pulse,
                                          start=lambda: plan.pulse_start,
                                          step=lambda: shared["cur_t"])
                    processors = LogitsProcessorList([*list(processors), pulser])
                if avoid_eta > 0:
                    # After the controller (raw scores); it only reads them.
                    from .dynamic_noise import DefaultAvoid, prefix_cache, rotate_away

                    turn_mode = str(getattr(plan, "avoid_mode", "avoid") or "avoid")
                    beta_m = float(getattr(plan, "avoid_momentum", 0.0) or 0.0)
                    reread = int(getattr(plan, "avoid_reread", 0) or 0)
                    avg = {}
                    moved = {"since": False, "rereads": 0}

                    def entropy_of(ls):
                        return -(ls.exp() * ls).sum()

                    def on_avoid(ids, target):
                        if cache_box.get("pkv") is None or shared["is_prefill"]:
                            return None
                        kind, token, sign = target
                        grads = self._token_gradients(
                            plan, blocks, normalized_layers, shared, cache_box["pkv"],
                            ids, entropy_of if kind == "entropy" else token, prefix_cache)
                        if grads is None:
                            return None
                        # The direction wanted: along +g to raise the objective
                        # (toward the word, or more entropy), -g to lower it.
                        want = {l: (None if g is None else sign * g / g.norm().clamp_min(1e-12))
                                for l, g in grads.items()}
                        if beta_m > 0:
                            # Toward the running average of the directions wanted.
                            for l, d in want.items():
                                if d is not None:
                                    avg[l] = d if l not in avg else beta_m * avg[l] + (1 - beta_m) * d
                            want = dict(avg)
                        cos = rotate_away(plan, {l: (None if d is None else -d)
                                                 for l, d in want.items()}, avoid_eta)
                        moved["since"] = True
                        return cos

                    def maybe_reread():
                        # Read the prompt again with the turned direction, every
                        # `reread` steps while the noise is on; the prompt's cache
                        # is replaced in place, the words written so far kept.
                        t_ = shared["cur_t"]
                        if (reread <= 0 or not moved["since"] or t_ <= 0 or t_ % reread
                                or cache_box.get("pkv") is None):
                            return
                        self._reread_prompt(inputs, shared, cache_box)
                        moved["since"] = False
                        moved["rereads"] += 1

                    avoid = DefaultAvoid(float(getattr(plan, "avoid_p1", 0.6)), on_avoid,
                                         active=lambda: plan.envelope_at(shared["cur_t"]) > 0.05,
                                         after=maybe_reread,
                                         cohere_alpha=float(getattr(plan, "cohere_alpha", 0.05)),
                                         mode=turn_mode,
                                         entropy_tau=float(getattr(plan, "entropy_tau", 0.5)))
                    processors = LogitsProcessorList([sizer, avoid, *list(processors)[1:]])
                if float(getattr(plan, "anchor_p1", 0.0) or 0.0) > 0:
                    # After the controller (raw scores) and before any cut-off.
                    from .dynamic_noise import ConfidentAnchor
                    anchor = ConfidentAnchor(float(plan.anchor_p1))
                    processors = LogitsProcessorList([sizer, anchor, *list(processors)[1:]])
            if (float(getattr(plan, "output_tilt", 0.0) or 0.0) > 0
                    and getattr(plan, "output_profile", None) is not None):
                # Last, after everything that reads the model's own scores and
                # before the sampler's own temperature and cut-offs.
                from .online_calibration import RuleTilt
                tilt = RuleTilt(plan.output_profile, float(plan.output_tilt))
                processors = LogitsProcessorList([*(processors or []), tilt])
            if shadow and getattr(plan, "flip_log", False):
                # After everything that changes the story's scores: where the
                # noise changed the clean model's choice, by its top-two gap.
                from .online_calibration import FlipRecorder
                stats = getattr(plan, "_flip_stats", None)
                if stats is None:
                    stats = plan._flip_stats = {}
                processors = LogitsProcessorList([*(processors or []), FlipRecorder(stats)])
            if dd > 0:
                # Last of all: at the end of the daydream every row writes the
                # paragraph break, whatever it would have chosen.
                n_in = int(inputs["input_ids"].shape[-1])

                tok = self.tokenizer

                class _DaydreamBreak(LogitsProcessor):
                    def __call__(self_, ids, scores):
                        s = int(ids.shape[-1]) - n_in
                        if dd_state["start"] is None and s >= dd:
                            last = tok.decode([int(ids[0, -1])]) if s > 0 else ""
                            # A closing quote ends a sentence only after . ! or ?,
                            # never after a comma ("...," she says).
                            tail = (tok.decode(ids[0, -3:].tolist()) if s > 0 else "").rstrip()
                            tail = tail.rstrip('"\u201d\u2019\'')
                            ended = "\n" in last or tail.endswith((".", "!", "?"))
                            if not dd_extra or ended or s >= dd + dd_extra:
                                dd_state["start"] = s
                                # The story's schedule starts at its first word.
                                plan._daydream_shift = s + len(dd_break)
                        b = dd_state["start"]
                        if b is not None and b <= s < b + len(dd_break):
                            forced = torch.full_like(scores, float("-inf"))
                            forced[:, dd_break[s - b]] = 0.0
                            return forced
                        return scores

                processors = LogitsProcessorList([*(processors or []), _DaydreamBreak()])
                target = float(getattr(plan, "daydream_surprise", 0.0) or 0.0)
                plan._dd_gain_live = None
                if target > 0 and shadow:
                    # First in the list, so it reads the raw scores of the story
                    # and its noise-free shadow.
                    plan._dd_gain_live = float(getattr(plan, "daydream_gain", 4.0) or 4.0)
                    sur = {"logp": None, "ent": None, "ratio": None, "log": []}

                    class _Surprise(LogitsProcessor):
                        def __call__(self_, ids, scores):
                            s = int(ids.shape[-1]) - n_in
                            in_dream = dd_state["start"] is None
                            if in_dream and s > 0 and sur["logp"] is not None:
                                chosen = int(ids[0, -1])
                                r = float(-sur["logp"][chosen]) / max(float(sur["ent"]), 0.1)
                                sur["ratio"] = r if sur["ratio"] is None else 0.7 * sur["ratio"] + 0.3 * r
                                step = (target / max(sur["ratio"], 1e-3)) ** 0.3
                                plan._dd_gain_live = min(max(plan._dd_gain_live * step, 1.0), 40.0)
                                sur["log"].append((round(r, 2), round(plan._dd_gain_live, 2)))
                            if in_dream and scores.shape[0] > 1:
                                lp = torch.log_softmax(scores[1].float(), dim=-1)
                                sur["logp"] = lp
                                sur["ent"] = float(-(lp.exp() * lp).sum())
                            return scores

                    processors = LogitsProcessorList([_Surprise(), *list(processors)])
                    plan._dd_surprise_log = sur["log"]
            if processors is not None:
                gen_kwargs["logits_processor"] = processors
            if int(num_beams) > 1:
                # Beam search over the steered model: every beam is a row of the
                # batch and gets the same push. Not with a per-story perturbation
                # or a shadow row, which assume one story in row 0.
                if shadow or online:
                    raise ValueError("beam search runs with the rule steering alone: "
                                     "no shadow row and no noise sized while writing")
                gen_kwargs.update(num_beams=int(num_beams), do_sample=False,
                                  num_return_sequences=int(num_return_sequences),
                                  length_penalty=float(length_penalty),
                                  early_stopping=True)
                for k in ("temperature", "top_p", "top_k"):
                    gen_kwargs.pop(k, None)
                if int(num_beam_groups) > 1:
                    # Diverse Beam Search (Vijayakumar et al., AAAI 2018): the beams
                    # split into groups, each penalised for repeating the tokens the
                    # groups before it chose at the same step. No longer in the
                    # transformers core (v5); it is loaded from the Hub.
                    gen_kwargs.update(custom_generate="transformers-community/group-beam-search",
                                      trust_remote_code=True,
                                      num_beam_groups=int(num_beam_groups),
                                      diversity_penalty=float(diversity_penalty))
            stopper = self._word_budget_stopper(
                inputs["input_ids"].shape[-1],
                (max_words + dd_hidden + dd_extra) if (max_words and dd_hidden and not dd_keep)
                else max_words)
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
            if shadow and float(getattr(plan, "prompt_fit_tau", 0.0) or 0.0) > 0:
                fit_base = float(plan.offset_prefill_gain)
                self._fit_prompt_noise(plan, inputs, shared, fit_base)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens + (0 if dd_keep else dd_hidden + dd_extra),
                **gen_kwargs
            )

        finally:
            if fit_base is not None:
                plan.offset_prefill_gain = fit_base
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            if arch is not None:
                arch.detach()
            if model_handle is not None:
                try:
                    model_handle.remove()
                except Exception:
                    pass

        if (getattr(plan, "flip_log", False) and getattr(plan, "_flip_stats", None)
                and story_index is not None and (int(story_index) + 1) % 25 == 0):
            from .online_calibration import FlipRecorder
            print(f"  [flips] after {int(story_index) + 1} stories, choices the noise "
                  f"changed by the clean model's top-two gap (nats): "
                  f"{FlipRecorder.summary(plan._flip_stats)}", flush=True)
        # The daydream and its break are context, not story.
        if dd_hidden and dd_state["start"] is not None:
            dd_hidden = int(dd_state["start"]) + len(dd_break)
        if dd_hidden and story_index is not None and int(story_index) < 3:
            dream = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:
                                                     input_ids.shape[-1] + dd_hidden],
                                          skip_special_tokens=True)
            print(f"  [daydream] story {story_index}: {' '.join(dream.split())[:240]}",
                  flush=True)
            slog = getattr(plan, "_dd_surprise_log", None)
            if slog:
                rs = [x[0] for x in slog]
                print(f"  [daydream surprise] story {story_index}: mean ratio "
                      f"{sum(rs) / len(rs):.2f} over {len(rs)} words, noise multiple "
                      f"{slog[0][1]:.1f} -> {slog[-1][1]:.1f}", flush=True)
        if dd_keep:
            dd_hidden = 0       # shown as the story's opening
        generated_ids = outputs[0][input_ids.shape[-1] + dd_hidden:]
        text = strip_reasoning(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        # Every returned sequence, best first, for a caller that keeps the set.
        self.last_candidates = [
            strip_reasoning(self.tokenizer.decode(o[input_ids.shape[-1] + dd_hidden:],
                                                  skip_special_tokens=True))
            for o in outputs[:max(1, int(num_return_sequences))]]
        if shadow:
            other = outputs[1][input_ids.shape[-1] + dd_hidden:]
            drift = int((generated_ids != other).sum())
            plan.shadow_drift = int(getattr(plan, "shadow_drift", 0) or 0) + drift
            if drift:
                print(f"  [shadow] the shadow copy left the story at {drift} of "
                      f"{generated_ids.numel()} positions -- what it protected or "
                      "measured was measured against the wrong text", flush=True)
        if sizer is not None:
            got = sizer.summary()
            start = next((lp.offset_length for lp in plan.layer_plans.values()
                          if lp.offset_length is not None), None)
            got["start_length"] = float(start) if start is not None else math.nan
            if getattr(plan, "online_log", None) is None:
                plan.online_log = []
            plan.online_log.append(got)
            if "late_gain" in got and getattr(plan, "online_carry", False):
                # The next story starts where this one settled.
                prev = float(getattr(plan, "_carried_gain", None) or 1.0)
                plan._carried_gain = prev * float(got["late_gain"])
                got["carried_into_next"] = plan._carried_gain
            if "final_gain" in got:
                print(f"  [online] length {got['start_length']:.2f} -> "
                      f"{got['start_length'] * got['final_gain']:.2f} "
                      f"(settled at {got['late_gain']:.2f}x), moved the predictions "
                      f"{got['achieved']:.2f} nucleus-units over the story "
                      f"(asked {float(plan.offset_online):.2f}); per step, top-p moves "
                      f"{got['budget']:.4f} and the noise {got['moved']:.4f}", flush=True)
                if str(getattr(plan, "offset_envelope", "")) == "budget":
                    end = plan.change_end
                    got["change_end"] = -1 if end is None else int(end)
                    got["change_spent"] = float(plan.change_spent)
                    print(f"  [budget] the noise changed {plan.change_spent:.2f} words in "
                          f"expectation and began to fade at step "
                          f"{'never' if end is None else int(end)}", flush=True)
        if anchor is not None:
            got = anchor.summary()
            if getattr(plan, "anchor_log", None) is None:
                plan.anchor_log = []
            plan.anchor_log.append(got)
            print(f"  [anchor] took the clean model's word at {int(got['anchored'])} of "
                  f"{int(got['steps'])} steps", flush=True)
        if guard is not None:
            got = guard.summary()
            if getattr(plan, "guard_log", None) is None:
                plan.guard_log = []
            plan.guard_log.append(got)
            print(f"  [guard] the noise pushed a token the clean model rules out at "
                  f"{int(got['violations'])} of {int(got['steps'])} steps; corrected the "
                  f"direction {int(got['corrections'])} times", flush=True)
        if pulser is not None:
            got = pulser.summary()
            if getattr(plan, "pulse_log", None) is None:
                plan.pulse_log = []
            plan.pulse_log.append(got)
            print(f"  [pulse] {int(got['pulses'])} new segments, at steps {got['at']}: a fresh "
                  f"direction each, the context read again with it", flush=True)
            plan.pulse_start = 0
        if avoid is not None:
            got = avoid.summary()
            ends = []
            for l, u0 in avoid_start.items():
                lp = plan.layer_plans[l]
                if lp.offset is not None:
                    a, b = u0.to(lp.offset.device), lp.offset.float()
                    ends.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)))
            got["start_end_cos"] = sum(ends) / len(ends) if ends else math.nan
            got["rereads"] = float(moved["rereads"])
            if getattr(plan, "avoid_log", None) is None:
                plan.avoid_log = []
            plan.avoid_log.append(got)
            kinds = ", ".join(f"{k[5:]} {int(v)}" for k, v in got.items() if k.startswith("kind_"))
            print(f"  [avoid] mode {getattr(plan, 'avoid_mode', 'avoid')}: turned the noise ({kinds or 'never'}) at "
                  f"{int(got['turns'])} of {int(got['steps'])} noisy steps (mean cosine "
                  f"{got['step_cos']:.4f} per turn); the direction ends at cosine "
                  f"{got['start_end_cos']:.3f} with where it started"
                  + (f"; the prompt was read again {int(got['rereads'])} times"
                     if got["rereads"] else ""), flush=True)
        if debt is not None:
            got = debt.summary()
            if getattr(plan, "debt_log", None) is None:
                plan.debt_log = []
            plan.debt_log.append(got)
            trig = ", ".join(f"{k[5:]} {int(v)}" for k, v in got.items() if k.startswith("trig_"))
            print(f"  [debt] the noise lowered an owed rule's words at "
                  f"{int(sum(v for k, v in got.items() if k.startswith('trig_')))} of "
                  f"{int(got['steps'])} noisy steps ({trig or 'none'}); corrected "
                  f"{int(got['corrections'])} times", flush=True)
        if fb_mode:
            turned = []
            for l, u0 in start_dirs.items():
                lp = plan.layer_plans[l]
                if lp.offset is not None:
                    a, b = u0.to(lp.offset.device), lp.offset.float()
                    turned.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)))
            got = {"steps": float(fb_stats["steps"]),
                   "step_cos": fb_stats["cos_sum"] / max(fb_stats["steps"], 1),
                   "start_end_cos": sum(turned) / len(turned) if turned else math.nan}
            if getattr(plan, "feedback_log", None) is None:
                plan.feedback_log = []
            plan.feedback_log.append(got)
            print(f"  [feedback] {fb_mode}: {int(got['steps'])} turns, mean cosine "
                  f"{got['step_cos']:.4f} per step; the direction ends at cosine "
                  f"{got['start_end_cos']:.3f} with where it started", flush=True)
        if tilt is not None:
            got = tilt.summary()
            if getattr(plan, "tilt_log", None) is None:
                plan.tilt_log = []
            plan.tilt_log.append(got)
            if got:
                print(f"  [tilt] toward the rules at beta {got['mean_beta']:.2f} on average, "
                      f"moving the predictions {got['achieved']:.2f} of top-p's distortion "
                      f"(asked {float(plan.output_tilt):.2f})", flush=True)
        if gate_threshold > 0 and entropy_state["steps"]:
            self.last_gate_rate = entropy_state["open"] / entropy_state["steps"]
        else:
            self.last_gate_rate = 1.0
        return text

    @torch.no_grad()
    def _reread_prompt(self, inputs, shared, cache_box) -> None:
        """Replace the prompt's part of the generation cache by a fresh read.

        The prompt (story and shadow rows) is read with the hooks attached, as
        at the start, so the story row gets the prompt noise in its current
        direction at its usual size; its keys and values overwrite the first
        positions of every layer of the generation's cache. The words written
        so far keep theirs. The step counters are left as they were.
        """
        saved = {k: shared[k] for k in ("forward_calls", "t", "cur_t", "is_prefill")}
        pkv = cache_box["pkv"]
        shared.update(forward_calls=0, t=0, cur_t=0, is_prefill=True)
        try:
            fresh = self.model(**inputs, use_cache=True).past_key_values
        finally:
            shared.update(saved)
            cache_box["pkv"] = pkv
        n = int(inputs["input_ids"].shape[-1])

        def kv(cache):
            # The layered cache, the older one with key/value lists, or tuples.
            if getattr(cache, "layers", None) is not None:
                return [(l.keys, l.values) for l in cache.layers]
            if hasattr(cache, "key_cache"):
                return list(zip(cache.key_cache, cache.value_cache))
            return [(k, v) for k, v in cache]

        for (ok, ov), (fk, fv) in zip(kv(pkv), kv(fresh)):
            ok[:, :, :n].copy_(fk[:, :, :n].to(ok.dtype))
            ov[:, :, :n].copy_(fv[:, :, :n].to(ov.dtype))

    @torch.no_grad()
    def _reread_story(self, ids, prompt_len: int, gain: float, plan, shared, cache_box) -> None:
        """Read the story written so far again, the prompt's reading kept.

        The story's positions (after ``prompt_len``) are read on top of the
        prompt's cached keys and values, with the hooks attached as at the
        prompt's read, so the story row gets the noise in its current direction
        at ``gain`` times the starting length (the shadow gets the steering
        only); their keys and values replace the story's part of the cache.
        """
        from transformers import DynamicCache
        pkv = cache_box["pkv"]

        def kv(cache):
            if getattr(cache, "layers", None) is not None:
                return [(l.keys, l.values) for l in cache.layers]
            if hasattr(cache, "key_cache"):
                return list(zip(cache.key_cache, cache.value_cache))
            return [(k, v) for k, v in cache]

        n = int(ids.shape[-1])
        ctx = DynamicCache()
        for i, (k, v) in enumerate(kv(pkv)):
            ctx.update(k[:, :, :prompt_len].clone(), v[:, :, :prompt_len].clone(), i)
        saved = {k: shared[k] for k in ("forward_calls", "t", "cur_t", "is_prefill")}
        saved_gain = plan.offset_prefill_gain
        shared.update(forward_calls=0, t=0, cur_t=0, is_prefill=True)
        plan.offset_prefill_gain = float(gain)
        try:
            fresh = self.model(input_ids=ids[:, prompt_len:], past_key_values=ctx,
                               use_cache=True).past_key_values
        finally:
            shared.update(saved)
            plan.offset_prefill_gain = saved_gain
            cache_box["pkv"] = pkv
        for (ok, ov), (fk, fv) in zip(kv(pkv), kv(fresh)):
            ok[:, :, prompt_len:n].copy_(fk[:, :, prompt_len:n].to(ok.dtype))
            ov[:, :, prompt_len:n].copy_(fv[:, :, prompt_len:n].to(ov.dtype))

    @torch.no_grad()
    def _fit_prompt_noise(self, plan, inputs, shared, base: float) -> float:
        """Size this story's prompt noise by what it does to the first word.

        Reads the prompt (story and shadow rows, hooks attached) at the
        prompt's noise times 1, 3/4, 1/2, 1/4 and 0, and keeps the first at
        which the story's chance of a non-story opening is at most the larger
        of ``prompt_fit_floor`` and the shadow's chance times e^tau. Leaves
        ``plan.offset_prefill_gain`` at that size and the step counters as if
        nothing had run; the caller restores the base size after the story.
        """
        from .dynamic_noise import slip_token_ids
        if getattr(self, "_slip_ids", None) is None:
            self._slip_ids = slip_token_ids(self.tokenizer)
        ids = torch.tensor(self._slip_ids, dtype=torch.long)
        tau = float(plan.prompt_fit_tau)
        floor = float(getattr(plan, "prompt_fit_floor", 0.05))
        redraw = str(getattr(plan, "prompt_fit_mode", "shrink")) == "redraw"
        # Full size with up to 8 fresh directions first when redrawing; then
        # the size steps down with whatever direction was drawn last.
        ladder = ([1.0] * 8 if redraw else [1.0]) + [0.75, 0.5, 0.25, 0.0]
        tried, draws = [], 1
        for i, m in enumerate(ladder):
            if redraw and 0 < i < 8:
                plan.resample_offset()
                draws += 1
            plan.offset_prefill_gain = base * m
            shared.update(forward_calls=0, t=0, cur_t=0, is_prefill=True)
            logits = self.model(**inputs, use_cache=False).logits[:2, -1].float()
            p = torch.softmax(logits, dim=-1)[:, ids.to(logits.device)].sum(dim=-1)
            story, clean = float(p[0]), float(p[1])
            tried.append((m, story))
            if story <= max(floor, clean * math.exp(tau)):
                break
        shared.update(forward_calls=0, t=0, cur_t=0, is_prefill=True)
        got = {"gain": m, "first": tried[0][1], "kept": story, "shadow": clean,
               "reads": float(len(tried)), "draws": float(draws)}
        if getattr(plan, "fit_log", None) is None:
            plan.fit_log = []
        plan.fit_log.append(got)
        print(f"  [fit] prompt noise at {m:.2f}x of its size"
              + (f", direction {draws} of those drawn" if redraw else "")
              + f": chance of a non-story opening {tried[0][1]:.3f} -> {story:.3f} "
              f"(noise-free {clean:.3f})", flush=True)
        return m

    def _content_mask(self) -> torch.Tensor:
        """True for vocabulary entries that are a whole word of five or more
        letters (word-initial marker, then letters only): mostly content words."""
        cached = getattr(self, "_content_cache", None)
        if cached is not None:
            return cached
        n = int(self.model.get_input_embeddings().weight.shape[0])
        try:
            n = min(n, len(self.tokenizer))
        except TypeError:
            pass
        toks = (self.tokenizer.convert_ids_to_tokens(list(range(n)))
                if hasattr(self.tokenizer, "convert_ids_to_tokens")
                else [self.tokenizer.decode([i]) for i in range(n)])
        mask = torch.zeros(n, dtype=torch.bool)
        for i, s in enumerate(toks):
            if (isinstance(s, str) and len(s) >= 6 and s[0] in "\u0120\u2581 "
                    and s[1:].isalpha()):
                mask[i] = True
        self._content_cache = mask
        return mask

    def _boundary_ids(self) -> set:
        """Vocabulary entries that end a sentence or a line: their text, past any
        closing quotes and spaces, ends with . ! or ?, or holds a line break."""
        cached = getattr(self, "_boundary_cache", None)
        if cached is not None:
            return cached
        n = int(self.model.get_input_embeddings().weight.shape[0])
        try:
            n = min(n, len(self.tokenizer))
        except TypeError:
            pass
        out = set()
        for i in range(n):
            s = self.tokenizer.decode([i])
            if "\n" in s or s.rstrip().rstrip('"\u201d\u2019\'').endswith((".", "!", "?")):
                out.add(i)
        self._boundary_cache = out
        return out

    def _content_word_vectors(self):
        """Ids and unit input embeddings, centred on the mean embedding, of the
        vocabulary entries that are a whole lower-case word of four letters or
        more -- mostly content words. No text is generated or read."""
        n = int(self.model.get_input_embeddings().weight.shape[0])
        try:
            n = min(n, len(self.tokenizer))
        except TypeError:
            pass
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            toks = self.tokenizer.convert_ids_to_tokens(list(range(n)))
        else:
            toks = [self.tokenizer.decode([i]) for i in range(n)]
        ids = [i for i, s in enumerate(toks)
               if isinstance(s, str) and len(s) >= 5 and s[0] in "\u0120\u2581 "
               and s[1:].isalpha() and s[1:].islower() and s[1:].isascii()]
        if not ids:
            raise ValueError("no content words found in the vocabulary")
        w = self.model.get_input_embeddings().weight.detach().float().cpu()
        v = w[ids] - w[:n].mean(0, keepdim=True)
        return ids, v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _word_start_mask(self) -> torch.Tensor:
        """True for vocabulary entries that begin a new word, or are punctuation,
        whitespace or special -- every entry that is not the inside of a word."""
        cached = getattr(self, "_word_start_cache", None)
        if cached is not None:
            return cached
        n = int(getattr(self.model.config, "vocab_size", 0) or 0)
        try:
            n = max(n, len(self.tokenizer))
        except TypeError:
            pass
        mask = torch.ones(n, dtype=torch.bool)
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            toks = self.tokenizer.convert_ids_to_tokens(list(range(n)))
        else:
            toks = [self.tokenizer.decode([i]) for i in range(n)]
        for i, s in enumerate(toks):
            # A word-initial piece is marked: "Ġrain" (byte-level BPE) or
            # "▁rain" (SentencePiece). A piece that starts with a bare letter or
            # digit continues the word before it.
            # The markers are themselves letters to str.isalnum() ("Ġ"), so
            # they are checked first.
            if (isinstance(s, str) and s and s[0] not in "\u0120\u010a\u2581 \n\t"
                    and s[0].isalnum()):
                mask[i] = False
        self._word_start_cache = mask
        return mask

    def _token_gradients(self, plan, blocks, layers, shared, cache, input_ids, token,
                         prefix_cache):
        """d log p(token) / d(a vector added at each steered layer), this step.

        Replays the step just taken for the story (row 0) with its own cache up
        to the previous position and the same perturbation (the hooks see the
        same step, and the gain the controller used for it), plus a zero vector
        per steered layer at the current position whose gradient is returned.
        The generation's cache and step count are untouched.
        """
        n = int(input_ids.shape[-1])
        try:
            ctx = prefix_cache(cache, 0, n - 1)
        except Exception as err:  # an unfamiliar cache layout
            if not getattr(self, "_replay_warned", False):
                print(f"  [guard] cannot replay the step ({err!r}); no correction", flush=True)
                self._replay_warned = True
            return None
        p0 = next(iter(self.model.parameters()))
        es = {l: torch.zeros(int(plan.dim), dtype=p0.dtype, device=p0.device,
                             requires_grad=True) for l in layers}

        def add_e(l):
            def h(module, inp, out):
                if isinstance(out, torch.Tensor):
                    return out + es[l].to(out.device).view(1, 1, -1)
                if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                    return (out[0] + es[l].to(out[0].device).view(1, 1, -1), *out[1:])
                return None
            return h

        hs = [blocks[l].register_forward_hook(add_e(l)) for l in layers]
        saved = getattr(plan, "online_gain", 1.0)
        plan.online_gain = shared.get("gain_used", saved)
        shared["replay"] = True
        try:
            with torch.enable_grad():
                out = self.model(input_ids=input_ids[:1, -1:], past_key_values=ctx,
                                 use_cache=True)
                ls = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                if callable(token):
                    # Any objective of the story's log-probabilities.
                    lp = token(ls)
                elif isinstance(token, (list, tuple)):
                    # A set of words: the log of their total probability.
                    lp = torch.logsumexp(ls[torch.tensor(token, device=ls.device)], dim=0)
                else:
                    lp = ls[int(token)]
                # Kept so a test can check the replay reproduced the step.
                self._last_replay_logprob = float(lp)
                grads = torch.autograd.grad(lp, [es[l] for l in layers], allow_unused=True)
        finally:
            shared["replay"] = False
            plan.online_gain = saved
            for h in hs:
                h.remove()
        return {l: (None if g is None else g.detach().float()) for l, g in zip(layers, grads)}

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
