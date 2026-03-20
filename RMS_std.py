import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union
from .EGRA_functions import EGRA
import torch


@dataclass
class RMSStats:
    """
    Accumulates per-layer RMS statistics of a tensor on the last token position.
    RMS(layer) = sqrt( mean_t( mean_{B,H}(x^2) ) )
    """
    sumsq: Dict[int, float] = field(default_factory=dict)
    count: Dict[int, int] = field(default_factory=dict)

    def update_last_token(self, layer_idx: int, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return

        xt = x[:, -1, :].float()

        if not torch.isfinite(xt).all():
            return

        v = (xt * xt).mean().item()
        self.sumsq[layer_idx] = self.sumsq.get(layer_idx, 0.0) + v
        self.count[layer_idx] = self.count.get(layer_idx, 0) + 1


    def rms(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k, s in self.sumsq.items():
            c = max(1, self.count.get(k, 0))
            out[k] = math.sqrt(s / c)
        return out


class RMSCalibrator:
    """
    Calibrates per-model noise standard deviations for two injection sites:

        1. Block output (residual stream noise):
               collect_block_rms()  ->  alpha_from_target_std()  ->  std_for_model()

        2. Self-attention output (attention output noise):
               collect_attn_rms()   ->  alpha_from_target_std()  ->  std_for_model()

    Both measure activation magnitudes during *decode* steps only, matching the
    noise injection implementations which skip the prompt prefill pass.

    Typical usage
    -------------
    Calibrate alpha on a reference model (e.g. Jais), then transfer to others:

        cal   = RMSCalibrator(egra_jais)

        # --- Residual stream ---
        rms_block = cal.collect_block_rms(prompt, layers=range(2, 30))
        alpha_res  = cal.alpha_from_target_std(target_std=5.25,
                         rms_by_layer=rms_block, layer_set=range(2, 30))

        # --- Attention output ---
        rms_attn  = cal.collect_attn_rms(prompt, layers=range(2, 30))
        alpha_attn = cal.alpha_from_target_std(target_std=5.25,
                         rms_by_layer=rms_attn, layer_set=range(2, 30))

        # Transfer to a new model:
        cal_new          = RMSCalibrator(egra_allam)
        rms_block_new    = cal_new.collect_block_rms(prompt, layers=range(2, 30))
        res_std_new      = cal_new.std_for_model(alpha_res,  rms_block_new, range(2, 30))

        rms_attn_new     = cal_new.collect_attn_rms(prompt,  layers=range(2, 30))
        attn_std_new     = cal_new.std_for_model(alpha_attn, rms_attn_new,  range(2, 30))
    """

    def __init__(self, egra: "EGRA"):
        self.egra = egra

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _model_device(self) -> torch.device:
        return next(self.egra.model.parameters()).device

    def _prompt_to_text(self, prompt) -> str:
        if isinstance(prompt, (list, tuple)) and len(prompt) > 0 and isinstance(prompt[0], dict):
            return self.egra.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        if isinstance(prompt, str):
            return prompt
        raise ValueError("prompt must be a chat-style list[dict] or a plain string.")

    def _extract_tensor(self, output) -> Optional[torch.Tensor]:
        """
        Extracts the primary hidden-state tensor from a module output.
        Works for both block outputs (hidden_states, ...) and self_attn outputs
        (attn_output, attn_weights, past_key_value) since both place the
        relevant tensor at position 0.
        """
        if isinstance(output, torch.Tensor):
            return output
        if (
            isinstance(output, (tuple, list))
            and len(output) > 0
            and isinstance(output[0], torch.Tensor)
        ):
            return output[0]
        return None

    @torch.no_grad()
    def _collect_rms(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        layers: Sequence[int],
        *,
        hook_target: str,           # "block" or "self_attn"
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        debug_shapes: bool = False,
    ) -> Dict[int, float]:
        """
        Shared collection loop used by both collect_block_rms and collect_attn_rms.
        Registers hooks on either blocks[li] or blocks[li].self_attn depending on
        hook_target, then runs a manual decode loop collecting decode-step-only RMS.
        """
        if hook_target not in ("block", "self_attn"):
            raise ValueError("hook_target must be 'block' or 'self_attn'.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        blocks = self.egra._get_transformer_blocks()
        n_layers = len(blocks)
        norm_layers = sorted({
            self.egra._normalize_layer_index(i, n_layers) for i in layers
        })

        stats = RMSStats()
        handles = []
        dbg = {"printed": 0}

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                t = self._extract_tensor(out)
                if t is None or t.dim() != 3:
                    return None

                # Only collect on decode steps (seq_len == 1 with KV cache active)
                if t.shape[1] != 1:
                    return None

                if debug_shapes and dbg["printed"] < 20:
                    print(
                        f"[RMS hook | {hook_target}] "
                        f"layer={layer_idx} shape={tuple(t.shape)} out_type={type(out)}"
                    )
                    dbg["printed"] += 1

                stats.update_last_token(layer_idx, t)
                return None
            return hook

        try:
            for li in norm_layers:
                target_module = (
                    blocks[li] if hook_target == "block"
                    else blocks[li].self_attn
                )
                handles.append(target_module.register_forward_hook(make_hook(li)))

            device = self._model_device()
            self.egra.model.eval()

            text = self._prompt_to_text(prompt)
            enc = self.egra.tokenizer(text, return_tensors="pt").to(device)
            enc.pop("token_type_ids", None)

            # Prefill
            out = self.egra.model(**enc, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

            # Manual decode loop — seq_len == 1 at every step
            for _ in range(max_new_tokens):
                if temperature <= 0 or not do_sample:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    safe_logits = (next_logits.float() / temperature)
                    safe_logits = torch.nan_to_num(
                        safe_logits,
                        nan=0.0,
                        posinf=1e9,
                        neginf=-1e9,
                    )
                    safe_logits = safe_logits - safe_logits.amax(dim=-1, keepdim=True)

                    probs = torch.softmax(safe_logits, dim=-1)
                    probs = torch.nan_to_num(
                        probs,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    probs_sum = probs.sum(dim=-1, keepdim=True)

                    if (
                        not torch.isfinite(probs).all()
                        or (probs < 0).any()
                        or (probs_sum <= 0).any()
                    ):
                        if debug_shapes:
                            print("[RMS sampler] Invalid probability row; falling back to argmax.")
                        next_token = torch.argmax(safe_logits, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs / probs_sum, num_samples=1)

                out = self.egra.model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                next_logits = out.logits[:, -1, :]

        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        return stats.rms()

    # ------------------------------------------------------------------ #
    #  Public collection methods                                           #
    # ------------------------------------------------------------------ #

    def collect_block_rms(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        layers: Sequence[int],
        *,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        debug_shapes: bool = False,
    ) -> Dict[int, float]:
        """
        Measures RMS of block outputs (full transformer block: attn + MLP + residual).
        Use this to calibrate noise std for residual stream noise injection.
        """
        return self._collect_rms(
            prompt, layers,
            hook_target="block",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            seed=seed,
            debug_shapes=debug_shapes,
        )

    def collect_attn_rms(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        layers: Sequence[int],
        *,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        debug_shapes: bool = False,
    ) -> Dict[int, float]:
        """
        Measures RMS of self_attn outputs (post o_proj, pre-MLP, pre-residual add).
        Use this to calibrate noise std for attention output noise injection.
        """
        return self._collect_rms(
            prompt, layers,
            hook_target="self_attn",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            seed=seed,
            debug_shapes=debug_shapes,
        )

    @torch.no_grad()
    def collect_embedding_rms(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        debug_shapes: bool = False,
    ) -> float:
        """
        Measures RMS of the token embedding lookup table output during decode
        steps only. Returns a single scalar (there is only one embedding layer).

        Use with alpha_from_target_std / std_for_model by wrapping the result
        in a single-key dict, e.g.:
            rms_val = cal.collect_embedding_rms(prompt)
            rms_dict = {0: rms_val}
            alpha = cal.alpha_from_target_std(target, rms_dict, layer_set=[0])
            std   = cal.std_for_model(alpha, rms_dict, layer_set=[0])
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        stats = RMSStats()
        handles = []
        dbg = {"printed": 0}

        embed_layer = self.egra.model.get_input_embeddings()

        def embed_hook(module, inp, out):
            if not isinstance(out, torch.Tensor) or out.dim() != 3:
                return None
            # Only collect on decode steps (seq_len == 1 with KV cache active)
            if out.shape[1] != 1:
                return None
            if debug_shapes and dbg["printed"] < 20:
                print(
                    f"[RMS hook | embedding] shape={tuple(out.shape)} "
                    f"type={type(out)}"
                )
                dbg["printed"] += 1
            stats.update_last_token(0, out)
            return None

        try:
            handles.append(embed_layer.register_forward_hook(embed_hook))

            device = self._model_device()
            self.egra.model.eval()

            text = self._prompt_to_text(prompt)
            enc = self.egra.tokenizer(text, return_tensors="pt").to(device)
            enc.pop("token_type_ids", None)

            # Prefill
            out = self.egra.model(**enc, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

            # Manual decode loop — seq_len == 1 at every step
            for _ in range(max_new_tokens):
                if temperature <= 0 or not do_sample:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    safe_logits = (next_logits.float() / temperature)
                    safe_logits = torch.nan_to_num(
                        safe_logits, nan=0.0, posinf=1e9, neginf=-1e9,
                    )
                    safe_logits = safe_logits - safe_logits.amax(dim=-1, keepdim=True)

                    probs = torch.softmax(safe_logits, dim=-1)
                    probs = torch.nan_to_num(
                        probs, nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    probs_sum = probs.sum(dim=-1, keepdim=True)

                    if (
                        not torch.isfinite(probs).all()
                        or (probs < 0).any()
                        or (probs_sum <= 0).any()
                    ):
                        if debug_shapes:
                            print("[RMS sampler] Invalid probability row; falling back to argmax.")
                        next_token = torch.argmax(safe_logits, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs / probs_sum, num_samples=1)

                out = self.egra.model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                next_logits = out.logits[:, -1, :]

        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        rms_dict = stats.rms()
        if 0 not in rms_dict:
            raise ValueError(
                "No embedding RMS samples collected. "
                "Check that the model uses a standard nn.Embedding layer."
            )
        return rms_dict[0]

    # ------------------------------------------------------------------ #
    #  Aggregation and alpha / std computation                             #
    # ------------------------------------------------------------------ #

    def aggregate_rms(
        self,
        rms_by_layer: Dict[int, float],
        layer_set: Sequence[int],
        *,
        agg: str = "median",
    ) -> float:
        blocks = self.egra._get_transformer_blocks()
        n_layers = len(blocks)
        idxs = [self.egra._normalize_layer_index(i, n_layers) for i in layer_set]
        vals = [rms_by_layer[i] for i in idxs if i in rms_by_layer and math.isfinite(rms_by_layer[i])]
        if not vals:
            raise ValueError(
                "No RMS values found for the requested layers. "
                "Did you run collect_block_rms / collect_attn_rms first?"
            )
        t = torch.tensor(vals, dtype=torch.float32)
        if agg == "median":
            return float(t.median().item())
        if agg == "mean":
            return float(t.mean().item())
        raise ValueError("agg must be 'median' or 'mean'.")

    def alpha_from_target_std(
        self,
        target_std: float,
        rms_by_layer: Dict[int, float],
        layer_set: Sequence[int],
        *,
        agg: str = "median",
    ) -> float:
        """
        Computes alpha such that: target_std = alpha * aggregate_rms(layer_set).

        Run this once on a reference model to fix alpha, then use std_for_model()
        to transfer the equivalent noise level to every other model.
        Works for both block RMS and attn RMS — pass the appropriate rms_by_layer.
        """
        a = self.aggregate_rms(rms_by_layer, layer_set, agg=agg)
        if a <= 0:
            raise ValueError("Non-positive aggregate RMS — check your layer indices.")
        return float(target_std / a)

    def std_for_model(
        self,
        alpha: float,
        rms_by_layer: Dict[int, float],
        layer_set: Sequence[int],
        *,
        agg: str = "median",
    ) -> float:
        """
        Computes the noise std for a model given a fixed alpha:
            noise_std = alpha * aggregate_rms(layer_set)

        Pass block RMS values for residual stream noise,
        or attn RMS values for attention output noise.
        """
        a = self.aggregate_rms(rms_by_layer, layer_set, agg=agg)
        return float(alpha * a)
