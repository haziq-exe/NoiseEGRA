import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
from .EGRA_functions import EGRA
import torch


@dataclass
class RMSStats:
    """
    Accumulates per-layer RMS statistics of the block output on the last token.
    RMS(layer) = sqrt( mean_t( mean_{B,H}(x^2) ) )
    """
    sumsq: Dict[int, float]
    count: Dict[int, int]

    def update_last_token(self, layer_idx: int, x: torch.Tensor) -> None:
        # Expect x is (B, T, H). Use last token only.
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return
        xt = x[:, -1, :].float()           # (B, H)
        v = (xt * xt).mean().item()        # scalar mean over B and H
        self.sumsq[layer_idx] = self.sumsq.get(layer_idx, 0.0) + v
        self.count[layer_idx] = self.count.get(layer_idx, 0) + 1

    def rms(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k, s in self.sumsq.items():
            c = max(1, self.count.get(k, 0))
            out[k] = math.sqrt(s / c)
        return out


class ResidualRMSCalibrator:
    """
    Designed to be used with your EGRA instance:

        egra = EGRA("inceptionai/jais-family-13b-chat")
        cal = ResidualRMSCalibrator(egra)
        rms = cal.collect_layer_rms(prompt, residual_layers=range(5,29), max_new_tokens=32)
        alpha = cal.alpha_from_target_std_fixed(5.25, rms, layer_set=range(5,29))

    Notes:
      - Measures block outputs during *decode* steps only (seq_len == 1 calls),
        matching your residual-noise implementation (which skips prefill).
      - Uses your EGRA._get_transformer_blocks and EGRA._normalize_layer_index.
      - Does NOT normalize by number of layers (keeps your experimental condition).
    """

    def __init__(self, egra: "EGRA"):
        self.egra = egra

    def _model_device(self) -> torch.device:
        # Avoid relying on hf_device_map; just use model params.
        return next(self.egra.model.parameters()).device

    def _extract_block_output_tensor(self, output) -> Optional[torch.Tensor]:
        # Many blocks return (hidden_states, ...). We want hidden_states.
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
        return None

    def _prompt_to_text(self, prompt) -> str:
        # Your EGRA assumes chat-style prompt (list[dict]).
        # Keep it consistent with your generation path.
        if isinstance(prompt, (list, tuple)) and len(prompt) > 0 and isinstance(prompt[0], dict):
            return self.egra.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, str):
            # If you ever pass raw strings, allow it.
            return prompt
        raise ValueError("prompt must be a chat-style list[dict] or a string.")

    @torch.no_grad()
    def collect_layer_rms(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        residual_layers: Sequence[int],
        *,
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        seed: Optional[int] = 0,
        include_prefill: bool = False,
        debug_shapes: bool = False,
    ) -> Dict[int, float]:
        """
        Collect per-layer RMS for the given prompt, at the specified residual_layers.

        include_prefill=False (default) means:
          - only count forward calls where seq_len == 1 (decode steps with KV cache).

        debug_shapes=True prints (layer_idx, shape, type(output)) for the first few hits.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        blocks = self.egra._get_transformer_blocks()
        n_layers = len(blocks)
        norm_layers = sorted({self.egra._normalize_layer_index(i, n_layers) for i in residual_layers})

        stats = RMSStats(sumsq={}, count={})
        handles = []

        dbg = {"printed": 0}

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                t = self._extract_block_output_tensor(out)
                if t is None or t.dim() != 3:
                    return None

                is_decode_step = (t.shape[1] == 1)
                if not include_prefill and not is_decode_step:
                    return None

                if debug_shapes and dbg["printed"] < 20:
                    print(f"[RMS hook] layer={layer_idx} shape={tuple(t.shape)} out_type={type(out)}")
                    dbg["printed"] += 1

                stats.update_last_token(layer_idx, t)
                return None
            return hook

        try:
            for li in norm_layers:
                handles.append(blocks[li].register_forward_hook(make_hook(li)))

            device = self._model_device()
            self.egra.model.eval()

            text = self._prompt_to_text(prompt)
            enc = self.egra.tokenizer(text, return_tensors="pt").to(device)
            enc.pop("token_type_ids", None)

            # Prefill (seq_len = prompt length)
            out = self.egra.model(**enc, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

            # Manual decode loop (seq_len == 1 each step)
            for _ in range(max_new_tokens):
                if temperature <= 0 or not do_sample:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

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
        vals = [rms_by_layer[i] for i in idxs if i in rms_by_layer]
        if not vals:
            raise ValueError("No RMS values found for requested layers (did you measure them?).")

        t = torch.tensor(vals, dtype=torch.float32)
        if agg == "median":
            return float(t.median().item())
        if agg == "mean":
            return float(t.mean().item())
        raise ValueError("agg must be 'median' or 'mean'")

    def alpha_from_target_std_fixed(
        self,
        target_residual_noise_std: float,
        rms_by_layer: Dict[int, float],
        layer_set: Sequence[int],
        *,
        agg: str = "median",
    ) -> float:
        """
        Computes alpha so that on the calibration model:
            target_residual_noise_std ~= alpha * agg_rms(layer_set)

        This keeps residual_noise_std independent of the number of layers (your condition).
        """
        a = self.aggregate_rms(rms_by_layer, layer_set, agg=agg)
        if a <= 0:
            raise ValueError("Non-positive aggregate RMS.")
        return float(target_residual_noise_std / a)

    def residual_std_for_new_model(
        self,
        alpha: float,
        rms_by_layer_new_model: Dict[int, float],
        layer_set: Sequence[int],
        *,
        agg: str = "median",
    ) -> float:
        """
        Given alpha learned on JAIS, compute residual_noise_std for a new model:
            residual_noise_std_new = alpha * agg_rms_new(layer_set)
        """
        a = self.aggregate_rms(rms_by_layer_new_model, layer_set, agg=agg)
        return float(alpha * a)