"""Estimate the directions a model's residual stream actually varies along.

A per-generation offset drawn isotropically in 4096 dimensions is almost entirely
in directions the model never uses, so it has to be enormous before it changes the
story, and by then it has broken fluency. Drawing it instead from the top
principal components of the model's own block outputs keeps the perturbation
in-distribution: the same magnitude produces a real stylistic shift without
leaving the manifold the model operates on.

It also makes the constraint-subspace projection matter. Constraint directions are
themselves high-variance behavioural directions, so they sit inside the top
components. A random offset drawn from that subspace overlaps the constraint
directions far more than an isotropic one would, which is exactly the situation
where projecting them out changes the outcome.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch


@torch.no_grad()
def collect_block_pcs(
    egra,
    prompts: Sequence[Union[str, List[Dict[str, str]]]],
    layers: Sequence[int],
    *,
    rank: int = 64,
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> Dict[int, torch.Tensor]:
    """Top-``rank`` principal components of each layer's block output.

    Samples decode-step activations across several prompts, centres them, and
    returns a per-layer (hidden_size, rank) orthonormal basis ordered by variance.
    Centring matters: the uncentred mean is a large always-on direction that
    carries no information about how outputs differ from each other.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    blocks = egra._get_transformer_blocks()
    norm_layers = sorted({egra._normalize_layer_index(int(i), len(blocks)) for i in layers})
    samples: Dict[int, List[torch.Tensor]] = {li: [] for li in norm_layers}
    handles = []

    def make_hook(li: int):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor) or t.dim() != 3 or t.shape[1] != 1:
                return None
            samples[li].append(t[0, -1, :].detach().to("cpu", torch.float32))
            return None
        return hook

    try:
        for li in norm_layers:
            handles.append(blocks[li].register_forward_hook(make_hook(li)))

        device = egra._input_device()
        egra.model.eval()

        for i, prompt in enumerate(prompts):
            text = (
                egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                if isinstance(prompt, (list, tuple))
                else prompt
            )
            enc = egra.tokenizer(text, return_tensors="pt").to(device)
            enc.pop("token_type_ids", None)

            out = egra.model(**enc, use_cache=True, return_dict=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]

            for _ in range(max_new_tokens):
                probs = torch.softmax(
                    torch.nan_to_num(logits.float() / max(temperature, 1e-6), nan=0.0), dim=-1
                )
                if not torch.isfinite(probs).all() or probs.sum() <= 0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    nxt = torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), 1)
                out = egra.model(input_ids=nxt, past_key_values=past, use_cache=True,
                                 return_dict=True)
                past, logits = out.past_key_values, out.logits[:, -1, :]

            if verbose:
                print(f"  [activation basis] prompt {i + 1}/{len(prompts)}", flush=True)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    basis: Dict[int, torch.Tensor] = {}
    for li, vecs in samples.items():
        if len(vecs) < 2:
            raise RuntimeError(f"only {len(vecs)} activation samples for layer {li}.")
        mat = torch.stack(vecs, dim=0)
        mat = mat - mat.mean(dim=0, keepdim=True)
        k = min(rank, mat.shape[0], mat.shape[1])
        _, _, vh = torch.linalg.svd(mat, full_matrices=False)
        basis[li] = vh[:k].t().contiguous()
    if verbose:
        li = norm_layers[0]
        print(f"  [activation basis] rank {basis[li].shape[1]} from "
              f"{len(samples[li])} decode steps per layer")
    return basis
