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


@torch.no_grad()
def collect_story_pcs(
    egra,
    prompt: Union[str, List[Dict[str, str]]],
    layers: Sequence[int],
    *,
    n_stories: int = 32,
    rank: int = 24,
    max_new_tokens: int = 120,
    skip_first: int = 4,
    temperature: float = 1.0,
    seed: Optional[int] = 0,
    verbose: bool = True,
) -> Dict[int, torch.Tensor]:
    """Directions along which one *story* differs from another.

    ``collect_block_pcs`` takes the principal components of individual decode-step
    activations, so its leading directions describe how one token position differs
    from another -- mid-word versus start-of-sentence, inside quotation marks
    versus outside. A constant offset along such a direction pushes the model
    toward a kind of token, which is not what we want.

    This function summarises each sampled story by the mean of its block outputs
    and takes the principal components *across stories*. What varies between those
    means is what varies between stories: who it is about, where it happens, how it
    ends. Pushing along one of them amplifies a way the model already varies, which
    is why it stays fluent at magnitudes that isotropic noise cannot survive.

    Returns a per-layer ``(hidden_size, rank)`` orthonormal basis, ordered by how
    much of the between-story variation each direction explains. ``rank`` is capped
    at ``n_stories - 1``: that is how many directions a set of ``n_stories`` points
    can span once it is centred.

    ``skip_first`` drops the opening decode steps, where every story is still
    writing the same first few words and the activations say nothing about which
    story this is.
    """
    blocks = egra._get_transformer_blocks()
    norm_layers = sorted({egra._normalize_layer_index(int(i), len(blocks)) for i in layers})

    # Per layer: the running sum of this story's decode-step activations, and the
    # count, so the mean is formed without holding every step in memory.
    cur: Dict[int, List[torch.Tensor]] = {li: [] for li in norm_layers}
    story_means: Dict[int, List[torch.Tensor]] = {li: [] for li in norm_layers}
    step = {"t": 0}
    handles = []

    def make_hook(li: int):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor) or t.dim() != 3 or t.shape[1] != 1:
                return None
            if step["t"] >= skip_first:
                cur[li].append(t[0, -1, :].detach().to("cpu", torch.float32))
            return None
        return hook

    try:
        for li in norm_layers:
            handles.append(blocks[li].register_forward_hook(make_hook(li)))

        device = egra._input_device()
        egra.model.eval()
        text = (
            egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if isinstance(prompt, (list, tuple))
            else prompt
        )
        enc = egra.tokenizer(text, return_tensors="pt").to(device)
        enc.pop("token_type_ids", None)

        for s in range(n_stories):
            if seed is not None:
                torch.manual_seed(seed + s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + s)
            for li in norm_layers:
                cur[li] = []
            step["t"] = 0

            out = egra.model(**enc, use_cache=True, return_dict=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            eos = egra.tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                probs = torch.softmax(
                    torch.nan_to_num(logits.float() / max(temperature, 1e-6), nan=0.0), dim=-1
                )
                if not torch.isfinite(probs).all() or probs.sum() <= 0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    nxt = torch.multinomial(probs / probs.sum(dim=-1, keepdim=True), 1)
                if eos is not None and int(nxt.item()) == int(eos):
                    break
                out = egra.model(input_ids=nxt, past_key_values=past, use_cache=True,
                                 return_dict=True)
                past, logits = out.past_key_values, out.logits[:, -1, :]
                step["t"] += 1

            for li in norm_layers:
                if cur[li]:
                    story_means[li].append(torch.stack(cur[li], dim=0).mean(dim=0))
            if verbose and (s + 1) % 8 == 0:
                print(f"  [story basis] {s + 1}/{n_stories} stories", flush=True)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    basis: Dict[int, torch.Tensor] = {}
    for li, means in story_means.items():
        if len(means) < 3:
            raise RuntimeError(f"only {len(means)} usable stories for layer {li}.")
        mat = torch.stack(means, dim=0)
        mat = mat - mat.mean(dim=0, keepdim=True)
        k = min(rank, mat.shape[0] - 1, mat.shape[1])
        _, sv, vh = torch.linalg.svd(mat, full_matrices=False)
        basis[li] = vh[:k].t().contiguous()
        if verbose and li == min(story_means):
            share = float((sv[:k] ** 2).sum() / (sv ** 2).sum().clamp_min(1e-12))
            print(f"  [story basis] rank {k} from {mat.shape[0]} stories; "
                  f"those directions carry {share:.0%} of the between-story variation")
    return basis
