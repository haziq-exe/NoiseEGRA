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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch


@dataclass
class StoryAxes:
    """How the model's own stories differ from one another, per layer.

    ``basis[layer]`` is a (hidden_size, rank) orthonormal matrix whose columns are
    the directions between-story variation runs along, ordered by how much of it
    each explains. ``mean[layer]`` is the average activation those directions are
    measured from: a story's position along an axis only means anything relative
    to it. ``explained`` is the share of between-story variation the kept
    directions carry, between 0 and 1.
    """
    basis: Dict[int, torch.Tensor]
    mean: Dict[int, torch.Tensor]
    # How far stories actually spread along each of those directions, in the
    # same order: the standard deviation of the sampled stories' positions.
    #
    # The basis alone says which way stories differ and not how much, and the
    # difference matters. Drawn uniformly on the sphere, a perturbation puts as
    # much weight on the last direction as on the first -- and the first carries
    # most of the between-story variation while the last carries almost none, so
    # a step of a given length along the last is a far larger departure from
    # anything the model does than the same step along the first. Weighting the
    # draw by this makes a perturbation of a given size look like a real
    # story-to-story difference instead.
    scale: Dict[int, torch.Tensor] = field(default_factory=dict)
    # The sampled stories themselves, centred: ``anchors[layer]`` is
    # (n_stories, hidden_size), one row per story, each the mean of that story's
    # decode-step activations minus ``mean[layer]``.
    #
    # The basis is a summary of these, and throwing them away costs something. A
    # displacement drawn in the span of the principal components is a mixture of
    # directions, and at a large enough magnitude it lands somewhere no story of
    # the model's own ever sat -- which is when the model stops writing a story
    # and starts refusing. A displacement aimed at one of these rows cannot: at
    # full magnitude it lands exactly on a state the model produced while
    # writing one.
    anchors: Dict[int, torch.Tensor] = field(default_factory=dict)
    explained: float = 0.0
    n_stories: int = 0


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
    push_plan=None,
    verbose: bool = True,
) -> "StoryAxes":
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

    Returns a :class:`StoryAxes`: the per-layer basis, and the mean activation the
    basis is centred on, which a method that amplifies a story's own deviation
    needs in order to know what it is deviating from. ``rank`` is capped at
    ``n_stories - 1``: that is how many directions a set of ``n_stories`` points
    can span once it is centred.

    ``skip_first`` drops the opening decode steps, where every story is still
    writing the same first few words and the activations say nothing about which
    story this is.

    ``push_plan`` samples the stories *under the constraint push* instead of
    from the untouched model. It matters for two reasons, both measured.

    The perturbation is applied during steered generation, so the cloud of
    states it is displacing within is the cloud of *steered* stories. Sampled
    without the push, the basis and the anchors describe a cloud the state is
    never in, and the displacement is measured from the wrong centre.

    And the sampled stories are what an anchored displacement aims *at*. Taken
    from the untouched model they are stories that break 4.3 requirements of
    twelve, so aiming at one drags the writing back towards breaking them: at
    one story's distance the anchored displacement scores 3.58 against the drawn
    one's 2.19, and it loses precisely on the two requirements the push is
    holding up. Under the push they are compliant stories, and aiming at one
    should cost nothing.

    The push is applied exactly as the generation hook applies it -- the same
    ``steering_only`` vector at the prompt positions if the plan steers there,
    the same spared head and tail, the same per-step vector while writing --
    so the states collected are the states the real run produces. The plan must
    carry no perturbation of its own; that is what is being measured.
    """
    if push_plan is not None and getattr(push_plan, "offset_gamma", 0.0):
        raise ValueError(
            "the plan used to sample the basis must not itself perturb: it is "
            "being used to find out what the unperturbed steered stories look "
            "like. Build it with offset_gamma=0."
        )
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
            if not isinstance(t, torch.Tensor) or t.dim() != 3:
                return None
            if t.shape[1] != 1:
                # The prompt, read in one pass. Push it exactly as the
                # generation hook does, sparing the same positions.
                if push_plan is not None and getattr(push_plan, "steer_prefill", False):
                    d = push_plan.steering_only(li, 0, device=t.device)
                    if d is not None:
                        keep = int(getattr(push_plan, "prompt_tail_clear", 0) or 0)
                        head = int(getattr(push_plan, "prompt_head_clear", 0) or 0)
                        n = t.shape[1]
                        lo = head if 0 < head < n else 0
                        hi = n - keep if 0 < keep < n - lo else n
                        d = d.to(t.dtype).view(1, 1, -1)
                        if hi > lo:
                            t[:, lo:hi, :].add_(d)
                        else:
                            t.add_(d)
                return None
            if push_plan is not None:
                d = push_plan.delta_for(li, step["t"], with_noise=False,
                                        with_offset=False, device=t.device)
                if d is not None:
                    t[:, -1:, :].add_(d.to(t.dtype).view(1, 1, -1))
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
    centre: Dict[int, torch.Tensor] = {}
    explained, n_used = 0.0, 0
    spread: Dict[int, torch.Tensor] = {}
    anchors: Dict[int, torch.Tensor] = {}
    for li, means in story_means.items():
        if len(means) < 3:
            raise RuntimeError(f"only {len(means)} usable stories for layer {li}.")
        mat = torch.stack(means, dim=0)
        mu = mat.mean(dim=0)
        centre[li] = mu.contiguous()
        mat = mat - mu.unsqueeze(0)
        anchors[li] = mat.contiguous()
        k = min(rank, mat.shape[0] - 1, mat.shape[1])
        _, sv, vh = torch.linalg.svd(mat, full_matrices=False)
        basis[li] = vh[:k].t().contiguous()
        # Singular values divided by sqrt(n-1) are the per-direction standard
        # deviations of the story positions this basis was built from.
        spread[li] = (sv[:k] / max(mat.shape[0] - 1, 1) ** 0.5).contiguous()
        n_used = mat.shape[0]
        if li == min(story_means):
            explained = float((sv[:k] ** 2).sum() / (sv ** 2).sum().clamp_min(1e-12))
            if verbose:
                print(f"  [story basis] rank {k} from {n_used} stories; those "
                      f"directions carry {explained:.0%} of the between-story variation")
    return StoryAxes(basis=basis, mean=centre, scale=spread, anchors=anchors,
                     explained=explained, n_stories=n_used)


@torch.no_grad()
def collect_prompt_pcs(
    egra,
    prompt: Union[str, List[Dict[str, str]]],
    layers: Sequence[int],
    *,
    rank: int = 24,
    skip_first: int = 4,
    verbose: bool = True,
) -> "StoryAxes":
    """Directions the residual stream varies along, from the prompt alone.

    One forward pass, no generation. The prompt is a few hundred tokens, and each
    of them has a hidden state at every layer; the principal components of those
    states are the directions this model's residual stream actually moves along
    while it is reading. That is enough to keep a perturbation on the manifold the
    model operates on, which is the property that matters -- an isotropic draw in
    4096 dimensions is almost entirely in directions the model never uses, so it
    has to be enormous before it changes anything, and by then fluency is gone.

    Compared with :func:`collect_story_pcs` this measures a different thing. Story
    axes are how one finished story differs from another, which needs stories to
    be sampled. Prompt axes are how one token position differs from another inside
    the instruction, which needs nothing but the instruction. The claim is only
    that both span the part of the space the model uses, not that they are the
    same directions.

    ``skip_first`` drops the opening positions, where the chat template's own
    boilerplate sits and every prompt looks alike.

    Returns a :class:`StoryAxes` so it is interchangeable with the sampled
    estimate: ``basis`` per layer, and the ``mean`` those directions are measured
    from, which the amplification variant needs.
    """
    blocks = egra._get_transformer_blocks()
    norm_layers = sorted({egra._normalize_layer_index(int(i), len(blocks)) for i in layers})
    caught: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(li: int):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.dim() == 3:
                caught[li] = t[0].detach().to("cpu", torch.float32)
            return None
        return hook

    try:
        for li in norm_layers:
            handles.append(blocks[li].register_forward_hook(make_hook(li)))

        text = (
            egra.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if isinstance(prompt, (list, tuple))
            else prompt
        )
        enc = egra.tokenizer(text, return_tensors="pt").to(egra._input_device())
        enc.pop("token_type_ids", None)
        egra.model.eval()
        egra.model(**enc, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    basis: Dict[int, torch.Tensor] = {}
    centre: Dict[int, torch.Tensor] = {}
    explained, n_pos = 0.0, 0
    for li in norm_layers:
        mat = caught.get(li)
        if mat is None or mat.shape[0] <= skip_first + 2:
            raise RuntimeError(
                f"layer {li} gave {0 if mat is None else mat.shape[0]} prompt positions; "
                "the prompt is too short to estimate anything from."
            )
        mat = mat[skip_first:]
        mu = mat.mean(dim=0)
        centre[li] = mu.contiguous()
        mat = mat - mu.unsqueeze(0)
        k = min(rank, mat.shape[0] - 1, mat.shape[1])
        _, sv, vh = torch.linalg.svd(mat, full_matrices=False)
        basis[li] = vh[:k].t().contiguous()
        n_pos = mat.shape[0]
        if li == norm_layers[0]:
            explained = float((sv[:k] ** 2).sum() / (sv ** 2).sum().clamp_min(1e-12))
            if verbose:
                print(f"  [prompt basis] rank {k} from {n_pos} prompt positions; those "
                      f"directions carry {explained:.0%} of the variation across them")
    return StoryAxes(basis=basis, mean=centre, explained=explained, n_stories=n_pos)
