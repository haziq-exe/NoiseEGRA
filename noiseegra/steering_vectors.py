"""Contrastive extraction of per-constraint steering directions.

Method is standard CAA / mean-difference (Rimsky et al., 2024): for each item we
teacher-force the model over a shared *prefix* followed by a ``positive`` and a
``negative`` continuation, read the residual stream (block output) at the
continuation positions only, average over positions, and take the difference.
Averaging that difference over items gives the direction for the constraint.

Two things are deliberate:

* The contrast is **within-item**: positive and negative share a prefix and differ
  only in the target property, so topic, names and length of the prefix cancel in
  the difference. This is what stops a "length" direction from secretly being a
  "talks about endings" topic direction.
* The context is the **real EGRA story prompt**, not a generic one. Steering
  vectors are known to degrade out of distribution, so the direction is measured
  in the same conversational context in which it will later be applied.

The injection site (block output, last position, decode steps) matches the
paper's L-Res method, so the steering vector and the noise live in the same basis
at the same layer and the orthogonality statement is well-posed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import re

import torch

from . import prompts

_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

DEFAULT_PAIRS_PATH = Path(__file__).resolve().parent / "data" / "steering_pairs_ar.json"


def load_pairs(path: Optional[str | Path] = None) -> Dict[str, Dict[str, object]]:
    """Load the bundled Arabic contrast pairs (or a custom JSON in the same shape)."""
    p = Path(path) if path is not None else DEFAULT_PAIRS_PATH
    with p.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw["constraints"]


def output_profile(pos: torch.Tensor, neg: torch.Tensor, *, floor: float = 1e-4,
                   eps: float = 1e-6) -> torch.Tensor:
    """Which next tokens rule-following text makes likelier than rule-breaking text.

    ``pos`` and ``neg`` are the model's next-token distributions averaged over
    the continuation positions of each side's texts. The profile is the
    log-ratio of the two, a mean contrastive difference taken in the output
    distribution rather than the residual stream: positive where the rule-
    following side predicts a token more, negative where the rule-breaking side
    does. Tokens neither side gives ``floor`` of probability are set to zero, so
    the ratio of two negligible numbers cannot dominate it.
    """
    pos = pos.float().clamp_min(0)
    neg = neg.float().clamp_min(0)
    out = (pos + eps).log() - (neg + eps).log()
    out[torch.maximum(pos, neg) < floor] = 0.0
    return out


# --------------------------------------------------------------------------- #
#  Container                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class SteeringVectorSet:
    """Per-constraint, per-layer directions plus their extraction diagnostics."""

    vectors: Dict[str, Dict[int, torch.Tensor]]
    components: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    # Mean activation of the *positive* side of each constraint's contrast pairs:
    # what text that satisfies the constraint looks like at this layer, rather
    # than which way to move to get there. Constant steering only needs the
    # difference; steering that corrects a story toward compliance needs to know
    # where compliance sits.
    positives: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    diagnostics: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)
    # Names whose direction the per-story perturbation must never move along.
    # A run-time choice, not something the extraction produces, so it is not
    # saved with the vectors: a cached file loads with nothing shielded.
    shield: List[str] = field(default_factory=list)
    # How many directions each shielded name contributes. 0 is the mean
    # difference alone, one direction. Above that, its top principal components
    # are added, so the shield is a subspace rather than a single axis.
    #
    # This matters because the mean alone is enough at a small perturbation and
    # not at a large one: at 0.15 it removed every refusal and every leaked
    # plan, and at 0.25 those failures came back in full. A region bounded in
    # many directions is not kept out of by naming one of them.
    shield_rank: int = 0
    # Which next tokens each constraint makes likelier, read off the same forward
    # passes the directions come from: the log-ratio of the model's average
    # next-token distribution over the continuation positions, rule-following
    # side against rule-breaking side, zero wherever neither side gives a token
    # real probability. One vector over the vocabulary per constraint. See
    # :meth:`output_profile`.
    output_profiles: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def names(self) -> List[str]:
        return sorted(self.vectors)

    @property
    def layers(self) -> List[int]:
        if not self.vectors:
            return []
        return sorted(next(iter(self.vectors.values())))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectors": {
                    c: {int(l): v.cpu() for l, v in per.items()}
                    for c, per in self.vectors.items()
                },
                "components": {
                    c: {int(l): v.cpu() for l, v in per.items()}
                    for c, per in self.components.items()
                },
                "positives": {
                    c: {int(l): v.cpu() for l, v in per.items()}
                    for c, per in self.positives.items()
                },
                "diagnostics": self.diagnostics,
                "meta": self.meta,
                "output_profiles": {c: v.cpu() for c, v in self.output_profiles.items()},
            },
            p,
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVectorSet":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            vectors={c: {int(l): v for l, v in per.items()} for c, per in blob["vectors"].items()},
            components={
                c: {int(l): v for l, v in per.items()}
                for c, per in blob.get("components", {}).items()
            },
            positives={
                c: {int(l): v for l, v in per.items()}
                for c, per in blob.get("positives", {}).items()
            },
            diagnostics=blob.get("diagnostics", {}),
            meta=blob.get("meta", {}),
            output_profiles=dict(blob.get("output_profiles", {}) or {}),
        )

    def profile_report(self, tokenizer, top: int = 12, bottom: int = 8) -> List[str]:
        """What each rule's output profile favours and disfavours, as lines.

        So a profile can be read before anything is tilted by it: dialogue should
        favour quotation marks and speech verbs, a simile "like" and "as".
        """
        lines = []
        for name, prof in sorted(self.output_profiles.items()):
            up = torch.topk(prof, min(top, prof.numel())).indices.tolist()
            down = torch.topk(-prof, min(bottom, prof.numel())).indices.tolist()
            show = lambda ids: " ".join(repr(tokenizer.decode([i])) for i in ids)
            lines.append(f"  {name:18s} output profile favours {show(up)}")
            lines.append(f"  {'':18s} and disfavours {show(down)}")
        return lines

    def output_profile(self, names: Sequence[str]) -> Optional[torch.Tensor]:
        """One vector over the vocabulary for a set of constraints.

        Each constraint's profile is scaled to unit length first, so a rule whose
        pairs happen to differ in more tokens does not outweigh the rest, and
        the scaled profiles are summed. ``None`` if none of them has one.
        """
        have = [self.output_profiles[n] for n in names if n in self.output_profiles]
        if not have:
            return None
        out = torch.zeros_like(have[0], dtype=torch.float32)
        for v in have:
            v = v.float()
            out += v / v.norm().clamp_min(1e-12)
        return out

    def protect_extra(self, names: Sequence[str], rank: int) -> Dict[int, torch.Tensor]:
        """Per-layer matrix of extra directions to shield the noise from.

        Concatenates the top-``rank`` principal components of each named
        constraint's per-item difference matrix. Use this to enlarge the protected
        subspace beyond the ``C`` mean directions -- with a 4096-d residual stream,
        protecting only 3 directions removes 0.07% of a random draw's energy and
        the ``orth`` arm is indistinguishable from isotropic noise.
        """
        if rank <= 0:
            return {}
        out: Dict[int, torch.Tensor] = {}
        for name in names:
            per = self.components.get(name)
            if not per:
                raise KeyError(
                    f"no principal components stored for '{name}'; "
                    "re-extract with pca_rank >= the rank you want to protect."
                )
            for layer, comp in per.items():
                take = comp[:, :rank]
                out[layer] = take if layer not in out else torch.cat([out[layer], take], dim=1)
        return out

    def shielded_subspace(self, names: Sequence[str],
                          rank: int) -> Optional[Dict[int, torch.Tensor]]:
        """Every direction the perturbation is to be projected clear of.

        Two different things end up in the same protected subspace.

        The first is ``protect_extra``: extra principal components of the
        *steered* constraints, which exist so that a perturbation cannot undo by
        accident the constraint the push is applying.

        The second is ``shield``, and it is not a constraint at all. The
        perturbation's measured cost is coherence, and reading the stories it
        breaks shows why: the failures are not garbled prose but the model
        leaving the story altogether -- refusing, or narrating its own planning
        ("The user wants a short story ..."). A large enough perturbation in an
        arbitrary direction moves the state out of the register that tells
        stories and into the one that talks about the task. Naming that axis and
        removing it from the subspace the perturbation is drawn from forbids the
        move, at the cost of one direction out of a few dozen.

        Nothing here is pushed. The steering basis is untouched, so the budget,
        the dose along every constraint and the text the push produces are all
        unchanged; only what the perturbation is allowed to do changes.
        """
        out: Dict[int, torch.Tensor] = {}
        if rank > 0:
            out = {l: v.clone() for l, v in self.protect_extra(names, rank).items()}
        for name in self.shield:
            per = self.vectors.get(name)
            if not per:
                raise KeyError(
                    f"no direction extracted for the shielded name '{name}'; "
                    "it has to be extracted even though it is never pushed."
                )
            comps = self.components.get(name) if self.shield_rank > 0 else None
            if self.shield_rank > 0 and not comps:
                raise KeyError(
                    f"shield rank {self.shield_rank} was asked for but no principal "
                    f"components are stored for '{name}'; re-extract with a pca_rank "
                    "at least that large."
                )
            for layer, vec in per.items():
                cols = [vec.detach().to(torch.float32).reshape(-1, 1)]
                if comps is not None:
                    have = comps[layer].shape[1]
                    if have < self.shield_rank:
                        raise ValueError(
                            f"shield rank {self.shield_rank} was asked for but only "
                            f"{have} components are stored for '{name}' at layer {layer}."
                        )
                    cols.append(comps[layer][:, :self.shield_rank].detach()
                                .to(torch.float32))
                col = torch.cat(cols, dim=1)
                out[layer] = col if layer not in out else torch.cat([out[layer], col], dim=1)
        return out or None

    def print_report(self) -> None:
        print("=== Steering Vector Extraction ===")
        for key in ("model", "n_layers", "pairs_path", "pca_rank"):
            if key in self.meta:
                print(f"{key}: {self.meta[key]}")
        print(f"{'constraint':>16} {'layer':>6} {'|d|':>10} {'|d|/RMS':>9} {'consistency':>12} "
              f"{'n':>4} {'word gap':>9}")
        for name in self.names:
            for layer in sorted(self.diagnostics.get(name, {})):
                d = self.diagnostics[name][layer]
                print(
                    f"{name:>16} {layer:>6} {d['norm']:>10.4f} {d['norm_over_rms']:>9.4f} "
                    f"{d['consistency']:>12.4f} {int(d['n_items']):>4} "
                    f"{d.get('mean_word_delta', float('nan')):>+9.2f}"
                )
        print(
            "\nconsistency = mean cosine between each item's difference vector and the\n"
            "pooled direction. Near 0 means the contrast set did not isolate a shared\n"
            "direction and the steering vector is mostly noise; > ~0.3 is a usable signal.\n"
            "word gap = mean (positive - negative) word count over the pairs. Away from\n"
            "0 the wording itself is length-confounded; the read window is matched to the\n"
            "shorter side either way, so this is residual imbalance only."
        )


# --------------------------------------------------------------------------- #
#  Extractor                                                                   #
# --------------------------------------------------------------------------- #

class SteeringVectorExtractor:
    """Reads contrastive activations out of an :class:`~noiseegra.EGRA_functions.EGRA`."""

    def __init__(self, egra):
        self.egra = egra

    # -- helpers ---------------------------------------------------------- #

    def _device(self) -> torch.device:
        return next(self.egra.model.parameters()).device

    def _context_ids(self, system: str, user: str) -> List[int]:
        """Token ids for the chat context up to (and including) the generation prompt.

        Uses ``egra.apply_chat_template`` rather than the tokenizer directly so
        that models with hand-written templates (AceGPT) are handled correctly,
        and tokenises exactly the way ``EGRA.generate`` does so the activations
        are collected in the same context the model will generate in.
        """
        text = self.egra.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(self.egra.tokenizer(text)["input_ids"])

    def _piece_ids(self, text: str) -> List[int]:
        return list(self.egra.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def _segment_means(
        self,
        input_ids: List[int],
        start: int,
        layers: Sequence[int],
        count: Optional[int] = None,
        window: str = "all",
        window_tokens: int = 4,
        with_output: bool = False,
    ):
        """Mean block output over the continuation positions, for each layer.

        With ``with_output`` it also returns the model's next-token distribution
        averaged over the predictions of those same positions -- read off the
        logits of the same forward pass, so it costs nothing extra.

        ``window`` chooses which of the continuation positions are averaged.
        ``all`` takes every one of them, which mixes the position where the
        property is decided with the content that follows it. ``first`` takes the
        opening ``window_tokens``, where "walks" or "walked" is actually chosen --
        this is closer to standard CAA, which reads at the single token carrying
        the decision. ``last`` takes the final position, which encodes having
        written the whole continuation.

        ``count`` caps how many positions after ``start`` are averaged. The caller
        passes the shorter of the two continuations' token counts, so the positive
        and negative sides are read over exactly the same number of positions. Left
        open, the mean over a longer continuation is systematically different from
        the mean over a shorter one -- later positions carry more accumulated
        context -- and the difference vector picks that up as if it were the
        property being contrasted.
        """
        blocks = self.egra._get_transformer_blocks()
        norm_layers = sorted(
            {self.egra._normalize_layer_index(int(i), len(blocks)) for i in layers}
        )

        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
                    return None
                stop = tensor.shape[1] if count is None else min(start + count, tensor.shape[1])
                seg = tensor[0, start:stop, :]
                if window == "first":
                    seg = seg[:max(window_tokens, 1)]
                elif window == "last":
                    seg = seg[-1:]
                if seg.shape[0] == 0:
                    return None
                captured[layer_idx] = seg.detach().to("cpu", torch.float32).mean(dim=0)
                return None
            return hook

        try:
            for li in norm_layers:
                handles.append(blocks[li].register_forward_hook(make_hook(li)))

            ids = torch.tensor([input_ids], dtype=torch.long, device=self._device())
            self.egra.model.eval()
            res = self.egra.model(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        missing = [li for li in norm_layers if li not in captured]
        if missing:
            raise RuntimeError(f"no activations captured for layers {missing}.")
        if not with_output:
            return captured
        # The logits at position i predict token i+1, so the continuation's own
        # tokens are predicted from start-1 onward. Same positions, same window.
        n = len(input_ids)
        stop = n if count is None else min(start + count, n)
        rows = torch.arange(max(start - 1, 0), max(stop - 1, 0))
        if window == "first":
            rows = rows[:max(window_tokens, 1)]
        elif window == "last":
            rows = rows[-1:]
        logits = res.logits[0, rows.to(res.logits.device), :].float()
        probs = torch.softmax(logits, dim=-1).mean(dim=0).to("cpu")
        return captured, probs

    # -- public ----------------------------------------------------------- #

    @torch.no_grad()
    def extract(
        self,
        constraints: Mapping[str, Mapping[str, object]],
        layers: Sequence[int],
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        pca_rank: int = 8,
        only: Optional[Sequence[str]] = None,
        window: str = "all",
        window_tokens: int = 4,
        verbose: bool = True,
    ) -> SteeringVectorSet:
        """Build a :class:`SteeringVectorSet` from contrast pairs.

        ``constraints`` maps a constraint name to ``{"pairs": [{prefix, positive,
        negative}, ...]}`` -- the shape returned by :func:`load_pairs`.
        """
        system = prompts.SYS_ZERO_SHOT if system is None else system
        user = prompts.PROMPT_ZERO_SHOT if user is None else user

        context = self._context_ids(system, user)
        names = list(constraints) if only is None else [n for n in only]

        vectors: Dict[str, Dict[int, torch.Tensor]] = {}
        components: Dict[str, Dict[int, torch.Tensor]] = {}
        positives: Dict[str, Dict[int, torch.Tensor]] = {}
        diagnostics: Dict[str, Dict[int, Dict[str, float]]] = {}
        output_profiles: Dict[str, torch.Tensor] = {}

        for name in names:
            if name not in constraints:
                raise KeyError(f"unknown constraint '{name}'; have {sorted(constraints)}")
            pairs = list(constraints[name]["pairs"])
            if not pairs:
                raise ValueError(f"constraint '{name}' has no pairs.")

            if verbose:
                print(f"[steering] extracting '{name}' from {len(pairs)} pairs ...")

            per_item: Dict[int, List[torch.Tensor]] = {}
            pos_items: Dict[int, List[torch.Tensor]] = {}
            out_sum: Dict[str, Optional[torch.Tensor]] = {"positive": None, "negative": None}
            act_sq: Dict[int, List[float]] = {}
            tok_deltas: List[int] = []
            word_deltas: List[int] = []

            for item in pairs:
                prefix_ids = self._piece_ids(item.get("prefix", ""))
                start = len(context) + len(prefix_ids)

                ids = {}
                for key in ("positive", "negative"):
                    ids[key] = self._piece_ids(item[key])
                    if not ids[key]:
                        raise ValueError(f"empty '{key}' continuation in constraint '{name}'.")
                # Both sides are read over the same number of positions, so the
                # difference cannot encode "one continuation is longer".
                matched = min(len(ids["positive"]), len(ids["negative"]))
                tok_deltas.append(len(ids["positive"]) - len(ids["negative"]))
                word_deltas.append(
                    len(_WORDS.findall(item["positive"])) - len(_WORDS.findall(item["negative"]))
                )

                side: Dict[str, Dict[int, torch.Tensor]] = {}
                for key in ("positive", "negative"):
                    side[key], probs = self._segment_means(
                        context + prefix_ids + ids[key], start, layers, count=matched,
                        window=window, window_tokens=window_tokens, with_output=True,
                    )
                    out_sum[key] = probs if out_sum[key] is None else out_sum[key] + probs

                for layer, pos_vec in side["positive"].items():
                    neg_vec = side["negative"][layer]
                    per_item.setdefault(layer, []).append(pos_vec - neg_vec)
                    pos_items.setdefault(layer, []).append(pos_vec)
                    act_sq.setdefault(layer, []).extend(
                        [float(pos_vec.pow(2).mean().item()), float(neg_vec.pow(2).mean().item())]
                    )

            vectors[name] = {}
            components[name] = {}
            positives[name] = {}
            diagnostics[name] = {}
            output_profiles[name] = output_profile(
                out_sum["positive"] / len(pairs), out_sum["negative"] / len(pairs))

            for layer, diffs in per_item.items():
                mat = torch.stack(diffs, dim=0)          # (n_items, dim)
                mean_vec = mat.mean(dim=0)               # (dim,)
                rms = float(torch.tensor(act_sq[layer]).mean().sqrt().item())

                unit = mean_vec / mean_vec.norm().clamp_min(1e-12)
                consistency = float(
                    torch.nn.functional.cosine_similarity(
                        mat, unit.unsqueeze(0).expand_as(mat), dim=1
                    ).mean().item()
                )

                vectors[name][layer] = mean_vec
                positives[name][layer] = torch.stack(pos_items[layer], dim=0).mean(dim=0)
                diagnostics[name][layer] = {
                    "norm": float(mean_vec.norm().item()),
                    "norm_over_rms": float(mean_vec.norm().item() / max(rms, 1e-12)),
                    "consistency": consistency,
                    "activation_rms": rms,
                    "n_items": float(mat.shape[0]),
                    "mean_word_delta": (sum(word_deltas) / len(word_deltas)) if word_deltas else 0.0,
                    "max_abs_word_delta": float(max((abs(d) for d in word_deltas), default=0)),
                    "mean_token_delta": (sum(tok_deltas) / len(tok_deltas)) if tok_deltas else 0.0,
                    "read_window": "matched",
                }

                if pca_rank > 0:
                    # Uncentred SVD: the leading component tracks the mean direction
                    # and the rest span the spread of the contrast, which is what we
                    # want to shield the noise from.
                    k = min(pca_rank, mat.shape[0], mat.shape[1])
                    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
                    components[name][layer] = vh[:k].t().contiguous()

        out = SteeringVectorSet(
            vectors=vectors,
            components=components,
            positives=positives,
            output_profiles=output_profiles,
            diagnostics=diagnostics,
            meta={
                "model": getattr(self.egra.model, "name_or_path", None)
                or getattr(getattr(self.egra.model, "config", None), "_name_or_path", "unknown"),
                "layers": sorted({int(i) for i in layers}),
                "n_layers": len(self.egra._get_transformer_blocks()),
                "pca_rank": pca_rank,
                "read_window": window,
                "read_window_tokens": window_tokens,
                "system_prompt": system,
                "user_prompt": user,
            },
        )
        if verbose:
            out.print_report()
            for line in out.profile_report(self.egra.tokenizer):
                print(line, flush=True)
        return out
