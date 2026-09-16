"""Subspace geometry for orthogonal constraint steering + direction-constrained noise.

This module is deliberately free of any model dependency beyond ``torch`` so that
the geometry can be unit-tested on CPU without loading a language model.

Two ideas live here:

1. **Orthogonalising the steering directions against each other.** Constraint
   directions recovered by contrastive mean-difference are *not* orthogonal (a
   story that concludes is also a shorter story), so naively summing them lets
   one constraint's push leak into another. We support three treatments so the
   effect of orthogonalisation is itself an ablation axis:

   ``none``          use the raw unit directions (naive sum baseline)
   ``gram_schmidt``  sequential QR; order-dependent, the first constraint keeps
                     its direction intact and the last is amputated the most
   ``lowdin``        symmetric (Loewdin/ZCA) orthonormalisation ``D (D^T D)^-1/2``;
                     order-free and the minimum-change orthogonal basis

2. **Constraining where the noise is allowed to live.** Given an orthonormal
   basis ``Q`` for the protected subspace, a standard normal draw ``g`` splits
   into ``g = QQ^T g + (I - QQ^T) g``. We can keep either half:

   ``iso``   isotropic noise (the published L-Res method)
   ``orth``  noise in the orthogonal complement of the steering subspace
   ``para``  noise confined *to* the steering subspace (the destructive control)

   Note the dimension counting: in a ``dim``-dimensional residual stream a random
   direction already puts only ``k/dim`` of its energy inside a ``k``-dimensional
   subspace. For ``k=3`` and ``dim=4096`` that is 0.07%, so ``orth`` is nearly a
   no-op *by construction* unless the protected subspace is enlarged (see
   ``protect_extra``). ``para`` is where the large effect lives, and with
   ``norm_match="energy"`` it is amplified by ``sqrt(dim/k)`` so that it carries
   the same total energy as the isotropic arm and the comparison isolates
   *direction* rather than *magnitude*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

# Reuse the paper's decay curve verbatim so schedules stay comparable.
from .EGRA_functions import _cosine_noise_decay

ORTHOGONALIZE_METHODS = ("none", "gram_schmidt", "lowdin")
NOISE_MODES = ("none", "iso", "orth", "para")
OFFSET_MODES = ("none", "orth", "free")
OFFSET_NORMS = ("energy", "raw")
# How f(S_c) perturbs the constraint vector itself. See SteeringPlan.jitter_*.
JITTER_MODES = ("none", "perp", "rotate", "gain", "frame")
JITTER_DRAWS = ("iso", "basis")
# How the constraint push is decided. See SteeringPlan.steer_mode.
STEER_MODES = ("constant", "feedback")
NORM_MATCH_MODES = ("energy", "none")
SCHEDULES = ("constant", "cosine_decay", "ramp", "linear_decay", "prefix")


# --------------------------------------------------------------------------- #
#  Scalar schedules                                                            #
# --------------------------------------------------------------------------- #

def schedule_factor(kind: str, t: int, horizon: int) -> float:
    """Multiplier in [0, 1] applied to a steering/noise magnitude at decode step ``t``.

    ``constant``      1.0 everywhere.
    ``cosine_decay``  the paper's cosine decay: 1.0 at t=0, 0.0 at t>=horizon.
    ``ramp``          0.0 at t=0 rising linearly to 1.0 at t>=horizon. Useful for
                      the closure direction: push the model to wrap up *more* the
                      longer the story has run.
    ``linear_decay``  1.0 at t=0 falling linearly to 0.0 at t>=horizon.
    ``prefix``        1.0 for the first ``horizon`` decode steps and 0.0 after.
                      A story's premise -- who it is about, where it happens,
                      what goes wrong -- is chosen in its first few dozen tokens.
                      Perturbing after that cannot change the premise and can only
                      cost grammar, and because every perturbed step is written to
                      the KV cache and read by every later step, the damage
                      compounds. Confining the perturbation to the opening keeps
                      the branch point and drops the compounding.
    """
    if kind == "constant":
        return 1.0
    if horizon <= 0:
        return 0.0
    if kind == "cosine_decay":
        return _cosine_noise_decay(t, horizon)
    if kind == "prefix":
        return 1.0 if t < horizon else 0.0
    frac = min(t, horizon) / horizon
    if kind == "ramp":
        return float(frac)
    if kind == "linear_decay":
        return float(1.0 - frac)
    raise ValueError(f"schedule must be one of {SCHEDULES}, got '{kind}'.")


# --------------------------------------------------------------------------- #
#  Direction geometry                                                          #
# --------------------------------------------------------------------------- #

def unit_columns(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalise each column of a (dim, C) matrix."""
    norms = mat.norm(dim=0, keepdim=True).clamp_min(eps)
    return mat / norms


def cosine_gram(mat: torch.Tensor) -> torch.Tensor:
    """(C, C) matrix of pairwise cosine similarities between the columns."""
    u = unit_columns(mat.float())
    return u.t() @ u


def orthonormalize(
    mat: torch.Tensor,
    method: str = "lowdin",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return an orthogonalised (dim, C) basis plus a diagnostic report.

    The returned columns always have unit norm and are sign-aligned with the
    corresponding input column, so column ``i`` remains *the* direction for
    constraint ``i`` and a positive coefficient still means "more of it".

    ``report['retained']`` gives, per constraint, the cosine between the original
    direction and its orthogonalised replacement. A value near 1.0 means
    orthogonalisation barely changed that constraint; a small value means the
    constraint was largely entangled with the others and most of its direction
    was removed. This number should be reported in any paper that uses the
    orthogonal variant.
    """
    if method not in ORTHOGONALIZE_METHODS:
        raise ValueError(f"method must be one of {ORTHOGONALIZE_METHODS}, got '{method}'.")

    raw = mat.float()
    if raw.dim() != 2:
        raise ValueError(f"expected a 2-D (dim, C) matrix, got shape {tuple(raw.shape)}.")

    d_hat = unit_columns(raw)
    gram = d_hat.t() @ d_hat
    n_cols = d_hat.shape[1]

    report: Dict[str, object] = {
        "method": method,
        "cosine_gram": gram.clone(),
        "max_offdiag_cosine": float(
            (gram - torch.eye(n_cols, dtype=gram.dtype, device=gram.device)).abs().max().item()
        )
        if n_cols > 1
        else 0.0,
    }

    if method == "none":
        basis = d_hat
    elif method == "gram_schmidt":
        q, r = torch.linalg.qr(d_hat, mode="reduced")
        # QR is sign-ambiguous; flip so q[:, i] points the same way as d_hat[:, i].
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1.0
        basis = q * signs.unsqueeze(0)
        report["qr_r_diag"] = torch.diagonal(r).abs().clone()
    else:  # lowdin
        # S = D (D^T D)^{-1/2}, computed through the eigendecomposition of the
        # (C, C) Gram matrix -- C is tiny so this is free.
        evals, evecs = torch.linalg.eigh(gram)
        report["gram_min_eigenvalue"] = float(evals.min().item())
        if float(evals.min().item()) <= 1e-8:
            raise ValueError(
                "Steering directions are (near-)linearly dependent "
                f"(min Gram eigenvalue = {evals.min().item():.3e}); "
                "symmetric orthonormalisation is undefined. Drop a constraint or "
                "use method='gram_schmidt'."
            )
        inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.t()
        basis = d_hat @ inv_sqrt

    basis = unit_columns(basis)
    report["retained"] = (d_hat * basis).sum(dim=0).clone()
    return basis, report


def orthonormal_basis(mat: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """Orthonormal basis for the column span of ``mat``, dropping rank-deficient columns.

    The tolerance has to sit well above float32 SVD noise. After projecting a
    subspace out of another, the removed directions come back with singular values
    around 1e-6 relative; at a 1e-6 cutoff one of them survives, and its left
    singular vector is arbitrary -- in practice it lands almost entirely inside the
    subspace that was just removed.
    """
    a = mat.float()
    u, s, _ = torch.linalg.svd(a, full_matrices=False)
    keep = s > (tol * s.max().clamp_min(1e-30))
    return u[:, keep].contiguous()


def complement_basis(mat: torch.Tensor, protect: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for span(mat) with span(protect) removed.

    Projects, orthonormalises, then projects and orthonormalises again. One pass
    leaves float32 residue that the second removes -- the same reason classical
    Gram-Schmidt is run twice.
    """
    out = mat
    for _ in range(2):
        out = orthonormal_basis(out - protect @ (protect.t() @ out))
    return out


# --------------------------------------------------------------------------- #
#  Noise sampling constrained to / away from a subspace                        #
# --------------------------------------------------------------------------- #

def split_noise(
    g: torch.Tensor,
    basis: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``g`` (..., dim) into (in-subspace, orthogonal-complement) components."""
    if basis is None or basis.numel() == 0:
        return torch.zeros_like(g), g
    coeff = g @ basis            # (..., k)
    parallel = coeff @ basis.t()  # (..., dim)
    return parallel, g - parallel


def constrained_noise(
    g: torch.Tensor,
    sigma: float,
    basis: Optional[torch.Tensor],
    mode: str = "iso",
    norm_match: str = "energy",
) -> Optional[torch.Tensor]:
    """Scale a standard-normal draw ``g`` into the requested noise arm.

    ``g`` must already be ~N(0, I) so that all three arms consume exactly one draw
    of the same shape -- that keeps the RNG stream aligned across conditions when
    a shared per-story seed is used.

    With ``norm_match='energy'`` the ``orth`` and ``para`` arms are rescaled so
    their expected squared norm matches the isotropic arm's ``sigma^2 * dim``.
    Without it, ``para`` carries only ``k/dim`` of the isotropic energy and the
    arms are not comparable.
    """
    if mode not in NOISE_MODES:
        raise ValueError(f"mode must be one of {NOISE_MODES}, got '{mode}'.")
    if norm_match not in NORM_MATCH_MODES:
        raise ValueError(f"norm_match must be one of {NORM_MATCH_MODES}, got '{norm_match}'.")
    if mode == "none" or sigma <= 0:
        return None
    if mode == "iso" or basis is None or basis.numel() == 0:
        return g * sigma

    dim = g.shape[-1]
    k = basis.shape[1]
    parallel, orthogonal = split_noise(g, basis)

    if mode == "para":
        if k >= dim:
            raise ValueError("protected subspace spans the full residual stream.")
        scale = math.sqrt(dim / k) if norm_match == "energy" else 1.0
        return parallel * (sigma * scale)

    # mode == "orth"
    if k >= dim:
        raise ValueError("protected subspace spans the full residual stream.")
    scale = math.sqrt(dim / (dim - k)) if norm_match == "energy" else 1.0
    return orthogonal * (sigma * scale)


# --------------------------------------------------------------------------- #
#  Per-layer plan                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class ConstraintSpec:
    """One steered constraint: which direction to use and how hard to push it."""
    name: str
    beta: float = 1.0
    schedule: str = "constant"

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got '{self.schedule}'.")


@dataclass
class LayerPlan:
    layer: int
    basis: torch.Tensor                    # (dim, C) unit steering directions
    protect: Optional[torch.Tensor]        # (dim, k) orthonormal protected basis
    report: Dict[str, object] = field(default_factory=dict)
    offset_basis: Optional[torch.Tensor] = None   # (dim, M) directions offsets may use
    offset: Optional[torch.Tensor] = None         # (dim,) this generation's offset
    amp_basis: Optional[torch.Tensor] = None      # (dim, r) directions stories differ along
    amp_mean: Optional[torch.Tensor] = None       # (dim,) what they differ *from*
    jitter_basis: Optional[torch.Tensor] = None   # (dim, M) directions f(S_c) may use
    jitter: Optional[torch.Tensor] = None         # (dim,) this generation's unit draw
    raw_basis: Optional[torch.Tensor] = None      # (dim, C) before orthogonalisation
    target: Optional[torch.Tensor] = None         # (C,) where compliant text sits

    def steering_delta(
        self,
        t: int,
        horizon: int,
        specs: Sequence[ConstraintSpec],
        rms_scale: float,
        gains: Optional[Sequence[float]] = None,
        budget: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Sum_c beta_c * g_c * f_c(t) * rms_scale * s_c  ->  a single (dim,) vector.

        ``gains`` are this generation's per-constraint multipliers, 1.0 each unless
        the plan is reallocating the mix per story.

        ``budget``, when given, is the total length of the push in units of the
        model's own activation scale, and the coefficients are renormalised to meet
        it. Without it, steering ``k`` constraints at coefficient ``beta`` gives a
        push of length ``beta * sqrt(k) * rms``, so *adding a constraint silently
        raises the dose*. That is not a detail: one direction at 3 is a push of
        4.52 and helps, two directions at 3 is a push of 6.39 and does not, and a
        single direction at 4.5 is a push of 6.78 and breaks the text outright. The
        two-direction arm was never compared against the one-direction arm at the
        same strength. With a budget, adding a constraint redistributes the push
        instead of enlarging it, and the number of constraints and the strength of
        the intervention stop being the same knob.
        """
        weights = [
            spec.beta * (1.0 if gains is None else float(gains[i]))
            for i, spec in enumerate(specs)
        ]
        if budget is not None:
            norm = math.sqrt(sum(w * w for w in weights))
            if norm <= 1e-12:
                return None
            weights = [w / norm * budget for w in weights]
        coeffs = torch.tensor(
            [
                w * schedule_factor(spec.schedule, t, horizon) * rms_scale
                for w, spec in zip(weights, specs)
            ],
            dtype=self.basis.dtype,
            device=self.basis.device,
        )
        if bool((coeffs == 0).all()):
            return None
        return self.basis @ coeffs


@dataclass
class SteeringPlan:
    """Everything the generation hook needs, precomputed once per run."""
    layers: List[int]
    specs: List[ConstraintSpec]
    layer_plans: Dict[int, LayerPlan]
    dim: int
    rms_scale: float
    orthogonalize: str
    noise_mode: str
    noise_alpha: float
    noise_norm_match: str
    noise_schedule: str
    offset_gamma: float = 0.0
    offset_mode: str = "none"
    # How ``offset_gamma`` is read. ``energy`` scales the drawn offset to a fixed
    # length, ``gamma * rms_scale * sqrt(dim)``, which is the expected length of
    # an isotropic noise draw at ``noise_alpha = gamma``. That makes gamma and
    # alpha the same dimensionless quantity: the perturbation's length as a
    # fraction of the hidden state's own length. ``raw`` keeps the historical
    # behaviour, where the drawn vector was used at its natural length and gamma
    # therefore meant something different for every subspace rank.
    offset_norm: str = "energy"
    # Which set of directions the offset was drawn from, kept so two conditions
    # that differ only in that cannot collide on disk. "step" is the principal
    # components of individual decode-step activations, "story" the components of
    # whole-story mean activations.
    offset_basis_kind: str = "step"
    # Whether the per-story offset is added during decode at all. Turning it off
    # while `offset_prefill` is on gives a prompt-only offset: the model is moved
    # somewhere else before it writes a token and then decodes completely
    # unperturbed, so a large shift costs no fluency.
    offset_decode: bool = True
    # Whether the per-story offset is also added to the prompt positions during
    # prefill. With it on, the offset shifts how the model reads the instruction
    # before it writes a single token, so the story starts somewhere else without
    # any per-token jitter at all.
    offset_prefill: bool = False
    # Amplification of a story's own deviation from the average story. At every
    # perturbed site the component of the current state that lies in the
    # between-story subspace is multiplied by this factor:
    #
    #     h  ->  h + (lambda - 1) * B B^T (h - mu)
    #
    # 1.0 is a no-op. Unlike an offset this adds nothing random: it pushes each
    # story further along whatever direction it was already taking, so two stories
    # that had started to diverge are driven apart rather than jointly displaced.
    # B has the constraint directions projected out, so the components the
    # requirements depend on are left at their original size.
    amplify_lambda: float = 1.0
    amplify_prefill: bool = False
    # --- f(S_c): perturb the constraint vector itself ---------------------- #
    #
    # Everything above adds a perturbation *beside* the constraint push and then
    # works to keep the two apart -- the noise is projected out of the constraint
    # subspace so it cannot move a requirement. This does the opposite. The
    # perturbation is applied *to* the constraint vector, and what is added to the
    # residual stream is one vector, f(S_c), not a sum of a signal and a
    # disturbance:
    #
    #   perp     f(S) = S + kappa * rms * sqrt(dim) * j,  j unit and perpendicular
    #            to S. The push along S survives exactly, so every constraint keeps
    #            the dose it had, and the sideways step is free to lie inside the
    #            constraint subspace -- which the protected-subspace offset forbids
    #            by construction. kappa is read on the same scale as the offset's
    #            gamma and the noise's alpha: the perturbation's length as a
    #            fraction of the hidden state's own length.
    #   rotate   f(S) = |S| * (S/|S| + kappa * j) / sqrt(1 + kappa^2). Same
    #            direction jitter, but norm-preserving: the constraint push is
    #            turned by atan(kappa) rather than added to, so the total amount of
    #            steering is identical to the unjittered run and only its aim moves.
    #   gain     f(S) = sum_c beta_c * exp(kappa z_c - kappa^2/2) * s_c, one
    #            lognormal draw per constraint per story (mean 1). Nothing leaves
    #            the constraint span at all: each story is written under a different
    #            emphasis of the same requirements.
    #
    # One draw per generation in every mode, held fixed for the whole story.
    jitter_kappa: float = 0.0
    jitter_mode: str = "none"
    # Where the sideways direction comes from: "iso" a fresh isotropic draw,
    # "basis" a draw restricted to the activation subspace, which keeps f(S_c) on
    # the manifold the model's own states occupy.
    jitter_draw: str = "iso"
    # Whether the (possibly jittered) constraint vector is added during decode.
    # Off, with ``steer_prefill`` on, gives the prompt-only variant: the model is
    # pushed once while it reads the instruction and then writes unperturbed.
    steer_decode: bool = True
    # How the constraint push is decided.
    #
    # ``constant`` adds the same vector to every story at every step, which is what
    # CAA and every run here has done. It is open loop: it pushes whether or not
    # this particular story needs pushing, and that is exactly why it buys
    # compliance by spending diversity -- forty stories all shoved the same way end
    # up more alike. Measured: 4.42 -> 3.88 requirements broken, 3.61 -> 3.12 Vendi.
    #
    # ``feedback`` measures where the story already sits on each constraint axis
    # and pushes only the shortfall:
    #
    #     a_c    = h . s_c                     how much of constraint c is present
    #     delta += beta_c * relu(tau_c - a_c) * s_c
    #
    # where tau_c is where text that satisfies the constraint sits, taken from the
    # positive side of that constraint's own contrast pairs. A story already past
    # tau_c is left alone entirely. So the correction is different for every story
    # rather than shared by all of them, which is the property the constant version
    # lacks, and the stories that are already fine are not homogenised toward a
    # single compliant point.
    steer_mode: str = "constant"
    # Ceiling on one feedback correction, as a fraction of the hidden state's own
    # length. Without it a story far from tau on several axes at once receives a
    # correction large enough to break the text, which is the failure mode every
    # over-strong constant arm has shown.
    feedback_cap: float = 0.1
    # This generation's per-constraint gains, for ``jitter_mode="gain"``.
    gains: Optional[List[float]] = None
    # Total length of the constraint push, in units of the model's own activation
    # scale, held fixed however many constraints are steered. None sums the
    # coefficients as they are, which is what every run before this did and which
    # makes "steer one more constraint" mean "push harder" as a side effect.
    steer_budget: Optional[float] = None
    # Where the constraint directions came from. "extracted" is the real thing;
    # "random" is the control that replaces each direction with a Gaussian draw of
    # the same norm, so a run can say whether the extracted direction meant
    # anything or whether a push of that size does the same whatever way it points.
    # Metadata only -- it changes nothing here, and exists so the two cannot share
    # a run id.
    direction_source: str = "extracted"

    # Decode steps the noise schedule is measured against. Defaults to ``horizon``.
    # Separate because ``horizon`` also drives the constraint schedules: a
    # perturbation confined to the first 24 tokens must not also compress the
    # closure ramp into 24 tokens.
    noise_horizon: Optional[int] = None
    # Entropy gate: the perturbation is applied only at decode steps where the
    # model's own next-token distribution was at least this uncertain, in nats.
    # 0.0 means every step, which is the ungated behaviour. ``gate_level`` is the
    # name the threshold was derived from, kept for the run id.
    gate_threshold: float = 0.0
    gate_level: str = "none"
    horizon: int = 200
    steer_prefill: bool = False
    protect_rank: int = 0

    # ---- construction ---------------------------------------------------- #

    @classmethod
    def build(
        cls,
        vectors: Mapping[str, Mapping[int, torch.Tensor]],
        layers: Sequence[int],
        specs: Sequence[ConstraintSpec],
        *,
        rms_scale: float,
        orthogonalize: str = "lowdin",
        noise_mode: str = "orth",
        noise_alpha: float = 0.175,
        noise_norm_match: str = "energy",
        noise_schedule: str = "constant",
        offset_gamma: float = 0.0,
        offset_mode: str = "none",
        offset_norm: str = "energy",
        offset_basis_kind: str = "step",
        offset_prefill: bool = False,
        offset_basis: Optional[Mapping[int, torch.Tensor]] = None,
        offset_decode: bool = True,
        amplify_lambda: float = 1.0,
        amplify_prefill: bool = False,
        amplify_basis: Optional[Mapping[int, torch.Tensor]] = None,
        amplify_mean: Optional[Mapping[int, torch.Tensor]] = None,
        noise_horizon: Optional[int] = None,
        jitter_kappa: float = 0.0,
        jitter_mode: str = "none",
        jitter_draw: str = "iso",
        steer_decode: bool = True,
        steer_budget: Optional[float] = None,
        steer_mode: str = "constant",
        feedback_cap: float = 0.1,
        targets: Optional[Mapping[str, Mapping[int, torch.Tensor]]] = None,
        direction_source: str = "extracted",
        gate_threshold: float = 0.0,
        gate_level: str = "none",
        horizon: int = 200,
        steer_prefill: bool = False,
        protect_extra: Optional[Mapping[int, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> "SteeringPlan":
        """Assemble a plan from raw per-constraint, per-layer direction vectors.

        ``vectors[name][layer]`` is a (dim,) tensor -- the raw mean-difference
        direction. ``protect_extra[layer]`` is an optional (dim, m) matrix of
        additional directions to shield the noise from (e.g. the top-m principal
        components of the per-item contrast differences), appended to the steering
        directions before building the protected basis.
        """
        specs = list(specs)
        if not specs:
            raise ValueError("at least one ConstraintSpec is required.")
        if jitter_mode not in JITTER_MODES:
            raise ValueError(f"jitter_mode must be one of {JITTER_MODES}, got '{jitter_mode}'.")
        if jitter_draw not in JITTER_DRAWS:
            raise ValueError(f"jitter_draw must be one of {JITTER_DRAWS}, got '{jitter_draw}'.")
        if steer_mode not in STEER_MODES:
            raise ValueError(f"steer_mode must be one of {STEER_MODES}, got '{steer_mode}'.")
        if steer_mode == "feedback" and not targets:
            raise ValueError(
                "steer_mode='feedback' needs `targets`: where text that satisfies "
                "each constraint sits on its own axis. Re-extract the steering "
                "vectors so the positive-side activations are stored."
            )
        layers = sorted({int(i) for i in layers})

        missing = [s.name for s in specs if s.name not in vectors]
        if missing:
            raise KeyError(
                f"no steering vectors for {missing}; available: {sorted(vectors)}"
            )

        layer_plans: Dict[int, LayerPlan] = {}
        dim = -1
        protect_rank = 0

        for layer in layers:
            cols = []
            for spec in specs:
                per_layer = vectors[spec.name]
                if layer not in per_layer:
                    raise KeyError(
                        f"constraint '{spec.name}' has no vector for layer {layer}; "
                        f"available layers: {sorted(per_layer)}"
                    )
                cols.append(per_layer[layer].detach().to(torch.float32).flatten())

            raw = torch.stack(cols, dim=1)  # (dim, C)
            if device is not None:
                raw = raw.to(device)
            dim = raw.shape[0]

            basis, report = orthonormalize(raw, method=orthogonalize)

            protect_cols = [basis]
            if protect_extra is not None and layer in protect_extra:
                extra = protect_extra[layer].detach().to(torch.float32)
                if device is not None:
                    extra = extra.to(device)
                if extra.dim() == 1:
                    extra = extra.unsqueeze(1)
                protect_cols.append(extra)
            protect = orthonormal_basis(torch.cat(protect_cols, dim=1))
            protect_rank = protect.shape[1]

            report["protect_rank"] = protect_rank

            jb = None
            if jitter_mode in ("perp", "rotate") and jitter_draw == "basis":
                if offset_basis is None or layer not in offset_basis:
                    raise ValueError(
                        "jitter_draw='basis' needs an activation basis; pass "
                        "offset_basis, or use jitter_draw='iso'."
                    )
                jb = offset_basis[layer].detach().to(torch.float32)
                if device is not None:
                    jb = jb.to(device)
                # Deliberately *not* projected against the constraint subspace. The
                # whole point of f(S_c) is that the perturbation may live inside
                # that subspace; it is kept off the constraint push by being made
                # perpendicular to S_c at use time instead.

            ob = None
            if offset_mode != "none" and offset_gamma > 0:
                if offset_basis is not None and layer in offset_basis:
                    ob = offset_basis[layer].detach().to(torch.float32)
                    if device is not None:
                        ob = ob.to(device)
                    if offset_mode == "orth":
                        # Strip the constraint subspace out of the directions the
                        # offset is allowed to use, once, at build time.
                        ob = complement_basis(ob, protect)
                elif offset_mode == "orth":
                    ob = None      # no basis: draw isotropically and project at draw time
                report["offset_rank"] = 0 if ob is None else ob.shape[1]

            ab = am = None
            if amplify_lambda != 1.0 and amplify_basis is not None and layer in amplify_basis:
                ab = amplify_basis[layer].detach().to(torch.float32)
                if device is not None:
                    ab = ab.to(device)
                if ab.dim() == 1:
                    ab = ab.unsqueeze(1)
                # Same projection as the offset: what the requirements depend on is
                # not amplified, so the arm changes diversity and nothing else.
                ab = complement_basis(ab, protect)
                if amplify_mean is not None and layer in amplify_mean:
                    am = amplify_mean[layer].detach().to(torch.float32).flatten()
                    if device is not None:
                        am = am.to(device)
                else:
                    am = torch.zeros(dim, dtype=torch.float32,
                                     device=ab.device if ab is not None else None)
                report["amplify_rank"] = 0 if ab is None else ab.shape[1]

            tgt = None
            if targets:
                missing = [sp.name for sp in specs
                           if layer not in targets.get(sp.name, {})]
                if missing:
                    raise KeyError(f"no target activation for {missing} at layer {layer}")
                # Where compliant text sits, expressed in the same orthogonalised
                # frame the push is applied in, so the projection read at
                # generation time and the target are the same quantity.
                tgt = torch.stack([
                    targets[sp.name][layer].detach().to(torch.float32).flatten()
                    for sp in specs
                ], dim=1)
                if device is not None:
                    tgt = tgt.to(basis.device)
                tgt = (basis * tgt.to(basis.device)).sum(dim=0)

            layer_plans[layer] = LayerPlan(
                layer=layer, basis=basis, protect=protect, report=report,
                offset_basis=ob, amp_basis=ab, amp_mean=am, jitter_basis=jb,
                raw_basis=unit_columns(raw).clone(), target=tgt,
            )

        return cls(
            layers=layers,
            specs=specs,
            layer_plans=layer_plans,
            dim=dim,
            rms_scale=float(rms_scale),
            orthogonalize=orthogonalize,
            noise_mode=noise_mode,
            noise_alpha=float(noise_alpha),
            noise_norm_match=noise_norm_match,
            noise_schedule=noise_schedule,
            offset_gamma=float(offset_gamma),
            offset_mode=offset_mode,
            offset_norm=offset_norm,
            offset_decode=bool(offset_decode),
            amplify_lambda=float(amplify_lambda),
            amplify_prefill=bool(amplify_prefill),
            offset_basis_kind=offset_basis_kind,
            offset_prefill=bool(offset_prefill),
            noise_horizon=None if noise_horizon is None else int(noise_horizon),
            jitter_kappa=float(jitter_kappa),
            jitter_mode=jitter_mode,
            jitter_draw=jitter_draw,
            steer_decode=bool(steer_decode),
            steer_budget=None if steer_budget is None else float(steer_budget),
            steer_mode=steer_mode,
            feedback_cap=float(feedback_cap),
            direction_source=direction_source,
            gate_threshold=float(gate_threshold),
            gate_level=gate_level,
            horizon=int(horizon),
            steer_prefill=bool(steer_prefill),
            protect_rank=protect_rank,
        )

    # ---- use ------------------------------------------------------------- #

    @property
    def sigma(self) -> float:
        """Base noise standard deviation, on the paper's ``alpha * median(RMS)`` scale."""
        return self.noise_alpha * self.rms_scale

    def to(self, device, dtype: torch.dtype = torch.float32) -> "SteeringPlan":
        for lp in self.layer_plans.values():
            lp.basis = lp.basis.to(device=device, dtype=dtype)
            if lp.protect is not None:
                lp.protect = lp.protect.to(device=device, dtype=dtype)
            if lp.offset_basis is not None:
                lp.offset_basis = lp.offset_basis.to(device=device, dtype=dtype)
            if lp.amp_basis is not None:
                lp.amp_basis = lp.amp_basis.to(device=device, dtype=dtype)
            if lp.amp_mean is not None:
                lp.amp_mean = lp.amp_mean.to(device=device, dtype=dtype)
            if lp.jitter_basis is not None:
                lp.jitter_basis = lp.jitter_basis.to(device=device, dtype=dtype)
            if lp.raw_basis is not None:
                lp.raw_basis = lp.raw_basis.to(device=device, dtype=dtype)
            if lp.jitter is not None:
                lp.jitter = lp.jitter.to(device=device, dtype=dtype)
            if lp.target is not None:
                lp.target = lp.target.to(device=device, dtype=dtype)
        return self

    def resample_jitter(self) -> None:
        """Draw this generation's perturbation of the constraint vector.

        Called from ``resample_offset`` so the generation loop needs no second
        hook. Like the offset, the draw is held fixed for the whole story: what
        varies between stories is *which* f(S_c) the model is written under, not
        which one it is written under at each token.
        """
        if self.jitter_mode == "none" or self.jitter_kappa <= 0:
            self.gains = None
            for lp in self.layer_plans.values():
                lp.jitter = None
            return

        if self.jitter_mode == "gain":
            # Lognormal with mean 1, so the expected constraint push is unchanged
            # and no draw can flip a constraint's sign and push against it.
            k = self.jitter_kappa
            z = torch.randn(len(self.specs))
            self.gains = [float(math.exp(k * float(zi) - 0.5 * k * k)) for zi in z]
            for lp in self.layer_plans.values():
                lp.jitter = None
            return

        if self.jitter_mode == "frame":
            # Perturb each constraint direction *before* the set is made mutually
            # orthogonal, rather than perturbing the summed vector afterwards.
            # Everything else here jitters one vector and leaves the orthogonal
            # frame identical for every story; this gives every story its own
            # frame. Where the perturbation lands is then decided by the
            # orthogonalisation, which redistributes it across the constraints
            # instead of letting it sit wherever it was drawn.
            self.gains = None
            k = self.jitter_kappa
            for lp in self.layer_plans.values():
                if lp.raw_basis is None:
                    continue
                dev, dt = lp.raw_basis.device, lp.raw_basis.dtype
                d = lp.raw_basis
                g = torch.randn(d.shape, dtype=dt, device=dev)
                # Only the part of each draw perpendicular to its own direction
                # matters: a component along it just rescales a unit column.
                g = g - d * (d * g).sum(dim=0, keepdim=True)
                g = g / g.norm(dim=0, keepdim=True).clamp_min(1e-12)
                lp.basis, _ = orthonormalize(d + g * k, method=self.orthogonalize)
                lp.jitter = None
            return

        self.gains = None
        for lp in self.layer_plans.values():
            dev, dt = lp.basis.device, lp.basis.dtype
            if self.jitter_draw == "basis" and lp.jitter_basis is not None:
                coeff = torch.randn(lp.jitter_basis.shape[1], dtype=dt, device=dev)
                vec = lp.jitter_basis @ coeff
            else:
                vec = torch.randn(self.dim, dtype=dt, device=dev)
            lp.jitter = vec / vec.norm().clamp_min(1e-12)

    def jitter_steering(
        self,
        layer: int,
        delta: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Apply f to a constraint vector that has already been summed.

        ``gain`` is handled inside ``LayerPlan.steering_delta`` because it acts on
        the per-constraint coefficients, not on the summed vector, so it is a no-op
        here.
        """
        if delta is None or self.jitter_kappa <= 0:
            return delta
        if self.jitter_mode not in ("perp", "rotate"):
            # "gain" acts on the coefficients and "frame" on the basis itself, so
            # by the time the vector is summed both are already applied.
            return delta
        lp = self.layer_plans[layer]
        if lp.jitter is None:
            return delta
        j = lp.jitter.to(device=delta.device, dtype=delta.dtype)
        n = delta.norm()
        if float(n) <= 1e-12:
            return delta
        u = delta / n
        # Only the part of the draw perpendicular to the constraint push is used,
        # so the dose along every constraint direction is exactly preserved and
        # kappa sets a known angle rather than an incidental one.
        j = j - u * (j @ u)
        jn = j.norm()
        if float(jn) <= 1e-12:
            return delta
        j = j / jn
        if self.jitter_mode == "rotate":
            k = self.jitter_kappa
            return (u + j * k) * (n / math.sqrt(1.0 + k * k))
        return delta + j * (self.jitter_kappa * self.rms_scale * math.sqrt(self.dim))

    def steering_only(
        self,
        layer: int,
        t: int,
        *,
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """The (possibly jittered) constraint vector alone, ignoring ``steer_decode``.

        Prefill uses this: whether the constraint vector is applied to the prompt
        is ``steer_prefill``'s decision, not ``steer_decode``'s.
        """
        lp = self.layer_plans[layer]
        if device is not None and lp.basis.device != device:
            lp.basis = lp.basis.to(device)
            if lp.jitter is not None:
                lp.jitter = lp.jitter.to(device)
        h = self.horizon if horizon is None else horizon
        return self.jitter_steering(
            layer, lp.steering_delta(t, h, self.specs, self.rms_scale,
                                     gains=self.gains, budget=self.steer_budget)
        )

    def resample_offset(self) -> None:
        """Draw a fresh constant offset for the next generation.

        Call once per story, after seeding. Unlike per-token noise this is held
        fixed for the whole generation, so it accumulates linearly the way the
        steering bias does and actually changes the story. That is also why it has
        to be kept out of the constraint subspace: a constant leak onto a
        constraint direction biases that constraint for the entire story.
        """
        self.resample_jitter()
        if self.offset_mode == "none" or self.offset_gamma <= 0:
            for lp in self.layer_plans.values():
                lp.offset = None
            return

        for lp in self.layer_plans.values():
            dev, dt = lp.basis.device, lp.basis.dtype
            if lp.offset_basis is not None:
                coeff = torch.randn(lp.offset_basis.shape[1], dtype=dt, device=dev)
                vec = lp.offset_basis @ coeff
            else:
                vec = torch.randn(self.dim, dtype=dt, device=dev)
                if self.offset_mode == "orth" and lp.protect is not None:
                    vec = vec - lp.protect @ (lp.protect.t() @ vec)
            if self.offset_norm == "energy":
                # Fixed length, so gamma means the same thing whatever the rank of
                # the subspace the offset was drawn from. A draw from a rank-r
                # subspace has natural length ~sqrt(r); without this, gamma silently
                # meant a perturbation sqrt(dim/r) times weaker than the same gamma
                # asked of the isotropic arm -- a factor of 8 at rank 64 in a
                # 4096-dimensional stream.
                vec = vec / vec.norm().clamp_min(1e-12)
                lp.offset = vec * (self.offset_gamma * self.rms_scale * math.sqrt(self.dim))
            else:
                lp.offset = vec * (self.offset_gamma * self.rms_scale)

    def delta_for(
        self,
        layer: int,
        t: int,
        *,
        horizon: Optional[int] = None,
        with_noise: bool = True,
        with_offset: bool = True,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Full (dim,) perturbation for ``layer`` at decode step ``t``.

        Noise is drawn from the ambient ``torch`` RNG, matching the rest of the
        codebase (which seeds per story via ``torch.manual_seed``). All three noise
        arms consume exactly one ``randn(dim)`` draw so a shared seed keeps the RNG
        stream aligned across conditions.

        ``device`` relocates this layer's tensors on first use, which is what makes
        the plan work under ``device_map="auto"`` when blocks are sharded.
        """
        lp = self.layer_plans[layer]
        if device is not None and lp.basis.device != device:
            lp.basis = lp.basis.to(device)
            if lp.protect is not None:
                lp.protect = lp.protect.to(device)
            if lp.offset_basis is not None:
                lp.offset_basis = lp.offset_basis.to(device)
            if lp.offset is not None:
                lp.offset = lp.offset.to(device)
            if lp.jitter is not None:
                lp.jitter = lp.jitter.to(device)

        h = self.horizon if horizon is None else horizon
        delta = None
        if self.steer_decode and self.steer_mode != "feedback":
            delta = self.jitter_steering(
                layer, lp.steering_delta(t, h, self.specs, self.rms_scale,
                                         gains=self.gains, budget=self.steer_budget)
            )

        # The per-story offset is a perturbation, so it is gated with the noise
        # rather than with the steering: an entropy gate closes on both together.
        if with_offset and self.offset_decode and lp.offset is not None:
            delta = lp.offset if delta is None else delta + lp.offset

        if with_noise and self.noise_mode != "none" and self.noise_alpha > 0:
            nh = h if self.noise_horizon is None else self.noise_horizon
            sigma = self.sigma * schedule_factor(self.noise_schedule, t, nh)
            if sigma > 0:
                g = torch.randn(self.dim, dtype=lp.basis.dtype, device=lp.basis.device)
                noise = constrained_noise(
                    g,
                    sigma=sigma,
                    basis=lp.protect,
                    mode=self.noise_mode,
                    norm_match=self.noise_norm_match,
                )
                if noise is not None:
                    delta = noise if delta is None else delta + noise

        return delta

    def feedback_delta(
        self,
        layer: int,
        state: torch.Tensor,
        t: int = 0,
        *,
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Close each constraint's shortfall for *this* story, not for all of them.

        ``state`` is one position of a block output, shape (dim,). The story's
        current position on constraint ``c`` is ``h . s_c``; anything short of
        ``tau_c`` is pushed up by ``beta_c`` times the gap, and anything already
        past it is left alone. Returns None when there is nothing to correct, so a
        story that is compliant on every axis is generated exactly as the model
        would have generated it.
        """
        if self.steer_mode != "feedback":
            return None
        lp = self.layer_plans[layer]
        if lp.target is None:
            return None
        # Relocate against the state we were handed, not against a `device`
        # argument, and check each tensor on its own. Keying the move off
        # `lp.basis.device` left `lp.target` on the CPU whenever some other path
        # had already moved the basis, which is a crash at the first decode step
        # under device_map="auto".
        want = state.device
        if lp.basis.device != want:
            lp.basis = lp.basis.to(want)
        if lp.target.device != want:
            lp.target = lp.target.to(want)

        h = self.horizon if horizon is None else horizon
        present = state.to(lp.basis.dtype) @ lp.basis          # (C,)
        gap = (lp.target - present).clamp_min(0.0)
        # A shortfall this small is floating-point residue from projecting in and
        # out of the basis, not a constraint the story is actually failing.
        gap = torch.where(gap > 1e-4 * self.rms_scale, gap, torch.zeros_like(gap))
        coeffs = torch.tensor(
            [
                spec.beta * schedule_factor(spec.schedule, t, h)
                * (1.0 if self.gains is None else float(self.gains[i]))
                for i, spec in enumerate(self.specs)
            ],
            dtype=lp.basis.dtype, device=lp.basis.device,
        ) * gap
        if bool((coeffs == 0).all()):
            return None
        delta = lp.basis @ coeffs
        cap = self.feedback_cap * self.rms_scale * math.sqrt(self.dim)
        n = float(delta.norm())
        if cap > 0 and n > cap:
            delta = delta * (cap / n)
        return delta

    def amplify_delta(
        self,
        layer: int,
        state: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """What to add to ``state`` to stretch its between-story component.

        ``state`` is a block output, either one position of shape (dim,) or a run of
        them of shape (T, dim) -- prefill hands over the whole prompt at once, and
        each position there has its own deviation from the average, so this is not
        one vector added everywhere. The result has the same shape as ``state`` and
        is ``(lambda - 1) * (state - mu) B B^T``: the part of each position that
        distinguishes this story from the average story, scaled up. Adding it
        multiplies that part by ``lambda`` and leaves everything else, including
        every constraint direction, exactly as it was.

        Returns None when there is nothing to do, so the hook can skip the work.
        """
        if self.amplify_lambda == 1.0:
            return None
        lp = self.layer_plans[layer]
        if lp.amp_basis is None:
            return None
        if device is not None and lp.amp_basis.device != device:
            lp.amp_basis = lp.amp_basis.to(device)
            if lp.amp_mean is not None:
                lp.amp_mean = lp.amp_mean.to(device)
        h = state.to(lp.amp_basis.dtype)
        if lp.amp_mean is not None:
            h = h - lp.amp_mean
        return ((h @ lp.amp_basis) @ lp.amp_basis.t()) * (self.amplify_lambda - 1.0)

    # ---- diagnostics ------------------------------------------------------ #

    def describe(self) -> Dict[str, object]:
        names = [s.name for s in self.specs]
        per_layer = {}
        for layer in self.layers:
            rep = self.layer_plans[layer].report
            per_layer[layer] = {
                "max_offdiag_cosine": rep.get("max_offdiag_cosine"),
                "retained": [round(float(v), 4) for v in rep.get("retained", [])],
                "protect_rank": rep.get("protect_rank"),
            }
        return {
            "constraints": names,
            "betas": [s.beta for s in self.specs],
            "schedules": [s.schedule for s in self.specs],
            "layers": self.layers,
            "dim": self.dim,
            "rms_scale": self.rms_scale,
            "orthogonalize": self.orthogonalize,
            "noise_mode": self.noise_mode,
            "noise_alpha": self.noise_alpha,
            "noise_sigma": self.sigma,
            "noise_norm_match": self.noise_norm_match,
            "noise_schedule": self.noise_schedule,
            "offset_gamma": self.offset_gamma,
            "offset_mode": self.offset_mode,
            "offset_norm": self.offset_norm,
            "offset_decode": self.offset_decode,
            "amplify_lambda": self.amplify_lambda,
            "amplify_prefill": self.amplify_prefill,
            "amplify_rank": self.layer_plans[self.layers[0]].report.get("amplify_rank"),
            "offset_basis_kind": self.offset_basis_kind,
            "offset_prefill": self.offset_prefill,
            "noise_horizon": self.noise_horizon,
            "jitter_mode": self.jitter_mode,
            "jitter_kappa": self.jitter_kappa,
            "jitter_draw": self.jitter_draw,
            "steer_decode": self.steer_decode,
            "steer_budget": self.steer_budget,
            "steer_mode": self.steer_mode,
            "feedback_cap": self.feedback_cap,
            "direction_source": self.direction_source,
            "offset_rank": self.layer_plans[self.layers[0]].report.get("offset_rank"),
            "horizon": self.horizon,
            "steer_prefill": self.steer_prefill,
            "protect_rank": self.protect_rank,
            "per_layer": per_layer,
        }

    def print_report(self) -> None:
        info = self.describe()
        names = info["constraints"]
        print("=== Steering Plan ===")
        print(f"constraints      : {names}"
              + ("   [RANDOM DIRECTIONS -- this is the control arm]"
                 if info["direction_source"] == "random" else ""))
        print(f"betas            : {info['betas']}")
        print(f"schedules        : {info['schedules']}")
        print(f"layers           : {info['layers']}")
        print(f"residual dim     : {info['dim']}")
        print(f"rms_scale        : {info['rms_scale']:.6g}")
        print(f"orthogonalisation: {info['orthogonalize']}")
        print(
            f"noise            : mode={info['noise_mode']} alpha={info['noise_alpha']:.6g} "
            f"sigma={info['noise_sigma']:.6g} norm_match={info['noise_norm_match']} "
            f"schedule={info['noise_schedule']} horizon={info['horizon']}"
        )
        if info["offset_mode"] != "none" and info["offset_gamma"]:
            print(f"per-story offset : gamma={info['offset_gamma']:.6g} mode={info['offset_mode']} "
                  f"rank={info['offset_rank']} basis={info['offset_basis_kind']} "
                  f"norm={info['offset_norm']} "
                  f"prefill={info['offset_prefill']}")
        if info["jitter_mode"] != "none" and info["jitter_kappa"]:
            print(f"f(S_c)           : mode={info['jitter_mode']} "
                  f"kappa={info['jitter_kappa']:.6g} draw={info['jitter_draw']} "
                  f"(one draw per story)")
        if info["amplify_lambda"] != 1.0:
            print(f"amplification    : lambda={info['amplify_lambda']:.6g} over "
                  f"{info['amplify_rank']} between-story directions "
                  f"(prefill={info['amplify_prefill']})")
        print(f"protected rank   : {info['protect_rank']} of {info['dim']} "
              f"({100.0 * info['protect_rank'] / max(info['dim'], 1):.3f}% of the stream)")
        print(f"steer at prefill : {info['steer_prefill']}  at decode: "
              f"{info['steer_decode']}")
        print("\n-- per-layer geometry --")
        for layer in self.layers:
            rep = self.layer_plans[layer].report
            gram = rep.get("cosine_gram")
            retained = ", ".join(
                f"{n}={float(v):.3f}" for n, v in zip(names, rep.get("retained", []))
            )
            print(f"  layer {layer:>3}: max|cos| between raw directions = "
                  f"{rep.get('max_offdiag_cosine', float('nan')):.3f} | retained after "
                  f"orthogonalisation: {retained}")
            if gram is not None and len(names) > 1:
                for i, n in enumerate(names):
                    row = "  ".join(f"{float(gram[i, j]):+.3f}" for j in range(len(names)))
                    print(f"            cos[{n:>16}] = {row}")
