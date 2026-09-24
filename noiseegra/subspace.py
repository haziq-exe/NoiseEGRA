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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, FrozenSet

import torch

from .colored_noise import colored_sequence

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
STEER_MODES = ("constant", "feedback", "error")
NORM_MATCH_MODES = ("energy", "none")
SCHEDULES = ("constant", "cosine_decay", "ramp", "linear_decay", "prefix", "tail")


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
    ``tail``          0.0 for the first ``horizon`` decode steps, then rising to
                      1.0 over the next ``horizon``. For a constraint about when
                      to stop: silent while the story is still within its budget
                      and pressing harder the longer it runs over.
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
    if kind == "tail":
        # Nothing until the story has run as long as it is allowed to, then rising.
        #
        # The counting constraints -- a word band, a sentence band, exactly two
        # quoted lines -- are the ones steering makes *worse*: 23% at baseline and
        # 6% steered. They are about when to stop, and a coefficient that is the
        # same at token five and token ninety cannot express that. The model has no
        # representation of "how many words so far", but the generation loop does,
        # so the controller can count even though the model cannot.
        if t < horizon:
            return 0.0
        return float(min(1.0, (t - horizon) / max(horizon, 1)))
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
    # The sampled stories themselves, centred: (n_stories, dim), one row per
    # story. Used when the displacement is aimed at a story rather than drawn in
    # the span of the principal components summarising them.
    offset_anchors: Optional[torch.Tensor] = None
    # How far one of the model's own stories typically sits from their average,
    # at this layer. This is the natural unit for a displacement and nothing
    # used it for fifty rounds: gamma was measured against the activation's own
    # length, which is far larger. At this model's layer 6 a real story sits 6.9
    # from the average and a displacement of 0.15 -- the size every good arm was
    # run at -- is 15.0, already twice as far out as any story the model wrote.
    # Every arm has been extrapolating, which is what the refusals are.
    story_radius: Optional[float] = None
    # A multiplier per sampled story, applied to the displacement's size when it
    # aims at that one. Removing the eight sampled stories that break the writing
    # costs the variety they were carrying -- the same eight are responsible for
    # both -- so they are travelled towards less far instead of being dropped.
    anchor_scale: Optional[torch.Tensor] = None
    report: Dict[str, object] = field(default_factory=dict)
    offset_basis: Optional[torch.Tensor] = None   # (dim, M) directions offsets may use
    # (M,) how far stories actually spread along each of those directions. Used
    # when the draw is shaped to the manifold rather than to the sphere.
    offset_scale: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None         # (dim,) this generation's offset
    # (T, M) coefficients, one row per decode step, when the displacement is a
    # trajectory rather than a fixed vector. Set only when a noise colour is
    # asked for. `offset` stays the value at step 0, so every code path that
    # reads it -- the prompt in particular -- behaves exactly as before.
    offset_traj: Optional[torch.Tensor] = None
    # The length `offset` was given, kept so the trajectory can be held at that
    # same length at every step. Only the direction is allowed to wander: the
    # size of the displacement is the one quantity in this project that has a
    # meaning ("gamma story-distances from the average story"), and letting it
    # drift with the noise colour would make an exponent sweep a size sweep too.
    offset_length: Optional[float] = None
    amp_basis: Optional[torch.Tensor] = None      # (dim, r) directions stories differ along
    amp_mean: Optional[torch.Tensor] = None       # (dim,) what they differ *from*
    jitter_basis: Optional[torch.Tensor] = None   # (dim, M) directions f(S_c) may use
    jitter: Optional[torch.Tensor] = None         # (dim,) this generation's unit draw
    raw_basis: Optional[torch.Tensor] = None      # (dim, C) before orthogonalisation
    target: Optional[torch.Tensor] = None         # (C,) where compliant text sits

    def relocate(self, device: "torch.device") -> None:
        """Move every tensor this layer owns onto ``device``, checking each.

        Not guarded by any single tensor's location: they are created at
        different times -- the basis once, the offset every story, the jitter
        every story, the protected subspace at build time -- so one of them
        being in the right place says nothing about the others. Two crashes came
        from assuming it did.
        """
        for name in ("basis", "protect", "offset_basis", "offset_anchors",
                     "anchor_scale", "offset", "offset_traj", "jitter",
                     "amp_basis", "amp_mean", "jitter_basis", "raw_basis",
                     "target", "offset_scale"):
            t = getattr(self, name, None)
            if t is not None and hasattr(t, "device") and t.device != device:
                setattr(self, name, t.to(device))

    def offset_at(self, t: int, envelope: float = 1.0) -> Optional[torch.Tensor]:
        """This story's displacement at decode step ``t``.

        Without a trajectory this is the fixed per-story displacement and ``t``
        is ignored, which is the mechanism this project already had. With one,
        the direction follows the coloured-noise trajectory while the length is
        held at what the draw gave it, so ``gamma`` still means the same number
        of story-distances at every step and at every noise colour.
        """
        if self.offset is None:
            return None
        if self.offset_traj is None or self.offset_basis is None:
            return self.offset if envelope == 1.0 else self.offset * envelope
        tr = self.offset_traj
        row = tr[min(int(t), tr.shape[0] - 1)]
        vec = self.offset_basis @ row
        length = self.offset_length
        if length is None:
            length = float(self.offset.norm())
        return vec * (length * envelope / float(vec.norm().clamp_min(1e-12)))

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


def _spread_on_sphere(pts: torch.Tensor, iters: int = 400,
                      step: float = 0.05) -> torch.Tensor:
    """Push unit vectors apart until they are as evenly spread as they get.

    Riesz energy: every pair repels with a force falling off as the inverse cube
    of the distance between them, and the points are renormalised back onto the
    sphere after each step. Signs are free -- a direction and its negative are the
    same axis for this purpose -- so the repulsion is computed on absolute
    similarity, which lets ``n`` points settle onto ``n`` orthogonal axes when the
    rank allows instead of onto antipodal pairs.

    This is a deterministic layout, not a sample. Two hundred points in a rank-32
    subspace cannot all be orthogonal, but they can be spread far better than
    independent draws manage, and the gap is exactly the diversity that iid
    sampling leaves on the table.
    """
    n, r = pts.shape
    if n < 2:
        return pts
    for _ in range(iters):
        # (n, n) cosine similarities; the sign-free version, so antipodes attract
        # no more than duplicates.
        sim = pts @ pts.t()
        sim.fill_diagonal_(0.0)
        # Force on i from j, along the component of j that i does not already
        # have, weighted by how close they are. sim^3 concentrates the force on
        # the pairs that are actually colliding.
        weight = sim.sign() * sim.abs().pow(3.0)
        push = pts - weight @ pts / max(n - 1, 1) * step * n
        pts = push / push.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return pts


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
    # The colour of the displacement's wandering along the token axis. None
    # leaves the displacement fixed for the whole story, which is what this
    # project has always done. 0.0 redraws its direction independently at every
    # step, which is the published per-token mechanism. In between, the
    # direction drifts: the power in the trajectory falls as 1/f^beta, so a
    # larger exponent means a slower drift. See `noiseegra.colored_noise`.
    noise_beta: Optional[float] = None
    # The slowest wobble allowed, in cycles across one story. Below one, the
    # slowest component does not complete a cycle inside the story and so
    # survives as a displacement that differs between stories -- which is the
    # only kind that can make a set of stories more varied.
    noise_fmin_cycles: float = 0.25
    # How many decode steps the trajectory is generated for.
    noise_traj_steps: int = 640
    # The span a "decay" or "rise" envelope takes, in decode steps. 0 uses the
    # trajectory's length, which at 640 is more than twice a story here: the
    # fade was still at half strength when the story ended.
    offset_envelope_steps: int = 0
    # For the "secured" envelope: how much the noise grows once every rule that
    # stays met is met. The envelope is 1 + boost * (share met), read each step
    # from what the constraint controller last saw of the story.
    offset_secured_boost: float = 0.0
    # For the "front" envelope: the noise starts at this multiple of its size
    # and falls back to it by offset_envelope_steps, on a cosine. The opening
    # words are where the steered stories are most alike.
    offset_front_gain: float = 1.0
    # A multiple of the offset applied to the prompt alone. 1 leaves the prompt
    # at the calibrated size.
    offset_prefill_gain: float = 1.0
    # Size the noise while the story is written instead of before it (see
    # noiseegra.online_calibration). The value is the target, in the Fisher
    # calibration's units: the noise moves the next-token prediction this many
    # times as far as top-p sampling at high temperature would at the same
    # step, on average over the story. 0 leaves it off. The prompt's noise is
    # written before the first measurement, so it keeps the starting length.
    offset_online: float = 0.0
    # The multiple of the starting length the controller has set for the next
    # decode step. Reset to 1 for every story.
    online_gain: float = 1.0
    # The largest multiple of the starting length the controller may reach. 2.5
    # held on Qwen3-1.7B, where the second-half gain never passed 1.5; a model
    # far less sensitive to a residual offset needs more room.
    online_max_gain: float = 2.5
    # Carry the size the controller settled on into the next story, prompt
    # included, as the adaptive scaling of Plappert et al. runs across episodes.
    online_carry: bool = False
    # Set the target before the stories from the model itself instead of fixing
    # it: the share of the top word's probability the noise may move each step
    # (see online_calibration.target_from_top_share). 0 keeps offset_online.
    online_rule_k: float = 0.0
    # With the rule, also measure the starting length before any story: the
    # length at which random draws of the noise already reach the target (see
    # online_calibration.start_for_target). Off keeps offset_gamma.
    online_rule_start: bool = False
    # Tilt the story's next-token distribution toward the tokens the rules make
    # likelier (the constraints' output profiles, from the same forward passes
    # as the steering directions). The value is the tilt's size at every step as
    # a share of how far top-p sampling at high temperature would move that
    # step's distribution, so it is measured in the same units as the noise.
    # 0 leaves it off. See noiseegra.online_calibration.RuleTilt.
    output_tilt: float = 0.0
    output_profile: Optional[torch.Tensor] = None
    # How the displacement's size runs over the story: "flat", "decay" (large
    # at the start, fading), or "rise" (small at the start, growing). The
    # register failures this project measures come from displacement early on,
    # where the model decides whether it is answering the reader or telling a
    # story, so a rising envelope is the one with a reason behind it.
    offset_envelope: str = "flat"
    # Which set of directions the offset was drawn from, kept so two conditions
    # that differ only in that cannot collide on disk. "step" is the principal
    # components of individual decode-step activations, "story" the components of
    # whole-story mean activations.
    offset_basis_kind: str = "step"
    # How the set of per-story offsets is chosen. "iid" draws each one
    # independently, which is what every round so far has done. "spread" lays the
    # whole set out in advance so the offsets repel one another, and story i takes
    # the i-th of them.
    #
    # The difference is not a re-aiming of one story's perturbation -- turning a
    # push of constant length was measured and is a null, +0.12 Vendi. It is a
    # choice about the *set*, and diversity is a property of the set. Independent
    # draws in a low-rank subspace collide: a hundred draws from a rank-8 basis
    # contain pairs 95% identical, so several stories are perturbed the same way
    # and the diversity that was paid for is not collected. Every offset keeps the
    # same length and the same subspace, so the constraint cost is unchanged by
    # construction; only the collisions go.
    offset_draw: str = "iid"
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
    # How fast f(S_c) changes from one decode step to the next, between 0 and 1.
    #
    # 0 is what every run so far has done: one draw per story, the same f(S_c)
    # at every token, so what varies is which f the story is written under and
    # not which one each token is written under. 1 redraws independently at
    # every step, which is white noise on the aim. In between, the aim performs
    # a correlated random walk -- j <- normalise(sqrt(1-w^2) * j + w * xi) --
    # whose correlation time is about 1/w steps.
    #
    # The reason to want the middle is the difference between the two ends. A
    # per-story draw cannot give token-level variety, and token-level variety is
    # the one axis the method does not win: raised temperature with top-k
    # sampling varies at every token and leads variety of what happens by 5.2 on
    # an interval that clears zero, and nothing that changes where, how hard or
    # how the perturbation is drawn has closed it. White noise gives token-level
    # variety and destroys the text: on this model the published per-token noise
    # broke every opening story at 0.4 and matched the baseline exactly at 0.2.
    #
    # A walk is the object between them. Adjacent tokens see almost the same
    # aim, so nothing breaks locally; over a story the aim explores, so what the
    # story is about can drift. In "rotate" mode the length of the push is
    # preserved exactly at every step, so the model is pushed as hard as ever
    # and only where it is aimed wanders -- which is why this is a function of
    # the constraint vector rather than a term added beside it.
    jitter_walk: float = 0.0
    # Which directions a per-story gain varies. None varies all of them.
    #
    # It matters which. For most of this project every steered direction was
    # about how a sentence is formed -- present tense, sensory words, a named
    # character, the format -- so varying their strength between stories varied
    # the style and not what the story was about, and measured flat. The event
    # direction is the first that changes what happens, so varying that one
    # alone gives between-story variation in content without displacing the
    # state off the region the model writes from, which is what the per-story
    # displacement costs coherence for.
    jitter_names: Optional[List[str]] = None
    # Which decode step the walk has been advanced to, so every layer sees the
    # same step rather than each advancing it again.
    _jitter_step: int = -1
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
    # ``error`` sets each direction's coefficient from the constraint error of
    # the text written so far, measured by the same checks that score the finished
    # story. A requirement currently satisfied gets a coefficient of zero and the
    # model writes unsteered; one outside its band gets a push proportional to how
    # far outside, in the direction that brings it back. This is the half that
    # neither of the other two can express: a count has no "more is better"
    # direction, so a constant coefficient sails past the target and activation
    # feedback saturates at "quote-like enough" rather than at "two quotes".
    # ``error`` sets each direction's coefficient from the constraint error of the
    # text written so far, measured by the same checks that score the finished
    # story. A requirement currently satisfied gets a coefficient of zero and the
    # model writes unsteered; one outside its band gets a push proportional to how
    # far outside, in the direction that brings it back. This is the half neither
    # of the other two can express: a count has no "more is better" direction, so
    # a constant coefficient sails past the target, and activation feedback
    # saturates at "quote-like enough" rather than at "two quotes".
    steer_mode: str = "constant"
    # Ceiling on one feedback correction, as a fraction of the hidden state's own
    # length. Without it a story far from tau on several axes at once receives a
    # correction large enough to break the text, which is the failure mode every
    # over-strong constant arm has shown.
    feedback_cap: float = 0.1
    # This generation's per-constraint gains, for ``jitter_mode="gain"``.
    gains: Optional[List[float]] = None
    # Live constraint errors, written each decode step by ConstraintProbe and read
    # by the hook on the next one. Empty until generation starts.
    control_state: Optional[dict] = None
    # The object that turns the story so far into a signed error per direction.
    controller: object = None
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
    # How hard to push at the prompt positions, as a multiple of the strength
    # used while writing. 1.0 keeps the two equal, which is what every run
    # before this used.
    #
    # They are separated because the two sitings turned out to do different
    # jobs. Pushing during decoding is what enforces a sustained property: at a
    # total strength of 2 it takes present-tense finite verbs from 13% to 94%.
    # Pushing at the prompt and leaving decoding alone does none of that -- the
    # tense share stays at the untouched model's 12% -- but takes variety of
    # wording from 9.7 to 16.7, close to what raising the sampling temperature
    # to 1.8 reaches, and leaves sentence length alone. A constant offset cannot
    # vary between stories, so that gain is not per-story variation; the shifted
    # prompt state is one the model is less practised at continuing, and it
    # continues it less predictably.
    #
    # With one strength for both, the two effects cannot be dosed apart. This
    # allows a large push at the prompt and a small one while writing.
    prefill_gain: float = 1.0
    # How many of the final prompt positions to leave untouched at prefill.
    #
    # The prompt does not end with the instruction: it ends with the chat
    # template's own tokens, the ones that say the user has stopped talking and
    # the assistant is answering now. Perturbing those along with everything
    # else weakens the only signal that this is a reply to an instruction, and
    # the model falls back on what text that starts with no instruction looks
    # like -- which is a document, and a document begins with a title.
    #
    # That is not speculative about the symptom. The instruction ends "Write
    # only the story itself: no title, heading, preamble or commentary", and the
    # untouched model, both raised-temperature arms and the constraint push on
    # its own produce no titles whatever. A per-story perturbation at the prompt
    # produces them in 3% of stories at 0.1 and 29% at 0.15:
    #
    #     **Title: The Day the Sky Grew Cold**
    #     **Short Story for Middle-School Readers: "The Last Drop"**
    #
    # Leaving the last few positions clear keeps the boundary intact while still
    # moving how the model read the instruction itself. 0 keeps every run before
    # this one.
    prompt_tail_clear: int = 0
    # How many of the *first* prompt positions to leave untouched: the mirror of
    # prompt_tail_clear.
    #
    # The prompt does not begin with the instruction any more than it ends with
    # it. It begins with the chat template's opening and the system line, which
    # here is "You write short stories for readers in middle school and early
    # high school" -- the only place the model is told what kind of thing it is
    # producing at all. Perturbing that along with everything else is a
    # candidate for why it falls back to writing a document, which has a title;
    # sparing the end alone does not remove them, 4% of stories still opening
    # with one at the size that scores best.
    prompt_head_clear: int = 0
    # Which layers each half of the method acts on, when they should differ.
    # Empty means "every layer the plan covers", which is what every run before
    # this used.
    #
    # The two halves have no reason to want the same band. The constraint push
    # controls properties of the sentence being written -- its tense, whether it
    # names the character -- and works at layers 6-13 of 28, fragmenting the text
    # at 14-22. The perturbation's job is different: it is supposed to send the
    # model to a different story, and which story gets told is not obviously
    # decided in the same place as how a sentence is worded. Every run so far has
    # applied both over one band because one `--layers` set both.
    push_layers: FrozenSet[int] = frozenset()
    offset_layers: FrozenSet[int] = frozenset()
    # How many decode steps the per-story perturbation lasts, when it is applied
    # while the story is written. 0 means every step, which is what
    # `offset_decode` alone has always meant.
    #
    # The two sitings turned out to do different jobs, and neither does both.
    # Perturbing the prompt changes what the story is about -- variety of what
    # happens 71.6 against the untouched model's 55 -- but past a certain
    # displacement the model stops treating the prompt as an instruction and
    # writes a document with a title. Perturbing while the story is written
    # never does that, 0% titles and 100 of 100 coherent, but barely changes
    # what happens: 56.4 at a perturbation of 0.05 and 63.1 at 0.1, below the
    # prompt siting at a smaller size.
    #
    # The reading is that a story's content is settled in its opening tokens,
    # which are written from a prompt this siting has not touched; perturbing
    # after that changes the texture of a story already chosen. If that is
    # right, perturbing the opening decode steps and stopping should buy the
    # content variety of the prompt siting without touching the prompt at all.
    offset_decode_steps: int = 0
    # How the per-story perturbation's coefficients are drawn.
    #
    #   "sphere"    every direction weighted equally, which is what every run
    #               before this used
    #   "manifold"  weighted by how far stories actually spread along each
    #               direction, so a perturbation of a given size is shaped like
    #               a real difference between two of the model's own stories
    #
    # This is aimed at a measured ceiling. Variety of what happens rises with
    # the size of the perturbation up to 0.15 and falls after it -- 67.2, then
    # 64.3 at 0.2, then 61.8 at 0.25 -- while variety of wording keeps climbing,
    # 17.9 to 24.1 to 26.5. Past the peak the displacement stops sending the
    # model to a different story and starts disturbing the wording of the one it
    # was already going to write.
    #
    # A uniform draw is a candidate for why. The basis is ordered by how much
    # between-story variation each direction carries, and the last directions
    # carry almost none; weighting them equally with the first means a step of a
    # given length lands far outside anything the model does. Shaping the draw
    # by the observed spread keeps it inside.
    offset_draw_shape: str = "sphere"
    # A fresh random subspace for every story, of this rank, in place of any
    # basis estimated from the model. 0 keeps the previous behaviour. The
    # subspace is drawn uniformly, so over many stories the displacement is
    # isotropic; within one story it gives the coloured-noise trajectory a
    # fixed set of directions to wander among. Nothing about it is learned.
    offset_random_rank: int = 0
    # Run a shadow copy of the story alongside it: the same words, the same
    # steering, no perturbation. At every steered layer the story's state is
    # made to agree with the shadow's along the protected directions, which
    # removes whatever part of the perturbation the earlier layers carried back
    # into them. Projecting the perturbation clear of those directions where it
    # is added protects them only at that layer.
    shadow_protect: bool = False
    # A random split of the steering budget, drawn afresh for every story from a
    # symmetric Dirichlet of this concentration and applied as per-rule gains.
    # The total push is unchanged -- the budget renormalises it -- but no two
    # stories are pushed in quite the same stylistic direction. 0 keeps every
    # story on the same split. Smaller is more varied: 1 is uniform over all
    # splits.
    steer_split_concentration: float = 0.0
    # Random noise at another place in the architecture (see
    # noiseegra.arch_noise): its name and its size in nucleus-units. Empty or
    # zero leaves it off.
    arch_mechanism: str = ""
    arch_size: float = 0.0
    # Fresh draws for every sentence the story writes, at the same size.
    arch_per_sentence: bool = False
    # How much the size of the per-story displacement varies between stories,
    # as a fraction of its nominal size. 0 gives every story the same
    # displacement, which is what every run so far has done.
    #
    # A fixed size puts every story on a shell around the unperturbed state
    # rather than filling the ball inside it, and Vendi measures spread. On the
    # geometry alone -- a hundred points at this basis's rank, scored with the
    # same Vendi -- drawing the radius uniformly over [0, 2r] scores 2.3 against
    # 1.9 for a fixed radius, with the mean displacement unchanged.
    #
    # It is the one property of the displacement never varied. Where it points
    # has been swept, how it is drawn has been swept, where it is applied has
    # been swept, and how far it goes has been swept in the mean -- never in the
    # spread.
    offset_gamma_spread: float = 0.0
    # How much of the perturbation survives at the last prompt position it
    # touches, as a fraction of its strength at the first. 1.0 is flat, which is
    # what every run before this used.
    offset_taper: float = 1.0
    # The name of a direction whose share of the push is scaled, per story, by
    # how far that story was displaced.
    #
    # The formatting ceiling binds per story and not on the average. Varying the
    # displacement size took headings from 2% to 9% while the *mean*
    # displacement stayed put, so the stories drawn near the top of the range
    # are crossing the threshold individually while the ones near the bottom are
    # nowhere near it. A push that is the same for both spends protection where
    # it is not needed and withholds it where it is.
    #
    # This gives the named direction a coefficient proportional to this story's
    # displacement. The total push is still renormalised to the same budget, so
    # a story that is displaced far trades some of its other steering for
    # formatting, and a story that is barely displaced keeps it.
    #
    # It is the one route left to more content variety that the measurements
    # allow: they say the cap cannot be got round by spending the same budget
    # unevenly, only by raising what an individual story can absorb.
    guard_direction: str = ""
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
        noise_beta: Optional[float] = None,
        noise_fmin_cycles: float = 0.25,
        noise_traj_steps: int = 640,
        offset_envelope_steps: int = 0,
        offset_secured_boost: float = 0.0,
        offset_front_gain: float = 1.0,
        offset_prefill_gain: float = 1.0,
        offset_envelope: str = "flat",
        offset_basis_kind: str = "step",
        offset_draw: str = "iid",
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
        jitter_walk: float = 0.0,
        jitter_names: Optional[Sequence[str]] = None,
        steer_decode: bool = True,
        steer_budget: Optional[float] = None,
        steer_mode: str = "constant",
        feedback_cap: float = 0.1,
        control_state: Optional[dict] = None,
        controller: object = None,
        targets: Optional[Mapping[str, Mapping[int, torch.Tensor]]] = None,
        direction_source: str = "extracted",
        gate_threshold: float = 0.0,
        gate_level: str = "none",
        horizon: int = 200,
        steer_prefill: bool = False,
        prefill_gain: float = 1.0,
        prompt_tail_clear: int = 0,
        prompt_head_clear: int = 0,
        push_layers: Optional[Sequence[int]] = None,
        offset_layers: Optional[Sequence[int]] = None,
        offset_decode_steps: int = 0,
        offset_scale: Optional[Mapping[int, torch.Tensor]] = None,
        offset_draw_shape: str = "sphere",
        offset_random_rank: int = 0,
        offset_online: float = 0.0,
        online_max_gain: float = 2.5,
        online_carry: bool = False,
        online_rule_k: float = 0.0,
        online_rule_start: bool = False,
        output_tilt: float = 0.0,
        output_profile: Optional[torch.Tensor] = None,
        shadow_protect: bool = False,
        steer_split_concentration: float = 0.0,
        arch_mechanism: str = "",
        arch_size: float = 0.0,
        arch_per_sentence: bool = False,
        offset_anchors: Optional[Mapping[int, torch.Tensor]] = None,
        anchor_scale: Optional[torch.Tensor] = None,
        offset_gamma_spread: float = 0.0,
        offset_taper: float = 1.0,
        guard_direction: str = "",
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
        if steer_mode == "error" and control_state is None:
            raise ValueError(
                "steer_mode='error' needs `control_state`, the dict a "
                "ConstraintProbe writes the live constraint errors into."
            )
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
                offset_basis=ob,
                offset_anchors=(None if offset_anchors is None
                                else offset_anchors.get(layer)),
                anchor_scale=anchor_scale,
                story_radius=(None if offset_anchors is None
                              or offset_anchors.get(layer) is None
                              else float(offset_anchors[layer].norm(dim=1).mean())),
                offset_scale=(None if offset_scale is None
                              else offset_scale.get(layer)),
                amp_basis=ab, amp_mean=am, jitter_basis=jb,
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
            noise_beta=None if noise_beta is None else float(noise_beta),
            noise_fmin_cycles=float(noise_fmin_cycles),
            noise_traj_steps=int(noise_traj_steps),
            offset_envelope_steps=int(offset_envelope_steps or 0),
            offset_secured_boost=float(offset_secured_boost or 0.0),
            offset_front_gain=float(offset_front_gain or 1.0),
            offset_prefill_gain=float(offset_prefill_gain or 1.0),
            offset_envelope=str(offset_envelope),
            offset_decode=bool(offset_decode),
            amplify_lambda=float(amplify_lambda),
            amplify_prefill=bool(amplify_prefill),
            offset_basis_kind=offset_basis_kind,
            offset_draw=offset_draw,
            offset_prefill=bool(offset_prefill),
            noise_horizon=None if noise_horizon is None else int(noise_horizon),
            jitter_kappa=float(jitter_kappa),
            jitter_mode=jitter_mode,
            jitter_walk=float(jitter_walk),
            jitter_names=(None if not jitter_names else list(jitter_names)),
            jitter_draw=jitter_draw,
            steer_decode=bool(steer_decode),
            steer_budget=None if steer_budget is None else float(steer_budget),
            steer_mode=steer_mode,
            feedback_cap=float(feedback_cap),
            control_state=control_state,
            controller=controller,
            direction_source=direction_source,
            gate_threshold=float(gate_threshold),
            gate_level=gate_level,
            horizon=int(horizon),
            steer_prefill=bool(steer_prefill),
            prefill_gain=float(prefill_gain),
            prompt_tail_clear=int(prompt_tail_clear),
            prompt_head_clear=int(prompt_head_clear),
            push_layers=frozenset(int(x) for x in (push_layers or ())),
            offset_layers=frozenset(int(x) for x in (offset_layers or ())),
            offset_decode_steps=int(offset_decode_steps),
            offset_draw_shape=str(offset_draw_shape),
            offset_random_rank=int(offset_random_rank or 0),
            offset_online=float(offset_online or 0.0),
            online_max_gain=float(online_max_gain or 2.5),
            online_carry=bool(online_carry),
            online_rule_k=float(online_rule_k or 0.0),
            online_rule_start=bool(online_rule_start),
            output_tilt=float(output_tilt or 0.0),
            output_profile=(None if not output_tilt or output_profile is None
                            else output_profile.detach().float().cpu()),
            shadow_protect=bool(shadow_protect),
            steer_split_concentration=float(steer_split_concentration or 0.0),
            arch_mechanism=str(arch_mechanism or ""),
            arch_size=float(arch_size or 0.0),
            arch_per_sentence=bool(arch_per_sentence),
            offset_gamma_spread=float(offset_gamma_spread),
            offset_taper=float(offset_taper),
            guard_direction=str(guard_direction),
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
            want = None if not self.jitter_names else set(self.jitter_names)
            self.gains = [
                float(math.exp(k * float(zi) - 0.5 * k * k))
                if want is None or sp.name in want else 1.0
                for sp, zi in zip(self.specs, z)
            ]
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

    def step_jitter(self, t: int) -> None:
        """Advance f(S_c)'s aim one decode step, if it is set to wander.

        j <- normalise(sqrt(1-w^2) j + w xi), a correlated random walk on the
        sphere with a correlation time of about 1/w steps. At w = 0 nothing
        moves and the story is written under one f, as before; at w = 1 the aim
        is redrawn independently every step, which is white noise on the
        direction.

        Called once per decode step, before the layers are visited, so every
        layer sees the same step of the walk rather than each taking its own.
        """
        w = float(self.jitter_walk)
        if w <= 0 or self.jitter_mode not in ("perp", "rotate"):
            return
        if t == self._jitter_step:
            return
        self._jitter_step = t
        keep = math.sqrt(max(0.0, 1.0 - w * w))
        for lp in self.layer_plans.values():
            if lp.jitter is None:
                continue
            xi = torch.randn(lp.jitter.shape, dtype=lp.jitter.dtype,
                             device=lp.jitter.device)
            j = lp.jitter * keep + xi * w
            lp.jitter = j / j.norm().clamp_min(1e-12)

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
        if self.push_layers and layer not in self.push_layers:
            return None
        h = self.horizon if horizon is None else horizon
        delta = self.jitter_steering(
            layer, lp.steering_delta(t, h, self.specs, self.rms_scale,
                                     gains=self.gains, budget=self.steer_budget)
        )
        if delta is None or self.prefill_gain == 1.0:
            return delta
        return delta * self.prefill_gain

    def plan_offsets(self, n_stories: int, seed: int = 0) -> None:
        """Lay out every story's perturbation at once, spread as far apart as they go.

        The perturbations are drawn from the subspace the model's own states
        occupy, which is low rank -- a few dozen directions out of a couple of
        thousand. Drawn independently, ``n`` points in a rank-``r`` subspace clump:
        for ``n`` of the order of ``r`` the expected largest pairwise similarity is
        far from the smallest it could be, so several stories get nearly the same
        perturbation and the diversity that was paid for is not collected. The
        high-dimensional intuition that two random vectors are near-orthogonal is
        true in the full stream and false here, which is the whole point of
        drawing on the manifold in the first place.

        So the set is chosen rather than sampled. The coefficients are spread over
        the sphere by minimising a Riesz energy -- points repel, the configuration
        settles into the most even one it can find -- and story ``i`` takes point
        ``i``. When ``n <= r`` this converges to a mutually orthogonal set, the
        furthest apart ``n`` directions can be; beyond that it degrades gracefully
        towards a well-separated covering.

        Every perturbation has the same length as before, drawn from the same
        subspace, and is still projected clear of the constraint directions, so
        the constraint cost is unchanged by construction. What changes is only
        that the stories no longer collide with one another by chance.

        Costs a few hundred iterations on an ``n x r`` matrix, once per run.
        """
        self._offset_plan = None
        if (self.offset_draw != "spread" or self.offset_mode == "none"
                or self.offset_gamma <= 0 or n_stories <= 1):
            return
        ranks = {lyr: (lp.offset_basis.shape[1] if lp.offset_basis is not None
                       else self.dim)
                 for lyr, lp in self.layer_plans.items()}
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        plan = {}
        for lyr, r in ranks.items():
            pts = torch.randn(n_stories, r, generator=gen, dtype=torch.float32)
            pts = pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12)
            pts = _spread_on_sphere(pts)
            if self.offset_draw_shape == "manifold":
                sc = self.layer_plans[lyr].offset_scale
                if sc is not None and sc.numel() >= r:
                    w = sc[:r].to(pts.dtype).clamp_min(1e-12)
                    pts = pts * w.unsqueeze(0)
                    pts = pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12)
            plan[lyr] = pts
        self._offset_plan = plan

    def envelope_at(self, t: int) -> float:
        """How large the displacement is at decode step ``t``, as a multiplier.

        "decay" is the usual shape, strongest where generation begins. "rise" is
        the opposite, and it is the one with an argument behind it here: the
        failures this project counts as broken stories are register failures,
        the model answering the reader instead of narrating, and they are
        decided in the first few tokens. Holding the displacement back until the
        story is under way leaves that decision alone and spends the variation
        where the story is already running.
        """
        mode = (self.offset_envelope or "flat").lower()
        if mode == "flat":
            return 1.0
        if mode == "secured":
            # More noise once the story has the things that, once written, stay
            # written. Read from the controller's last look at the story, so it
            # needs error-driven steering to be running.
            got = float(((self.control_state or {}).get("secured", 0.0)) or 0.0)
            return 1.0 + float(self.offset_secured_boost) * got
        span = max(1, int(self.offset_envelope_steps or self.noise_traj_steps))
        frac = min(max(float(t) / span, 0.0), 1.0)
        if mode == "decay":
            return float(0.5 * (1.0 + math.cos(math.pi * frac)))
        if mode == "front":
            extra = float(self.offset_front_gain) - 1.0
            return float(1.0 + extra * 0.5 * (1.0 + math.cos(math.pi * frac)))
        if mode == "rise":
            return float(0.5 * (1.0 - math.cos(math.pi * frac)))
        raise ValueError("offset_envelope must be flat, decay, rise, front or secured; "
                         f"got {self.offset_envelope!r}")

    def resample_offset(self, story_index: Optional[int] = None) -> None:
        """Draw a fresh constant offset for the next generation.

        Call once per story, after seeding. Unlike per-token noise this is held
        fixed for the whole generation, so it accumulates linearly the way the
        steering bias does and actually changes the story. That is also why it has
        to be kept out of the constraint subspace: a constant leak onto a
        constraint direction biases that constraint for the entire story.
        """
        self.resample_jitter()
        if float(getattr(self, "steer_split_concentration", 0.0) or 0.0) > 0:
            # This story's share of the push for each rule. Drawn before any
            # early return, so a plan with no offset still gets one.
            k = len(self.specs)
            conc = torch.full((k,), float(self.steer_split_concentration))
            split = torch.distributions.Dirichlet(conc).sample()
            self.gains = [float(x) * k for x in split]
        self._gamma_this_story = None
        self._anchor_gain = 1.0
        # A fresh story restarts the walk, so one story's wandering aim is not
        # inherited by the next.
        self._jitter_step = -1
        # And the size its noise is written at: an online controller starts
        # every story from the starting length, not from where the last ended.
        self.online_gain = 1.0
        if self.offset_mode == "none" or self.offset_gamma <= 0:
            for lp in self.layer_plans.values():
                lp.offset = None
                lp.offset_traj = None
                lp.offset_length = None
            return

        laid_out = getattr(self, "_offset_plan", None)
        for lyr, lp in self.layer_plans.items():
            dev, dt = lp.basis.device, lp.basis.dtype
            # Relocate the tensors this draw touches to where the steering basis
            # lives, checking each on its own. Whether they were already moved
            # depends on which other code path ran first, and one combination
            # (steering at the prompt only, offset at the prompt) reached here
            # with the offset basis still on the CPU while the coefficient was
            # created on the GPU -- a crash at the first story, same failure
            # family as the feedback_delta relocation above.
            if lp.offset_basis is not None and lp.offset_basis.device != dev:
                lp.offset_basis = lp.offset_basis.to(dev)
            if lp.protect is not None and lp.protect.device != dev:
                lp.protect = lp.protect.to(dev)
            if self.offset_random_rank > 0:
                # This story's own random subspace, drawn from the ambient RNG
                # after seeding like every other draw here. Taken out of the
                # rule directions before use, so the noise cannot move the
                # story along the directions the push holds.
                k = min(int(self.offset_random_rank), self.dim - 1)
                q = torch.linalg.qr(torch.randn(self.dim, k, dtype=torch.float32,
                                                device=dev))[0]
                if self.offset_mode == "orth" and lp.protect is not None:
                    q = complement_basis(q, lp.protect.to(torch.float32))
                lp.offset_basis = q.to(dt)
            # A coefficient vector: taken from the spread-out layout when one has
            # been planned and this story's index is known, drawn independently
            # otherwise.
            if laid_out is not None and story_index is not None and lyr in laid_out:
                pts = laid_out[lyr]
                coeff = pts[int(story_index) % pts.shape[0]].to(device=dev, dtype=dt)
            else:
                coeff = None
            if self.offset_draw_shape == "anchor" and lp.offset_anchors is not None:
                # Aim at one of the model's own stories rather than at a point in
                # the span of the directions summarising them. Every displacement
                # is then a direction in which the model's own writing actually
                # varies, and at full magnitude the state lands on a place it has
                # genuinely been -- which a mixture of principal components, at a
                # large enough magnitude, does not.
                rows = lp.offset_anchors
                which = (0 if story_index is None
                         else int(story_index) % int(rows.shape[0]))
                vec = rows[which].to(device=dev, dtype=dt)
                if lp.anchor_scale is not None:
                    self._anchor_gain = float(lp.anchor_scale[which])
                # The anchors are raw activations, not the complement basis, so
                # they still have to be taken out of the constraint subspace by
                # hand; the basis path gets this for free at build time.
                if self.offset_mode == "orth" and lp.protect is not None:
                    vec = vec - lp.protect @ (lp.protect.t() @ vec)
            elif lp.offset_basis is not None:
                rank = lp.offset_basis.shape[1]

                def _shape(c):
                    # Shape the draw like a real difference between two of the
                    # model's own stories, rather than treating every direction
                    # alike. The basis is ordered by how much between-story
                    # variation each direction carries and the last ones carry
                    # almost none, so an even draw spends as much on them as on
                    # the first and lands outside anything the model does.
                    if self.offset_draw_shape != "manifold":
                        return c
                    sc = lp.offset_scale
                    if sc is None or sc.numel() < rank:
                        return c
                    return c * sc[:rank].to(device=dev, dtype=dt).clamp_min(1e-12)

                if self.noise_beta is not None and coeff is None:
                    # The displacement wanders instead of standing still. One
                    # trajectory per story, drawn from the ambient RNG like
                    # every other draw here, so it is reproducible per story.
                    steps = max(1, int(self.noise_traj_steps))
                    traj = colored_sequence(
                        steps, rank, beta=float(self.noise_beta),
                        fmin_cycles=float(self.noise_fmin_cycles),
                        dtype=torch.float32,
                    ).to(device=dev, dtype=dt)
                    lp.offset_traj = _shape(traj)
                    coeff = lp.offset_traj[0]
                else:
                    lp.offset_traj = None
                    if coeff is None:
                        coeff = _shape(
                            torch.randn(rank, dtype=dt, device=dev))
                vec = lp.offset_basis @ coeff
            else:
                vec = coeff if coeff is not None else torch.randn(self.dim, dtype=dt,
                                                                  device=dev)
                if self.offset_mode == "orth" and lp.protect is not None:
                    vec = vec - lp.protect @ (lp.protect.t() @ vec)
            # This story's displacement size, drawn once per story, uniform
            # about the nominal size so the stories fill the ball rather than
            # sitting on a shell. The mean size is unchanged.
            gamma = self.offset_gamma * float(getattr(self, "_anchor_gain", 1.0) or 1.0)
            if self.offset_gamma_spread > 0:
                if getattr(self, "_gamma_this_story", None) is None:
                    u = float(torch.rand(1).item())
                    self._gamma_this_story = gamma * (
                        1.0 + self.offset_gamma_spread * (2.0 * u - 1.0))
                gamma = float(self._gamma_this_story)
                # Protect this story in proportion to how far it is displaced.
                # The budget renormalises afterwards, so a far-displaced story
                # trades other steering for formatting and a near one does not.
                if self.guard_direction:
                    names = [sp.name for sp in self.specs]
                    if self.guard_direction in names:
                        k = names.index(self.guard_direction)
                        share = gamma / max(self.offset_gamma, 1e-9)
                        g = [1.0] * len(names)
                        g[k] = max(share, 0.05)
                        self.gains = g
            if self.offset_norm == "story" and lp.story_radius:
                # Gamma in units of the data: 1.0 puts the displaced state as
                # far from the average as one of the model's own stories, so
                # below 1.0 is interpolation towards a story it has written and
                # above 1.0 is extrapolation past every one of them.
                vec = vec / vec.norm().clamp_min(1e-12)
                lp.offset = vec * (gamma * lp.story_radius)
                lp.offset_length = float(lp.offset.norm())
            elif self.offset_norm == "fisher":
                lp._offset_unit = None
                # A provisional length. The real one is set per story once the
                # prompt is known, by measuring how far the draw moves the
                # model's predictions -- see noiseegra.fisher_calibration.
                vec = vec / vec.norm().clamp_min(1e-12)
                lp.offset = vec * (0.1 * self.rms_scale * math.sqrt(self.dim))
                lp.offset_length = float(lp.offset.norm())
            elif self.offset_norm == "energy":
                # Fixed length, so gamma means the same thing whatever the rank of
                # the subspace the offset was drawn from. A draw from a rank-r
                # subspace has natural length ~sqrt(r); without this, gamma silently
                # meant a perturbation sqrt(dim/r) times weaker than the same gamma
                # asked of the isotropic arm -- a factor of 8 at rank 64 in a
                # 4096-dimensional stream.
                vec = vec / vec.norm().clamp_min(1e-12)
                lp.offset = vec * (gamma * self.rms_scale * math.sqrt(self.dim))
                lp.offset_length = float(lp.offset.norm())
            else:
                lp.offset = vec * (gamma * self.rms_scale)
                lp.offset_length = float(lp.offset.norm())
        # Sized while writing, with the size carried over: this story starts --
        # prompt included -- at the size the controller settled on in the last
        # one, instead of at the nominal starting length. The direction is still
        # this story's own.
        carried = getattr(self, "_carried_gain", None)
        if (float(getattr(self, "offset_online", 0.0) or 0.0) > 0
                and getattr(self, "online_carry", False) and carried):
            for lp in self.layer_plans.values():
                if lp.offset is not None:
                    lp.offset = lp.offset * float(carried)
                    lp.offset_length = float(lp.offset.norm())

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
        # Relocate every tensor this layer owns, each checked on its own.
        #
        # This used to be guarded by whether the *basis* was on the wrong
        # device, which relocated the rest only on the call that moved the
        # basis. Anything created or redrawn afterwards stayed where it was
        # made, and the crash came at whichever later step first used it. That
        # has now happened twice: the per-story offset, redrawn every story, was
        # left on the CPU from the second story onward; and the protected
        # subspace, which only the per-token noise path reads, was left there by
        # every run that did not use it -- until one did, and died on its first
        # story.
        #
        # Checking each tensor is a handful of device comparisons per decode
        # step and removes the whole family.
        if device is not None:
            lp.relocate(device)

        h = self.horizon if horizon is None else horizon
        # Advance f(S_c)'s aim once per decode step. Guarded on the step number
        # inside, so the first layer to arrive moves it and the rest see the
        # same move.
        self.step_jitter(t)
        delta = None
        if (self.steer_decode and self.steer_mode not in ("feedback", "error")
                and (not self.push_layers or layer in self.push_layers)):
            delta = self.jitter_steering(
                layer, lp.steering_delta(t, h, self.specs, self.rms_scale,
                                         gains=self.gains, budget=self.steer_budget)
            )

        # The per-story offset is a perturbation, so it is gated with the noise
        # rather than with the steering: an entropy gate closes on both together.
        if (with_offset and self.offset_decode and lp.offset is not None
                and (not self.offset_layers or layer in self.offset_layers)
                and (self.offset_decode_steps <= 0 or t < self.offset_decode_steps)):
            # The online controller's multiple of the starting length, 1 when
            # the noise is sized before the story or not at all.
            gain = getattr(self, "online_gain", 1.0)
            gain = 1.0 if gain is None else float(gain)
            off = lp.offset_at(t, self.envelope_at(t) * gain)
            if off is not None:
                delta = off if delta is None else delta + off

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

    def error_delta(
        self,
        layer: int,
        t: int = 0,
        *,
        horizon: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Push each direction by how far its requirement currently sits outside.

        Reads the errors the probe measured on the decoded text one step ago. A
        direction with no error, or one the controller has no probe for, is
        silent. With a budget set, the coefficients are renormalised so the total
        push is the same whether one requirement is failing or four are -- what
        changes is which way it points.
        """
        if self.steer_mode != "error" or not self.control_state:
            return None
        lp = self.layer_plans[layer]
        if device is not None and lp.basis.device != device:
            lp.basis = lp.basis.to(device)
        h = self.horizon if horizon is None else horizon
        errs = self.control_state.get("errors") or {}
        # No probe has reported anything. Treating that as "nothing is watched"
        # would quietly turn this into a constant push over every direction, which
        # is the arm this mode exists to differ from, so it stays silent. The
        # probe seeds a zero for each requirement it watches before the first
        # token, so in a real run this is never the state during decoding.
        if not errs:
            return None

        # The requirements split by shape and the two halves want opposite
        # controls, so the push is built as two components with separate budgets.
        #
        # A direction the controller watches is scaled by how far its requirement
        # currently sits outside its band, and goes silent inside it. A direction
        # the controller has no probe for -- more present tense, simpler
        # vocabulary, where there is no such thing as enough -- keeps a constant
        # coefficient.
        #
        # The budgets have to be separate. Sharing one ceiling over the sum let a
        # large banded error scale the whole vector down, monotone components
        # included, so the constant half's dose moved inversely with how wrong the
        # banded half happened to be: monotone came back at 35% against the 64% a
        # constant push alone reaches. Each half now gets its own dose and neither
        # can rob the other.
        watched = [i for i, s in enumerate(self.specs) if s.name in errs]
        free = [i for i, s in enumerate(self.specs) if s.name not in errs]

        def component(idx, weights, budget, renormalise):
            """Assemble one half of the push, at its own budget."""
            if not idx:
                return None
            if budget is not None and renormalise:
                norm = math.sqrt(sum(w * w for w in weights))
                if norm <= 1e-12:
                    return None
                weights = [w / norm * budget for w in weights]
            full = [0.0] * len(self.specs)
            for i, w in zip(idx, weights):
                full[i] = w * self.rms_scale
            coeffs = torch.tensor(full, dtype=lp.basis.dtype, device=lp.basis.device)
            if bool((coeffs == 0).all()):
                return None
            d = lp.basis @ coeffs
            if budget is not None and not renormalise:
                # A ceiling, not a target. Renormalising the error-driven half
                # would hand a story one word over its limit the same shove as a
                # story fifty over -- a constant push pointed by the sign of the
                # error rather than a proportional controller -- and would divide
                # the gain out entirely: two runs at different gains once came
                # back byte-identical.
                cap = budget * self.rms_scale
                n = float(d.norm())
                if n > cap:
                    d = d * (cap / n)
            return d

        parts = []
        # Constant half: fixed dose, renormalised so the number of directions
        # does not set the strength. Present from the first token, before the
        # probe has decoded anything.
        const = component(
            free,
            [self.specs[i].beta * schedule_factor(self.specs[i].schedule, t, h)
             for i in free],
            self.steer_budget,
            True,
        )
        if const is not None:
            parts.append(const)
        # Error-driven half: proportional to the measured shortfall, capped.
        err_part = component(
            watched,
            [self.specs[i].beta * float(errs[self.specs[i].name])
             * schedule_factor(self.specs[i].schedule, t, h)
             for i in watched],
            self.steer_budget,
            False,
        )
        if err_part is not None:
            parts.append(err_part)
        if not parts:
            return None
        delta = parts[0] if len(parts) == 1 else parts[0] + parts[1]
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
            "noise_beta": self.noise_beta,
            "noise_fmin_cycles": self.noise_fmin_cycles,
            "offset_envelope": self.offset_envelope,
            "offset_online": self.offset_online,
            "online_max_gain": self.online_max_gain,
            "online_carry": self.online_carry,
            "online_rule_k": self.online_rule_k,
            "online_rule_start": self.online_rule_start,
            "output_tilt": self.output_tilt,
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
            "jitter_walk": self.jitter_walk,
            "steer_decode": self.steer_decode,
            "steer_budget": self.steer_budget,
            "steer_mode": self.steer_mode,
            "feedback_cap": self.feedback_cap,
            "direction_source": self.direction_source,
            "offset_rank": self.layer_plans[self.layers[0]].report.get("offset_rank"),
            "horizon": self.horizon,
            "steer_prefill": self.steer_prefill,
            "prefill_gain": self.prefill_gain,
            "prompt_tail_clear": self.prompt_tail_clear,
            "prompt_head_clear": self.prompt_head_clear,
            "push_layers": sorted(self.push_layers),
            "offset_layers": sorted(self.offset_layers),
            "offset_decode_steps": self.offset_decode_steps,
            "offset_draw_shape": self.offset_draw_shape,
            "offset_gamma_spread": self.offset_gamma_spread,
            "guard_direction": self.guard_direction,
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
