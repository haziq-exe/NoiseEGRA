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
NORM_MATCH_MODES = ("energy", "none")
SCHEDULES = ("constant", "cosine_decay", "ramp", "linear_decay")


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
    """
    if kind == "constant":
        return 1.0
    if horizon <= 0:
        return 0.0
    if kind == "cosine_decay":
        return _cosine_noise_decay(t, horizon)
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

    def steering_delta(
        self,
        t: int,
        horizon: int,
        specs: Sequence[ConstraintSpec],
        rms_scale: float,
    ) -> Optional[torch.Tensor]:
        """Sum_c beta_c * f_c(t) * rms_scale * s_c  ->  a single (dim,) vector."""
        coeffs = torch.tensor(
            [
                spec.beta * schedule_factor(spec.schedule, t, horizon) * rms_scale
                for spec in specs
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
        offset_basis: Optional[Mapping[int, torch.Tensor]] = None,
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

            layer_plans[layer] = LayerPlan(
                layer=layer, basis=basis, protect=protect, report=report, offset_basis=ob
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
        return self

    def resample_offset(self) -> None:
        """Draw a fresh constant offset for the next generation.

        Call once per story, after seeding. Unlike per-token noise this is held
        fixed for the whole generation, so it accumulates linearly the way the
        steering bias does and actually changes the story. That is also why it has
        to be kept out of the constraint subspace: a constant leak onto a
        constraint direction biases that constraint for the entire story.
        """
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

        h = self.horizon if horizon is None else horizon
        delta = lp.steering_delta(t, h, self.specs, self.rms_scale)

        # The per-story offset is a perturbation, so it is gated with the noise
        # rather than with the steering: prefill never sees it, and an entropy
        # gate closes on both together.
        if with_offset and lp.offset is not None:
            delta = lp.offset if delta is None else delta + lp.offset

        if with_noise and self.noise_mode != "none" and self.noise_alpha > 0:
            sigma = self.sigma * schedule_factor(self.noise_schedule, t, h)
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
        print(f"constraints      : {names}")
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
                  f"rank={info['offset_rank']}")
        print(f"protected rank   : {info['protect_rank']} of {info['dim']} "
              f"({100.0 * info['protect_rank'] / max(info['dim'], 1):.3f}% of the stream)")
        print(f"steer at prefill : {info['steer_prefill']}")
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
