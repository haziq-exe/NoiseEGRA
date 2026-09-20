"""Measuring a perturbation by how far it moves the model's own predictions.

Round 16 recorded a negative result that never got an answer: the whole set of
per-story perturbations was laid out in advance by repulsion so it covered the
perturbation subspace evenly, and every matched pair came back slightly worse
than drawing each one independently. The reason given was that "covering the
offset subspace evenly does not cover the output space evenly: distance in the
subspace the perturbation is drawn from does not predict distance between the
stories that come out."

That is a statement about a metric, and it has a standard fix. Arvanitidis et
al. (AISTATS 2022) treat a decoder as a map into the space of output
distributions, which carries the Fisher-Rao metric, and pull that metric back to
the latent space: M(z) = J(z)' I(h(z)) J(z). Locally the Fisher-Rao metric is
the KL divergence between the perturbed and unperturbed outputs, so the pullback
can be measured by finite differences without any Jacobian at all.

Our output is a categorical distribution over the vocabulary, and for
categorical distributions the Fisher-Rao distance has a closed form: the angle
between the square roots of the two probability vectors on the unit sphere. So
the whole construction costs a handful of forward passes and an arccos.

What this buys. The perturbation basis can be rescaled so that a step of a given
length changes the model's own next-token distribution by the same amount
whatever direction it points in. Two things follow. Covering the subspace evenly
now means covering the model's predictions evenly, which is what round 16 wanted
and did not have. And the size of the perturbation stops being measured in units
of one model's activations -- it is measured in how far that model's predictions
move -- so the same setting means the same thing on a different model, which is
where carrying a setting from the 1.7B to the 8B went wrong.

The metric is estimated at a finite step rather than infinitesimally, because
the perturbation we actually apply is not infinitesimal. That makes it a secant
approximation to the pullback metric, which is the honest description.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between categorical distributions, 2*arccos(sum sqrt(pq)).

    The simplex under the Fisher-Rao metric is a sphere of radius 2 under the
    square-root map, so the distance is an angle and is bounded by pi. Both
    arguments must be probability vectors over the same support.
    """
    p = p.double()
    q = q.double()
    bc = (p.clamp_min(0).sqrt() * q.clamp_min(0).sqrt()).sum(dim=-1)
    return 2.0 * torch.acos(bc.clamp(-1.0, 1.0))


def subspace_metric(
    predict: Callable[[Optional[torch.Tensor]], torch.Tensor],
    basis: torch.Tensor,
    *,
    step: float = 1.0,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """The pulled-back metric on the span of `basis`, by finite differences.

    `basis` is dim x k, its columns the directions a perturbation is built from.
    `predict(delta)` returns the model's next-token probabilities with `delta`
    added to the residual stream, and `predict(None)` returns them unperturbed.

    The squared Fisher-Rao distance of a displacement is a quadratic form in its
    coefficients, d^2(Uc) ~= c' G c, so G is recovered from distances alone --
    the diagonal from one direction at a time, the off-diagonal by polarisation.
    That is k(k+1)/2 + 1 calls to `predict` and no gradients.

    `step` is the size of the probe displacement. The metric of a curved space
    is only constant infinitesimally, so probing at the size the perturbation is
    actually used at describes the geometry where it is actually applied.
    """
    if basis.dim() != 2:
        raise ValueError("basis must be dim x k")
    k = basis.shape[1]
    if k < 1:
        raise ValueError("basis must have at least one column")
    if step <= 0:
        raise ValueError("step must be positive")

    base = predict(None)
    cols = [basis[:, i] * step for i in range(k)]

    def d2(delta):
        return float(fisher_rao_distance(base, predict(delta))) ** 2

    diag = [d2(c) for c in cols]
    g = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        g[i, i] = diag[i]
        for j in range(i + 1, k):
            both = d2(cols[i] + cols[j])
            # d2(u+v) = d2(u) + d2(v) + 2 u'Gv
            g[i, j] = g[j, i] = 0.5 * (both - diag[i] - diag[j])

    g = g / (step * step)
    # A finite-difference estimate of a positive definite form need not come out
    # positive definite, so the spectrum is floored rather than the result being
    # trusted blindly. Keeping the eigenvectors and lifting only the eigenvalues
    # preserves which directions the model is sensitive along.
    g = 0.5 * (g + g.T)
    evals, evecs = torch.linalg.eigh(g)
    floor = float(evals.max()) * jitter if float(evals.max()) > 0 else jitter
    evals = evals.clamp_min(floor)
    return (evecs @ torch.diag(evals) @ evecs.T)


def whiten_basis(basis: torch.Tensor, metric: torch.Tensor,
                 *, max_gain: float = 8.0) -> torch.Tensor:
    """Rescale `basis` so a unit coefficient vector is a unit Fisher-Rao step.

    With U the basis and G the pulled-back metric, the whitened basis is
    U G^(-1/2): a coefficient vector c of length one then gives a displacement
    whose Fisher-Rao length is one, whichever direction c points in. Drawing c
    uniformly on the sphere afterwards therefore spreads the perturbations
    evenly over the model's predictions rather than evenly over its activations.

    `max_gain` caps how far any direction is stretched relative to the one the
    model responds to most strongly, and it is not optional in practice. Exact
    whitening divides by the square root of each eigenvalue, so a direction the
    model barely responds to is stretched without limit -- and on a real
    subspace some directions sit below what the probe can resolve at all. The
    displacement would then spend almost its whole length pushing somewhere the
    model does not notice, which is both useless and far outside anything the
    model's own stories do. Capping the gain keeps the correction where there is
    signal and leaves the rest alone.
    """
    if basis.shape[1] != metric.shape[0] or metric.shape[0] != metric.shape[1]:
        raise ValueError("metric must be k x k for a dim x k basis")
    if max_gain < 1.0:
        raise ValueError("max_gain is a stretch relative to the stiffest "
                         "direction and cannot be below 1")
    evals, evecs = torch.linalg.eigh(metric.double())
    top = float(evals.max())
    if top <= 0:
        return basis
    floor = top / (float(max_gain) ** 2)
    evals = evals.clamp_min(floor)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    # Scaled so the stiffest direction keeps the length it had. Without this the
    # whole basis is divided by the metric's units, and `gamma` would stop
    # meaning a number of story-distances.
    inv_sqrt = inv_sqrt * (top ** 0.5)
    return (basis.double() @ inv_sqrt).to(basis.dtype)


def anisotropy(metric: torch.Tensor) -> float:
    """How unequal the directions are: the ratio of largest to smallest scale.

    One means the model's predictions move equally along every direction of the
    subspace, and whitening would change nothing. A large value means an
    unwhitened draw spends most of its length in directions the model barely
    notices, which is the failure this module exists to fix. Reported as a
    length ratio rather than an eigenvalue ratio, so it is in the same units as
    the perturbation size.
    """
    evals = torch.linalg.eigvalsh(metric.double()).clamp_min(0)
    lo = float(evals.min())
    hi = float(evals.max())
    if hi <= 0:
        return 1.0
    if lo <= 0:
        # Some direction moves the predictions by nothing the probe can
        # resolve. Reporting infinity is true and useless; what the caller
        # needs to know is that the subspace is degenerate, which
        # `resolved_directions` answers.
        return float("inf")
    return (hi / lo) ** 0.5


def resolved_directions(metric: torch.Tensor, *, floor_ratio: float = 1e-4) -> int:
    """How many directions the probe could actually measure a response along.

    A direction below this is one the model does not visibly react to at the
    probe size. Whitening would stretch it hardest of all, which is why the
    gain is capped, and a subspace with few resolved directions is one where
    this correction has little to work with.
    """
    evals = torch.linalg.eigvalsh(metric.double())
    top = float(evals.max())
    if top <= 0:
        return 0
    return int((evals > top * floor_ratio).sum())
