#!/usr/bin/env python
"""How much do the directions stories differ along overlap the constraint directions?

    python scripts/subspace_overlap.py --input-dir experiments/qwen-round2/state/Qwen3-8B

The orthogonal-noise argument is that a perturbation drawn in a ``k``-dimensional
protected subspace's complement loses only ``k/dim`` of its energy, so projecting
is nearly free. That holds only if the subspace the perturbation is drawn from is
unrelated to the constraint subspace. This measures whether it is.

Two numbers per layer:

``inside``   the share of the activation basis's energy that lies in the protected
             span, against the ``k/dim`` a random subspace of the same rank would
             give. Above chance means the axes along which the model's own stories
             differ from one another are partly the axes the constraints live on.
``max cos``  the largest principal cosine between the two subspaces: the single
             most aligned pair of directions. 1.0 would mean one direction is
             shared outright.

This is a correlation, not a cause. The experiment that makes it causal is
``--suite ablate``: the same offset drawn with the constraint span removed and
with it left in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from noiseegra.activation_basis import StoryAxes  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from noiseegra.subspace import orthonormal_basis, orthonormalize  # noqa: E402


def protected_basis(vectors: SteeringVectorSet, names, layer: int, protect_rank: int):
    """The span the noise is kept out of: the steering directions plus their spread.

    Built the same way :meth:`SteeringPlan.build` builds it, so the number this
    script reports is the overlap with the subspace that is actually protected at
    generation time and not with a lookalike.
    """
    cols = [vectors.vectors[n][layer].detach().to(torch.float32).flatten() for n in names]
    basis, _ = orthonormalize(torch.stack(cols, dim=1), method="lowdin")
    parts = [basis]
    if protect_rank > 0:
        for n in names:
            comp = vectors.components.get(n, {}).get(layer)
            if comp is None:
                raise SystemExit(
                    f"no principal components stored for '{n}'; re-extract with "
                    f"pca_rank >= {protect_rank}."
                )
            parts.append(comp[:, :protect_rank].to(torch.float32))
    return orthonormal_basis(torch.cat(parts, dim=1))


def overlap(act: torch.Tensor, protect: torch.Tensor):
    """(energy share inside the protected span, largest principal cosine)."""
    q = torch.linalg.qr(act.to(torch.float32))[0]        # orthonormal, whatever came in
    m = protect.t() @ q                                   # (k, r)
    share = float((m ** 2).sum() / q.shape[1])            # mean over the r directions
    max_cos = float(torch.linalg.svdvals(m).max())
    return share, max_cos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="directory holding steering_<model>.pt and actpcs_<kind>_<model>.pt")
    ap.add_argument("--protect-rank", type=int, default=8,
                    help="principal components per constraint added to the protected "
                         "span; must match the run being explained")
    ap.add_argument("--names", nargs="*",
                    help="steered directions; default: every direction in the file")
    args = ap.parse_args()

    d = Path(args.input_dir)
    vec_paths = sorted(d.glob("steering_*.pt"))
    if not vec_paths:
        raise SystemExit(f"no steering_*.pt in {d}")
    vectors = SteeringVectorSet.load(vec_paths[0])
    names = args.names or vectors.names
    layers = sorted(set(vectors.vectors[names[0]]))

    bases = {}
    for p in sorted(d.glob("actpcs_*.pt")):
        kind = p.stem.split("_")[1]
        blob = torch.load(p, map_location="cpu", weights_only=False)
        bases[kind] = blob.basis if isinstance(blob, StoryAxes) else blob
    if not bases:
        raise SystemExit(f"no actpcs_*.pt in {d}")

    print(f"steering vectors : {vec_paths[0].name}")
    print(f"directions       : {', '.join(names)}")
    print(f"protected span   : {len(names)} directions + {args.protect_rank} components each")
    print(f"activation bases : {', '.join(sorted(bases))}\n")

    header = f"{'basis':>8} {'layer':>6} {'rank':>5} {'protected':>10} {'inside':>8} {'chance':>8} {'x chance':>9} {'max cos':>8}"
    print(header)
    print("-" * len(header))
    for kind in sorted(bases):
        per_layer = bases[kind]
        shares, chances, cosines = [], [], []
        for layer in layers:
            if layer not in per_layer:
                continue
            protect = protected_basis(vectors, names, layer, args.protect_rank)
            act = per_layer[layer].to(torch.float32)
            share, max_cos = overlap(act, protect)
            dim, k, r = act.shape[0], protect.shape[1], act.shape[1]
            chance = k / dim
            shares.append(share); chances.append(chance); cosines.append(max_cos)
            print(f"{kind:>8} {layer:>6} {r:>5} {k:>10} {share:>7.2%} {chance:>7.2%} "
                  f"{share / chance:>8.1f}x {max_cos:>8.3f}")
        if shares:
            n = len(shares)
            print(f"{kind:>8} {'mean':>6} {'':>5} {'':>10} {sum(shares)/n:>7.2%} "
                  f"{sum(chances)/n:>7.2%} {(sum(shares)/n)/(sum(chances)/n):>8.1f}x "
                  f"{max(cosines):>8.3f}\n")

    print("inside   share of the activation basis's energy lying in the protected span")
    print("chance   what a random subspace of the same rank would give, k/dim")
    print("max cos  largest principal cosine between the two subspaces")
    print("\nAbove chance means the axes stories differ along are partly the axes the")
    print("constraints live on, so projecting the constraint span out of a perturbation")
    print("removes real diversity rather than a negligible slice. Correlation only --")
    print("`--suite ablate` is the experiment that says whether it is causal.")


if __name__ == "__main__":
    main()
