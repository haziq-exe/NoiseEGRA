#!/usr/bin/env python
"""Extract and cache per-constraint steering vectors for one model.

    python scripts/build_steering_vectors.py --model Fanar
    python scripts/build_steering_vectors.py --model-id microsoft/Phi-4-mini-instruct \
        --model-name Phi-4-mini --layers 12 21

Writes ``steering_vectors/<model_name>.pt`` (override with --out) holding, per
constraint and per layer: the mean-difference direction, the top principal
components of the per-item differences (used to enlarge the protected subspace)
and extraction diagnostics.

Run this once per model. ``run_orthosteer_experiment.py`` reloads the cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noiseegra.defaults import MODEL_HF_IDS, MODEL_LAYER_RANGES  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor,
    load_pairs,
)


WRAPPERS = {
    "inceptionai/Jais-2-8B-Chat": ("noiseegra.models.Jais", "Jais"),
    "FreedomIntelligence/AceGPT-v2-8B-Chat": ("noiseegra.models.AceGPT", "AceGPT"),
    "humain-ai/ALLaM-7B-Instruct-preview": ("noiseegra.models.Allam", "Allam"),
    "QCRI/Fanar-1-9B-Instruct": ("noiseegra.models.Fanar", "Fanar"),
}


def build_model(model_id: str, use_aeni: bool = False, dtype=None, wrapper_for: str = None):
    """Prefer the per-model wrapper (custom chat templates), else generic EGRA.

    ``wrapper_for`` lets a local snapshot directory still pick up the right
    wrapper, e.g. build_model("/kaggle/input/acegpt", wrapper_for=<hf id>).
    """
    import importlib

    from noiseegra.EGRA_functions import EGRA

    key = wrapper_for or model_id
    if key in WRAPPERS:
        module, cls = WRAPPERS[key]
        klass = getattr(importlib.import_module(module), cls)
        if key == model_id:
            return klass(dtype=dtype)
        # Local path: bypass the wrapper's pinned id but keep its chat template.
        obj = klass.__new__(klass)
        EGRA.__init__(obj, model_id, use_AENI=use_aeni, dtype=dtype)
        return obj
    return EGRA(model_id, use_AENI=use_aeni, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_HF_IDS), help="paper model key")
    ap.add_argument("--model-id", help="explicit Hugging Face id (overrides --model)")
    ap.add_argument("--model-name", help="short name used in filenames (defaults to --model)")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"),
                    help="layer range [LO, HI); defaults to the paper range for --model")
    ap.add_argument("--pairs", help="custom contrast-pair JSON (defaults to the bundled Arabic set)")
    ap.add_argument("--constraints", nargs="*", help="subset of constraints to extract")
    ap.add_argument("--pca-rank", type=int, default=8,
                    help="principal components stored per constraint (default 8); "
                         "these enlarge the protected subspace at experiment time")
    ap.add_argument("--out", help="output .pt path")
    args = ap.parse_args()

    if not args.model and not args.model_id:
        ap.error("pass --model or --model-id")

    model_id = args.model_id or MODEL_HF_IDS[args.model]
    model_name = args.model_name or args.model or model_id.split("/")[-1]

    if args.layers:
        lo, hi = args.layers
    elif args.model:
        lo, hi = MODEL_LAYER_RANGES[args.model]
    else:
        ap.error("pass --layers when using --model-id")
    layers = list(range(lo, hi))

    print(f"model      : {model_id}")
    print(f"layers     : {layers}")
    print(f"pca rank   : {args.pca_rank}\n")

    model = build_model(model_id)
    constraints = load_pairs(args.pairs)

    vectors = SteeringVectorExtractor(model).extract(
        constraints,
        layers,
        pca_rank=args.pca_rank,
        only=args.constraints,
    )
    vectors.meta["pairs_path"] = args.pairs or "bundled"
    vectors.meta["model_name"] = model_name

    out = Path(args.out) if args.out else Path("steering_vectors") / f"{model_name}.pt"
    vectors.save(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
