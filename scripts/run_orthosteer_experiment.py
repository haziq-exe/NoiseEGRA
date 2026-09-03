#!/usr/bin/env python
"""Run the orthogonal-constraint-steering ablation.

    # once per model
    python scripts/build_steering_vectors.py --model Fanar

    # then
    python scripts/run_orthosteer_experiment.py --model Fanar --suite core --num-stories 50

Suites
------
``method`` just the proposed method, a single condition -- for a quick test run
           that does not regenerate the Baseline and L-Res references.
``noise``  the four steering variants only -- no noise, noise with the constraint
           directions removed, ordinary noise, and noise confined to the constraint
           directions. This is the ablation itself, without regenerating references.
``core``   the main comparison: Baseline, published L-Res (isotropic noise),
           steering with no noise, and steering + {orth, iso, para} noise.
           ``para`` is the destructive control that confines all the noise energy
           to the constraint subspace.
``ortho``  steering + orthogonal noise under each orthogonalisation of the
           steering vectors: none (naive sum), Gram-Schmidt, Loewdin.
``beta``   systematically relax the constraint pressure: a sweep over beta.
``loo``    leave-one-out over the constraints, to see which one carries the effect
           and whether they interfere.
``all``    every suite above.

Everything is written under ``--output-dir`` in the same layout the rest of the
repo uses: one CSV of stories per run, per-run RESULTS/*.txt (creativity +
published constraint report + the exact LLM-free constraint report), and a
combined <model>_RESULTS.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra import prompts  # noqa: E402
from noiseegra.defaults import MODEL_HF_IDS, MODEL_LAYER_RANGES, RMS_ALPHA  # noqa: E402
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402
from noiseegra.setup_experiment import make_specs, run_story_experiments  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_steering_vectors import build_model  # noqa: E402

DEFAULT_CONSTRAINTS = ["closure", "present_tense", "simple_register"]
# The closure direction ramps up: push the model to wrap up harder the longer the
# story has already run. The other two are properties of every sentence, so they
# stay constant.
DEFAULT_SCHEDULES = {"closure": "ramp", "present_tense": "constant", "simple_register": "constant"}


def make_plan(
    vectors: SteeringVectorSet,
    layers,
    names,
    *,
    beta,
    rms_scale,
    orthogonalize="lowdin",
    noise_mode="orth",
    noise_alpha=RMS_ALPHA,
    protect_rank=0,
    schedules=None,
    horizon=200,
    steer_prefill=False,
    noise_norm_match="energy",
) -> SteeringPlan:
    schedules = schedules or DEFAULT_SCHEDULES
    betas = beta if isinstance(beta, dict) else {n: float(beta) for n in names}
    specs = [
        ConstraintSpec(n, beta=betas.get(n, 0.0), schedule=schedules.get(n, "constant"))
        for n in names
    ]
    extra = vectors.protect_extra(names, protect_rank) if protect_rank > 0 else None
    return SteeringPlan.build(
        vectors.vectors,
        layers,
        specs,
        rms_scale=rms_scale,
        orthogonalize=orthogonalize,
        noise_mode=noise_mode,
        noise_alpha=noise_alpha,
        noise_norm_match=noise_norm_match,
        horizon=horizon,
        steer_prefill=steer_prefill,
        protect_extra=extra,
    )


def build_suite(name, vectors, layers, names, rms_scale, args):
    """Return (list_of_spec_dicts, description) for one suite."""
    common = dict(
        vectors=vectors, layers=layers, names=names, rms_scale=rms_scale,
        protect_rank=args.protect_rank, horizon=args.horizon,
        steer_prefill=args.steer_prefill,
    )
    resid_std = args.alpha * rms_scale
    items = []

    if name == "method":
        # Just the proposed method, one condition. For a quick smoke run that does
        # not regenerate the Baseline and L-Res references you already have.
        items = [{"plan": make_plan(beta=args.beta, noise_mode="orth",
                                    noise_alpha=args.alpha, **common)}]
        return items, "the proposed method only (steering + constraint-free noise)"

    # The four steering variants, differing only in where the noise is allowed to sit.
    noise_arms = [
        {"plan": make_plan(beta=args.beta, noise_mode=mode,
                           noise_alpha=0.0 if mode == "none" else args.alpha, **common)}
        for mode in ("none", "orth", "iso", "para")
    ]

    if name == "compare":
        # The minimal head-to-head: unmodified generation vs. the proposed method.
        return ([ "baseline", noise_arms[1] ],
                "Baseline vs. the proposed method")

    if name == "noise":
        return noise_arms, ("the four steering variants: no noise / noise with constraint "
                            "directions removed / ordinary noise / noise confined to the "
                            "constraint directions")

    if name == "core":
        items += [
            "baseline",
            {
                "mode": "residual_stream_noise",
                "residual_layers": layers,
                "residual_noise_std": resid_std,
                "disable_residual_noise_decay": True,
            },
        ]
        items += noise_arms
        return items, "Baseline / L-Res / the four steering variants"

    if name == "ortho":
        items = [
            {"plan": make_plan(beta=args.beta, orthogonalize=method,
                               noise_mode="orth", noise_alpha=args.alpha, **common)}
            for method in ("none", "gram_schmidt", "lowdin")
        ]
        return items, "orthogonalisation of the steering vectors: none / GS / Loewdin"

    if name == "beta":
        items = [
            {"plan": make_plan(beta=b, noise_mode="orth", noise_alpha=args.alpha, **common)}
            for b in args.beta_sweep
        ]
        return items, f"beta sweep {args.beta_sweep}"

    if name == "loo":
        for drop in names:
            kept = [n for n in names if n != drop]
            sub = dict(common)
            sub["names"] = kept
            items.append(
                {"plan": make_plan(beta=args.beta, noise_mode="orth",
                                   noise_alpha=args.alpha, **sub)}
            )
        return items, "leave-one-out over constraints"

    raise ValueError(f"unknown suite '{name}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_HF_IDS))
    ap.add_argument("--model-id")
    ap.add_argument("--model-name")
    ap.add_argument("--vectors", help="path to the .pt from build_steering_vectors.py")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--suite", nargs="+", default=["core"],
                    choices=["compare", "method", "noise", "core", "ortho", "beta", "loo", "all"])
    ap.add_argument("--constraints", nargs="*", default=DEFAULT_CONSTRAINTS)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="steering strength as a multiple of median block RMS (default 1.0)")
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--alpha", type=float, default=RMS_ALPHA,
                    help=f"noise strength, paper default {RMS_ALPHA}")
    ap.add_argument("--protect-rank", type=int, default=8,
                    help="principal components protected per constraint on top of the mean "
                         "direction. 0 protects only the C mean directions, which in a "
                         "4096-d stream removes ~0.07%% of a random draw and makes 'orth' "
                         "indistinguishable from 'iso' (default 8)")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--steer-prefill", action="store_true",
                    help="also steer across the prompt positions (CAA convention)")
    ap.add_argument("--num-stories", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output-dir", default="experiment_results/OrthoSteer")
    ap.add_argument("--sanity-check", type=int, default=3,
                    help="print the first N stories of each run (0 disables)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the plans, print the geometry and the run ids, generate nothing")
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

    vec_path = Path(args.vectors) if args.vectors else Path("steering_vectors") / f"{model_name}.pt"
    if not vec_path.is_file():
        raise SystemExit(
            f"no steering vectors at {vec_path}. Run:\n"
            f"  python scripts/build_steering_vectors.py --model {args.model or model_name}"
        )
    vectors = SteeringVectorSet.load(vec_path)
    missing = [n for n in args.constraints if n not in vectors.vectors]
    if missing:
        raise SystemExit(f"{vec_path} has no vectors for {missing}; has {vectors.names}")

    print(f"model    : {model_id}")
    print(f"layers   : {layers}")
    print(f"vectors  : {vec_path}")
    vectors.print_report()

    model = build_model(model_id)

    # Same calibration as the paper: sigma = alpha * median_layer(block RMS).
    print("\n[calibration] measuring block RMS ...")
    cal = RMSCalibrator(model)
    rms = cal.collect_block_rms(
        [
            {"role": "system", "content": prompts.SYS_ZERO_SHOT},
            {"role": "user", "content": prompts.PROMPT_ZERO_SHOT},
        ],
        layers=layers,
    )
    rms_scale = float(np.median(list(rms.values())))
    print(f"[calibration] median block RMS = {rms_scale:.6g}  "
          f"-> noise sigma = {args.alpha * rms_scale:.6g}, "
          f"steering magnitude = {args.beta * rms_scale:.6g} per constraint")

    suites = ["core", "ortho", "beta", "loo"] if "all" in args.suite else args.suite
    items, descriptions = [], []
    for suite in suites:
        built, desc = build_suite(suite, vectors, layers, args.constraints, rms_scale, args)
        items.extend(built)
        descriptions.append(f"{suite}: {desc} ({len(built)} runs)")

    print("\n=== suites ===")
    for d in descriptions:
        print("  " + d)

    # Thread the shared sampling settings onto every run.
    normalised = []
    for it in items:
        d = {"mode": it} if isinstance(it, str) else dict(it)
        d.setdefault("temperature", args.temperature)
        d.setdefault("max_new_tokens_plan", args.max_new_tokens)
        d.setdefault("max_new_tokens_story", args.max_new_tokens)
        normalised.append(d)

    # Overlapping suites can produce the same condition twice; run ids are the
    # output filenames, so a duplicate would append two runs into one CSV.
    from noiseegra.setup_experiment import _spec_to_run_id

    specs, seen = [], set()
    for spec in make_specs(*normalised):
        rid = _spec_to_run_id(model_name, spec)
        if rid in seen:
            print(f"[dedupe] skipping duplicate condition: {rid}")
            continue
        seen.add(rid)
        specs.append(spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_id": model_id,
        "model_name": model_name,
        "layers": layers,
        "rms_scale": rms_scale,
        "alpha": args.alpha,
        "beta": args.beta,
        "constraints": args.constraints,
        "protect_rank": args.protect_rank,
        "suites": suites,
        "num_stories": [args.start, args.start + args.num_stories],
        "runs": [
            {
                "run_id": _spec_to_run_id(model_name, spec),
                "plan": (
                    spec.steering_plan.describe()
                    if spec.steering_plan is not None
                    else None
                ),
            }
            for spec in specs
        ],
    }
    (out_dir / f"{model_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    if args.dry_run:
        print(f"\n=== {len(specs)} run ids (dry run) ===")
        for spec in specs:
            print("  " + _spec_to_run_id(model_name, spec))
        return

    run_story_experiments(
        model=model,
        model_name=model_name,
        num_stories=(args.start, args.start + args.num_stories),
        specs=specs,
        output_dir=str(out_dir),
        sanity_check=args.sanity_check > 0,
        sanity_check_n=args.sanity_check,
    )
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
