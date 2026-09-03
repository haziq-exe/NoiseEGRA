#!/usr/bin/env python
"""Resumable runner for the orthogonal-steering ablation (built for Kaggle 2x T4).

Every generated story is written straight into a single JSON checkpoint, so if
the notebook dies, times out, or you stop it, re-running the exact same command
picks up from the next unfinished story. Nothing is regenerated.

    python scripts/kaggle_orthosteer.py --model Fanar --suite core --num-stories 50

Or from inside a notebook, which keeps the model in memory so a retry does not
reload 16 GB of weights:

    from kaggle_orthosteer import load_model, run, make_args
    model = load_model("Jais")                       # once
    run(model, make_args(model="Jais", suite=["method"], num_stories=5))

Outputs, all under --out:
    state.json                 the checkpoint (also the raw story archive)
    steering_<model>.pt        cached steering vectors (built once)
    <run_id>.csv               one story per row, readable by score_orthosteer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from noiseegra import prompts  # noqa: E402
from noiseegra.defaults import MODEL_HF_IDS, MODEL_LAYER_RANGES, RMS_ALPHA  # noqa: E402
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402
from noiseegra.setup_experiment import _spec_mode, _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor,
    SteeringVectorSet,
    load_pairs,
)

from build_steering_vectors import build_model  # noqa: E402
from run_orthosteer_experiment import DEFAULT_CONSTRAINTS, build_suite  # noqa: E402


def seed_for_story(x: int) -> int:
    """Same per-story seed the published experiments use."""
    return 42 * (x ** 7) * 217


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"rms_scale": {}, "runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    # Older checkpoints stored a single activation scale for the whole file. It
    # belongs to whichever model ran first, and reusing it for a second model
    # mis-scales every steering and noise vector, so drop it and recalibrate.
    if not isinstance(state.get("rms_scale"), dict):
        if state.get("rms_scale") is not None:
            print("[warn] checkpoint predates per-model calibration; recalibrating. "
                  "Stories already generated are kept, but any generated for a model "
                  "other than the first one in this folder used the wrong scale.")
        state["rms_scale"] = {}
    return state


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic: a kill mid-write cannot corrupt the checkpoint


def write_csvs(out: Path, state: dict) -> None:
    import csv

    for run_id, stories in state["runs"].items():
        rows = [stories[k] for k in sorted(stories, key=int)]
        with (out / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for text in rows:
                writer.writerow([text])


def generate_one(model, spec, mode, story_prompt, seed, max_new_tokens):
    if mode == "orthogonal_steering":
        return model.generate_with_orthogonal_steering(
            story_prompt, spec.steering_plan,
            max_new_tokens=max_new_tokens, do_sample=spec.do_sample,
            temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, seed=seed,
        )
    if mode == "residual_stream_noise":
        return model.generate_with_residual_stream_noise(
            story_prompt,
            residual_layers=list(spec.residual_layers or []),
            residual_noise_std=spec.residual_noise_std,
            residual_noise_decay=spec.residual_noise_decay,
            max_noise_tokens=spec.max_noise_tokens,
            disable_residual_noise_decay=spec.disable_residual_noise_decay,
            max_new_tokens=max_new_tokens, do_sample=spec.do_sample,
            temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, seed=seed,
        )
    if mode == "baseline":
        return model.generate(
            story_prompt, max_new_tokens=max_new_tokens, do_sample=spec.do_sample,
            temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, seed=seed,
        )
    raise ValueError(f"kaggle_orthosteer does not handle mode '{mode}'")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Fanar", choices=sorted(MODEL_HF_IDS))
    ap.add_argument("--model-id", help="HF id or a local snapshot directory; "
                                       "overrides the id --model maps to")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"),
                    help="layer range [LO, HI); defaults to the paper range for --model")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"],
                    help="auto lets each model wrapper choose (Jais requires bfloat16; "
                         "the rest default to float16). Only override deliberately.")
    ap.add_argument("--suite", nargs="+", default=["core"],
                    choices=["compare", "method", "noise", "core", "ortho", "beta", "loo", "all"])
    ap.add_argument("--num-stories", type=int, default=50)
    ap.add_argument("--out", default="/kaggle/working/orthosteer")
    ap.add_argument("--constraints", nargs="*", default=DEFAULT_CONSTRAINTS)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--alpha", type=float, default=RMS_ALPHA)
    ap.add_argument("--protect-rank", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--steer-prefill", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pca-rank", type=int, default=8)
    return ap


def make_args(**overrides) -> argparse.Namespace:
    """Defaults from the CLI parser, with any keyword overridden.

    Lets a notebook call ``run(model, make_args(suite=["method"], num_stories=5))``
    without going through the command line.
    """
    args = _parser().parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown setting '{key}'; see --help for the full list")
        setattr(args, key, value)
    return args


def load_model(model: str = "Fanar", model_id: str = None, dtype=None):
    """Load a model once, so it can be reused across repeated run() calls.

    ``model`` selects the paper preset (layers + chat template). ``model_id``
    optionally points somewhere else for the weights themselves -- a different
    Hugging Face id, or a local directory you downloaded into yourself.
    ``dtype=None`` lets the wrapper choose (Jais pins bfloat16).
    """
    hf_id = MODEL_HF_IDS[model]
    obj = build_model(model_id or hf_id, dtype=dtype, wrapper_for=hf_id)
    loaded = next(obj.model.parameters()).dtype
    if torch.cuda.is_available() and loaded not in (torch.float16, torch.bfloat16):
        raise SystemExit(
            f"model loaded in {loaded}, which will not fit on 2x T4. "
            "Upgrade transformers (>=4.56) so the `dtype=` argument is honoured."
        )
    print(f"loaded {model_id or hf_id}  dtype={loaded}  "
          f"devices={sorted({str(p.device) for p in obj.model.parameters()})}")
    return obj


def run(model, args: argparse.Namespace) -> Path:
    """Generate every condition in ``args.suite``, resuming from the checkpoint."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)

    model_id = args.model_id or MODEL_HF_IDS[args.model]
    lo, hi = args.layers if args.layers else MODEL_LAYER_RANGES[args.model]
    layers = list(range(lo, hi))

    done_before = sum(len(v) for v in state["runs"].values())
    print(f"model    : {model_id}")
    print(f"layers   : {layers}")
    print(f"out      : {out}")
    print(f"resuming : {done_before} stories already in {state_path.name}\n")


    # ---- steering vectors (built once, then cached) ---------------------- #
    vec_path = out / f"steering_{args.model}.pt"
    if vec_path.is_file():
        vectors = SteeringVectorSet.load(vec_path)
        print(f"steering vectors: loaded from {vec_path.name}")
    else:
        print("steering vectors: extracting (a few minutes, once) ...")
        vectors = SteeringVectorExtractor(model).extract(
            load_pairs(), layers, pca_rank=args.pca_rank,
            only=args.constraints, verbose=False,
        )
        vectors.save(vec_path)
        print(f"steering vectors: saved to {vec_path.name}")
    for name in args.constraints:
        cons = [vectors.diagnostics[name][l]["consistency"] for l in layers]
        flag = "" if min(cons) > 0.3 else "   <-- weak, direction may be mostly noise"
        print(f"  {name:<16} agreement across pairs: {min(cons):.2f}-{max(cons):.2f}{flag}")

    # ---- noise/steering scale (calibrated once, then cached) ------------- #
    # Keyed by model AND layer range: activation scale differs by ~30x across
    # these models (Fanar 3.57, AceGPT 0.11), so sharing one value across models
    # in a single output folder would mis-scale everything.
    cal_key = f"{args.model}|{lo}-{hi}"
    if cal_key not in state["rms_scale"]:
        print("\ncalibrating activation scale ...")
        rms = RMSCalibrator(model).collect_block_rms(
            [{"role": "system", "content": prompts.SYS_ZERO_SHOT},
             {"role": "user", "content": prompts.PROMPT_ZERO_SHOT}],
            layers=layers,
        )
        state["rms_scale"][cal_key] = float(np.median(list(rms.values())))
        save_state(state_path, state)
    rms_scale = state["rms_scale"][cal_key]
    print(f"activation scale [{cal_key}] = {rms_scale:.4g}  "
          f"(noise size {args.alpha * rms_scale:.4g}, steering size {args.beta * rms_scale:.4g})")

    others = sorted({r.split("__")[0] for r in state["runs"]} - {args.model})
    if others:
        print(f"[note] this folder also holds runs for {others}. Their stories are kept "
              f"under their own run ids, but score_orthosteer.py scores every CSV here, "
              f"so use one folder per model if you want clean tables.")

    # ---- conditions ------------------------------------------------------ #
    suites = ["core", "ortho", "beta", "loo"] if "all" in args.suite else args.suite
    items = []
    for suite in suites:
        built, _ = build_suite(suite, vectors, layers, args.constraints, rms_scale, args)
        items.extend(built)

    normalised = []
    for it in items:
        d = {"mode": it} if isinstance(it, str) else dict(it)
        d.setdefault("temperature", args.temperature)
        d.setdefault("max_new_tokens_plan", args.max_new_tokens)
        d.setdefault("max_new_tokens_story", args.max_new_tokens)
        normalised.append(d)

    specs, run_ids, seen = [], [], set()
    for spec in make_specs(*normalised):
        rid = _spec_to_run_id(args.model, spec)
        if rid in seen:
            continue
        seen.add(rid)
        specs.append(spec)
        run_ids.append(rid)
        state["runs"].setdefault(rid, {})

    total = len(specs) * args.num_stories
    print(f"\n{len(specs)} conditions x {args.num_stories} stories = {total} generations")
    for rid in run_ids:
        print(f"  [{len(state['runs'][rid]):>3}/{args.num_stories}] {rid}")

    story_prompt = [
        {"role": "system", "content": prompts.SYS_ZERO_SHOT},
        {"role": "user", "content": prompts.PROMPT_ZERO_SHOT},
    ]

    # ---- generate, story-major so all conditions advance together -------- #
    done = sum(min(len(v), args.num_stories) for v in state["runs"].values())
    started, t0 = done, time.time()
    print()

    for x in range(args.num_stories):
        seed = seed_for_story(x)
        for spec, rid in zip(specs, run_ids):
            if str(x) in state["runs"][rid]:
                continue
            text = generate_one(
                model, spec, _spec_mode(spec), story_prompt, seed, args.max_new_tokens
            )
            state["runs"][rid][str(x)] = text
            save_state(state_path, state)

            done += 1
            elapsed = time.time() - t0
            rate = elapsed / max(done - started, 1)
            eta = (total - done) * rate / 60
            print(f"[{done:>4}/{total}] story {x:>3} | {rid[:58]:<58} "
                  f"| {rate:5.1f}s/story | ETA {eta:6.1f} min", flush=True)

        torch.cuda.empty_cache()

    write_csvs(out, state)
    print(f"\ndone. {done}/{total} stories -> {out}")
    print(f"score with:\n  python scripts/score_orthosteer.py --input-dir {out}")
    return out


def main() -> None:
    args = _parser().parse_args()
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model = load_model(args.model, args.model_id, dtype)
    run(model, args)


if __name__ == "__main__":
    main()
