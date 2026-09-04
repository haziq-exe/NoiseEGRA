#!/usr/bin/env python
"""Orthogonal constraint steering on English WritingPrompts. Resumable.

    python scripts/run_english_experiment.py --model Qwen3-8B --suite compare \
        --num-prompts 10 --stories-per-prompt 5

Generates K stories for each of P prompts under every condition. Diversity has to
be measured within a prompt -- stories from different prompts are trivially
different -- so the scorer groups by prompt before averaging.

Every story is written into state.json as it is produced; re-running the same
command resumes and regenerates nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from noiseegra import writingprompts as wp  # noqa: E402
from noiseegra.defaults import (  # noqa: E402
    EN_CONSTRAINTS,
    EN_MAX_GRADE_LEVEL,
    EN_MAX_WORDS,
    EN_MODEL_HF_IDS,
    EN_MODEL_LAYER_RANGES,
    RMS_ALPHA,
)
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402
from noiseegra.setup_experiment import _spec_mode, _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import (  # noqa: E402
    SteeringVectorExtractor,
    SteeringVectorSet,
    load_pairs,
)

from build_steering_vectors import build_model  # noqa: E402
from kaggle_orthosteer import generate_one, load_state, save_state  # noqa: E402
from run_orthosteer_experiment import build_suite  # noqa: E402

EN_PAIRS = Path(__file__).resolve().parents[1] / "noiseegra" / "data" / "steering_pairs_en.json"

# closure ramps up so the pressure to wrap up grows as the story runs long;
# the others are properties of every sentence and stay flat.
EN_SCHEDULES = {
    "closure": "ramp",
    "present_tense": "constant",
    "simple_register": "constant",
    "dialogue": "constant",
}


def seed_for(prompt_idx: int, story_idx: int) -> int:
    """Stable per (prompt, story) seed, shared across conditions."""
    return (42 + prompt_idx * 100003 + story_idx * 7919) % (2 ** 31)


def write_csvs(out: Path, state: dict) -> None:
    for run_id, cells in state["runs"].items():
        rows = []
        for key, text in cells.items():
            p_idx, s_idx = (int(x) for x in key.split(":"))
            rows.append((p_idx, s_idx, text))
        rows.sort()
        with (out / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_index", "story_index", "story"])
            w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen3-8B", choices=sorted(EN_MODEL_HF_IDS))
    ap.add_argument("--model-id", help="HF id or local snapshot dir; overrides --model's id")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--suite", nargs="+", default=["compare"],
                    choices=["compare", "method", "noise", "core", "ortho", "alpha", "beta", "loo", "all"])
    ap.add_argument("--constraints", nargs="*", default=list(EN_CONSTRAINTS))
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--stories-per-prompt", type=int, default=5)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--prompt-split", default="test")
    ap.add_argument("--max-words", type=int, default=EN_MAX_WORDS)
    ap.add_argument("--max-grade", type=float, default=EN_MAX_GRADE_LEVEL)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--alpha-sweep", nargs="*", type=float,
                    default=[0.0, 0.0875, 0.175, 0.35, 0.7],
                    help="noise strengths used by --suite alpha")
    ap.add_argument("--alpha", type=float, default=RMS_ALPHA)
    ap.add_argument("--protect-rank", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--steer-prefill", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pca-rank", type=int, default=8)
    ap.add_argument("--out", default="/kaggle/working/english")
    args = ap.parse_args()

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)

    hf_id = EN_MODEL_HF_IDS[args.model]
    model_id = args.model_id or hf_id
    lo, hi = args.layers if args.layers else EN_MODEL_LAYER_RANGES[args.model]
    layers = list(range(lo, hi))
    dtype_arg = None if args.dtype == "auto" else getattr(torch, args.dtype)

    print(f"model       : {model_id}")
    print(f"layers      : {layers}")
    print(f"constraints : {args.constraints}")
    print(f"out         : {out}")
    print(f"resuming    : {sum(len(v) for v in state['runs'].values())} stories already saved\n")

    # ---- prompts (fixed and cached, so a resume reuses exactly the same set) --
    prompts = state.get("prompts")
    if not prompts or len(prompts) != args.num_prompts:
        prompts = wp.load_prompts(
            args.num_prompts, seed=args.prompt_seed, split=args.prompt_split,
            cache=out / "prompts.json",
        )
        state["prompts"] = prompts
        save_state(state_path, state)
    print(f"{len(prompts)} WritingPrompts prompts, e.g.:")
    for t in prompts[:3]:
        print(f"   - {t[:110]}{'...' if len(t) > 110 else ''}")

    print("\nloading model ...")
    model = build_model(model_id, dtype=dtype_arg, wrapper_for=hf_id)
    dtype = next(model.model.parameters()).dtype
    if torch.cuda.is_available() and dtype not in (torch.float16, torch.bfloat16):
        raise SystemExit(f"model loaded in {dtype}; upgrade transformers (>=4.56).")
    print(f"  dtype={dtype}")

    # ---- steering vectors ------------------------------------------------- #
    vec_path = out / f"steering_{args.model}.pt"
    if vec_path.is_file():
        vectors = SteeringVectorSet.load(vec_path)
        print(f"steering vectors: loaded from {vec_path.name}")
    else:
        print("steering vectors: extracting (once) ...")
        vectors = SteeringVectorExtractor(model).extract(
            load_pairs(EN_PAIRS), layers,
            system=wp.SYSTEM_PROMPT, user=wp.EXTRACTION_PROMPT,
            pca_rank=args.pca_rank, only=args.constraints, verbose=False,
        )
        vectors.save(vec_path)
        print(f"steering vectors: saved to {vec_path.name}")
    for name in args.constraints:
        cons = [vectors.diagnostics[name][l]["consistency"] for l in layers]
        flag = "" if min(cons) > 0.3 else "   <-- weak, direction may be mostly noise"
        print(f"  {name:<16} agreement across pairs: {min(cons):.2f}-{max(cons):.2f}{flag}")

    # ---- activation scale, keyed by model+layers -------------------------- #
    cal_key = f"{args.model}|{lo}-{hi}"
    if cal_key not in state["rms_scale"]:
        print("\ncalibrating activation scale ...")
        rms = RMSCalibrator(model).collect_block_rms(
            wp.build_messages(prompts[0], args.constraints,
                              max_words=args.max_words, max_grade=args.max_grade),
            layers=layers,
        )
        state["rms_scale"][cal_key] = float(np.median(list(rms.values())))
        save_state(state_path, state)
    rms_scale = state["rms_scale"][cal_key]
    print(f"activation scale [{cal_key}] = {rms_scale:.4g}  "
          f"(noise {args.alpha * rms_scale:.4g}, steering {args.beta * rms_scale:.4g})")

    # ---- conditions -------------------------------------------------------- #
    suites = ["core", "ortho", "alpha", "beta", "loo"] if "all" in args.suite else args.suite
    items = []
    for suite in suites:
        built, desc = build_suite(suite, vectors, layers, args.constraints, rms_scale, args)
        items.extend(built)
        print(f"  suite {suite}: {desc} ({len(built)} runs)")

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

    P, K = len(prompts), args.stories_per_prompt
    total = len(specs) * P * K
    print(f"\n{len(specs)} conditions x {P} prompts x {K} stories = {total} generations")
    for rid in run_ids:
        print(f"  [{len(state['runs'][rid]):>4}/{P * K}] {rid}")

    messages = [
        wp.build_messages(t, args.constraints, max_words=args.max_words, max_grade=args.max_grade)
        for t in prompts
    ]

    done = sum(len(v) for v in state["runs"].values())
    started, t0 = done, time.time()
    print()

    for k in range(K):
        for p_idx in range(P):
            key = f"{p_idx}:{k}"
            seed = seed_for(p_idx, k)
            for spec, rid in zip(specs, run_ids):
                if key in state["runs"][rid]:
                    continue
                text = generate_one(
                    model, spec, _spec_mode(spec), messages[p_idx], seed, args.max_new_tokens
                )
                state["runs"][rid][key] = text
                save_state(state_path, state)

                done += 1
                rate = (time.time() - t0) / max(done - started, 1)
                print(f"[{done:>4}/{total}] prompt {p_idx:>2} story {k:>2} | {rid[:52]:<52} "
                      f"| {rate:5.1f}s | ETA {(total - done) * rate / 60:6.1f} min", flush=True)
        torch.cuda.empty_cache()

    write_csvs(out, state)
    print(f"\ndone. {done}/{total} -> {out}")
    print(f"score with:\n  python scripts/score_english.py --input-dir {out}")


if __name__ == "__main__":
    main()
