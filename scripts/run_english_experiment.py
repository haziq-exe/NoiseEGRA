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
from noiseegra.activation_basis import collect_block_pcs  # noqa: E402
from noiseegra.constraint_metrics_en import EnglishConstraintChecker  # noqa: E402
from noiseegra.defaults import (  # noqa: E402
    EN_STEER_VECTORS,
    EN_TASK_CONSTRAINTS,
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
from score_english import (  # noqa: E402
    constraint_legend, live_table, score_condition,
)

from noiseegra.run_labels import SUBSPACE_MODES, plan_summary  # noqa: E402
from noiseegra.constraint_metrics_en import (  # noqa: E402
    DEFAULT_MAX_OPENING_WORDS, DEFAULT_MAX_SENTENCE_WORDS, DEFAULT_MAX_SYLLABLES,
    DEFAULT_MIN_NAME_USES, DEFAULT_SENTENCE_RANGE,
)
from noiseegra.entropy_gate import (  # noqa: E402
    GATE_LEVELS, collect_decode_entropies, describe as describe_gate, gate_threshold,
)

EN_PAIRS = Path(__file__).resolve().parents[1] / "noiseegra" / "data" / "steering_pairs_en.json"


def weights_are_cached(model_id: str) -> bool:
    """True if the full snapshot is already on local disk."""
    if Path(model_id).is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False

# closure ramps up so the pressure to wrap up grows as the story runs long;
# the others are properties of every sentence and stay flat.
EN_SCHEDULES = {
    "closure": "ramp",
    "present_tense": "constant",
    "simple_register": "constant",
    "dialogue": "constant",
}


def condition_label(spec) -> str:
    """Readable name for a condition, for the live table.

    Kept identical to what ``noiseegra.run_labels`` recovers from the run id, so
    the live table and every later scoring pass name the same thing the same way.
    """
    plan = getattr(spec, "steering_plan", None)
    if plan is None:
        return "baseline"
    if plan.offset_gamma > 0 and plan.offset_mode != "none":
        mode = SUBSPACE_MODES.get(plan.offset_mode, plan.offset_mode)
        return f"per-story offset g={plan.offset_gamma:g} ({mode})"
    if plan.noise_alpha > 0 and plan.noise_mode != "none":
        mode = SUBSPACE_MODES.get(plan.noise_mode, plan.noise_mode)
        return f"per-token noise a={plan.noise_alpha:g} ({mode})"
    return "steering only, no perturbation"


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
                    choices=["compare", "method", "noise", "offset", "core", "ortho",
                             "alpha", "gate", "beta", "loo", "all"])
    ap.add_argument("--task", default="generic", choices=["generic", "scenario"],
                    help="'generic' is the published design: one instruction with no "
                         "scenario, many requirements, and every story in one group, so "
                         "the measure is how many different stories the model invents. "
                         "'scenario' draws WritingPrompts scenarios, which supply the "
                         "content and cap how different the stories can be")
    ap.add_argument("--stories", type=int, default=100,
                    help="stories for the generic task, all in one group")
    ap.add_argument("--gate", default="none", choices=sorted(GATE_LEVELS),
                    help="entropy gate for every perturbed condition outside --suite "
                         "gate: 'none' perturbs every decode step, 'median' the more "
                         "uncertain half, 'high' the most uncertain tenth")
    ap.add_argument("--gate-samples", type=int, default=6,
                    help="unsteered generations used to measure the model's own entropy "
                         "distribution before setting the gate threshold")
    ap.add_argument("--allow-task-change", action="store_true",
                    help="proceed even though the checkpoint was written under different "
                         "constraint thresholds or a different prompt selection")
    ap.add_argument("--no-diversity", dest="diversity", action="store_false",
                    help="skip Vendi/Self-BLEU while scoring conditions as they finish "
                         "(avoids downloading the embedding model)")
    ap.add_argument("--with-baseline", action="store_true",
                    help="prepend an unsteered baseline condition to whichever suite is run "
                         "(already included in `compare` and `noise`)")
    ap.add_argument("--constraints", nargs="*", default=list(EN_TASK_CONSTRAINTS),
                    help="what the prompt asks for and the scorer checks")
    ap.add_argument("--steer-vectors", nargs="*", default=list(EN_STEER_VECTORS),
                    help="which of those steering is applied along; the rest are asked "
                         "for in the prompt only")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--stories-per-prompt", type=int, default=5)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--prompt-split", default="test")
    ap.add_argument("--max-words", type=int, default=EN_MAX_WORDS)
    ap.add_argument("--max-grade", type=float, default=EN_MAX_GRADE_LEVEL)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--gamma-sweep", nargs="*", type=float, default=[0.05, 0.15, 0.4],
                    help="per-story offset magnitudes used by --suite offset")
    ap.add_argument("--offset-rank", type=int, default=64,
                    help="how many activation principal components offsets may use")
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
    ap.add_argument("--embedding-model", default=None,
                    help="diversity embedding model: registry key or HF id "
                         "(default qwen3-0.6b; use bge-m3 for the published Arabic setup)")
    ap.add_argument("--truncate-words", type=int, default=None,
                    help="cut every story to its first N words before scoring diversity")
    ap.add_argument("--embedding-device", default="auto",
                    help="where to put the diversity embedding model: 'auto' picks a "
                         "GPU with room and falls back to the CPU, which is what you "
                         "want while an 8B model is holding the card")
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

    # The constraint thresholds and the prompt selection are baked into the text
    # the model is given, but not into the run id. Reusing a checkpoint under
    # different values would silently mix stories written to different
    # instructions, so the task setup is pinned on first write.
    task = {
        "task": args.task,
        "constraints": list(args.constraints),
        "steer_vectors": list(args.steer_vectors),
        "max_words": args.max_words,
        "max_grade": args.max_grade,
        "num_prompts": args.num_prompts if args.task == "scenario" else 1,
        "stories": args.stories if args.task == "generic" else args.stories_per_prompt,
        "prompt_seed": args.prompt_seed,
        "prompt_split": args.prompt_split,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    prev = state.get("task")
    if prev is None:
        state["task"] = task
        save_state(state_path, state)
    elif prev != task and not args.allow_task_change:
        changed = [f"    {k}: {prev.get(k)!r} -> {task[k]!r}"
                   for k in task if prev.get(k) != task[k]]
        raise SystemExit(
            f"{state_path} holds stories generated under a different task setup:\n"
            + "\n".join(changed)
            + "\n\n  These values go into the prompt itself, so old and new stories are not\n"
              "  comparable. Either point --out at a fresh directory, or pass\n"
              "  --allow-task-change if you are certain you want them mixed."
        )

    print(f"model       : {model_id}")
    print(f"layers      : {layers}")
    print(f"constraints : {args.constraints}")
    print(f"out         : {out}")
    print(f"resuming    : {sum(len(v) for v in state['runs'].values())} stories already saved\n")

    # ---- the task -------------------------------------------------------- #
    # The checker is built first and the prompt is generated from its
    # `requirements()`, so the sentence the model is given and the rule it is
    # scored against are the same object. A threshold cannot change in one place
    # and not the other.
    checker = EnglishConstraintChecker(
        max_words=args.max_words, max_grade_level=args.max_grade,
        constraints=list(args.constraints),
    )

    if args.task == "generic":
        prompts = ["<generic instruction>"]
        messages = [wp.build_generic_messages(checker.requirements(), args.constraints)]
        stories_per_prompt = args.stories
        print(f"task: one generic instruction, {len(args.constraints)} requirements, "
              f"{stories_per_prompt} stories in a single group")
        print("\n" + messages[0][1]["content"] + "\n")
    else:
        prompts = state.get("prompts")
        if not prompts or len(prompts) != args.num_prompts:
            prompts = wp.load_prompts(
                args.num_prompts, seed=args.prompt_seed, split=args.prompt_split,
                cache=out / "prompts.json",
            )
            state["prompts"] = prompts
            save_state(state_path, state)
        messages = [
            wp.build_messages(t, args.constraints,
                              requirements=checker.requirements())
            for t in prompts
        ]
        stories_per_prompt = args.stories_per_prompt
        print(f"task: {len(prompts)} WritingPrompts scenarios, e.g.:")
        for t in prompts[:3]:
            print(f"   - {t[:110]}{'...' if len(t) > 110 else ''}")

    # Loading is deferred: with the steering vectors and the activation scale both
    # cached, a fully generated model needs no weights at all and exits in seconds.
    holder = {"model": None, "dtype": None}

    def get_model():
        if holder["model"] is None:
            if weights_are_cached(model_id):
                print("\nloading model from local cache ...", flush=True)
            else:
                # Kaggle wipes ~/.cache between sessions even when /kaggle/working is
                # restored, so a resumed run re-downloads the weights. Say so, because
                # otherwise this looks like a hang.
                print(
                    f"\n{model_id} is not in the local cache. Downloading the weights "
                    "(~14-24 GB) before anything can run.\n"
                    "  This is a download, not a hang. Progress appears below; if it "
                    "falls under ~1 MB/s, interrupt and re-run -- it restarts at full "
                    "speed and no generated stories are lost.",
                    flush=True,
                )
            print("", flush=True)
            m = build_model(model_id, dtype=dtype_arg, wrapper_for=hf_id)
            d = next(m.model.parameters()).dtype
            if torch.cuda.is_available() and d not in (torch.float16, torch.bfloat16):
                raise SystemExit(f"model loaded in {d}; upgrade transformers (>=4.56).")
            print(f"  dtype={d}")
            holder["model"], holder["dtype"] = m, d
        return holder["model"]

    # ---- steering vectors ------------------------------------------------- #
    vec_path = out / f"steering_{args.model}.pt"
    if vec_path.is_file():
        vectors = SteeringVectorSet.load(vec_path)
        print(f"steering vectors: loaded from {vec_path.name}")
    else:
        print("steering vectors: extracting (once) ...")
        vectors = SteeringVectorExtractor(get_model()).extract(
            load_pairs(EN_PAIRS), layers,
            system=wp.SYSTEM_PROMPT, user=wp.EXTRACTION_PROMPT,
            pca_rank=args.pca_rank, only=args.steer_vectors, verbose=False,
        )
        vectors.save(vec_path)
        print(f"steering vectors: saved to {vec_path.name}")
    for name in args.steer_vectors:
        cons = [vectors.diagnostics[name][l]["consistency"] for l in layers]
        flag = "" if min(cons) > 0.3 else "   <-- weak, direction may be mostly noise"
        print(f"  {name:<16} agreement across pairs: {min(cons):.2f}-{max(cons):.2f}{flag}")

    # ---- activation scale, keyed by model+layers -------------------------- #
    cal_key = f"{args.model}|{lo}-{hi}"
    if cal_key not in state["rms_scale"]:
        print("\ncalibrating activation scale ...")
        rms = RMSCalibrator(get_model()).collect_block_rms(messages[0], layers=layers)
        state["rms_scale"][cal_key] = float(np.median(list(rms.values())))
        save_state(state_path, state)
    rms_scale = state["rms_scale"][cal_key]
    if not (rms_scale > 0) or rms_scale != rms_scale:
        raise SystemExit(
            f"calibration returned {rms_scale}; the forward pass produced non-finite "
            "activations. This model probably cannot run in float16 -- retry with "
            "--dtype bfloat16 (slower on a T4, but correct)."
        )
    print(f"activation scale [{cal_key}] = {rms_scale:.4g}  "
          f"(noise {args.alpha * rms_scale:.4g}, steering {args.beta * rms_scale:.4g})")
    if holder["dtype"] == torch.float16 and rms_scale > 100:
        print(f"[warn] activation scale {rms_scale:.4g} is large for float16 "
              "(max representable 65504). If the stories come out empty or garbled, "
              "rerun with --dtype bfloat16.")

    # ---- directions a per-story offset is allowed to use -------------------- #
    args.offset_basis = None
    suites_req = (["core", "ortho", "alpha", "gate", "beta", "loo"]
                  if "all" in args.suite else args.suite)
    if "offset" in suites_req:
        pc_path = out / f"actpcs_{args.model}.pt"
        if pc_path.is_file():
            args.offset_basis = torch.load(pc_path, map_location="cpu", weights_only=False)
            print(f"activation basis: loaded from {pc_path.name}")
        else:
            print("activation basis: estimating principal components (once) ...")
            args.offset_basis = collect_block_pcs(
                get_model(),
                messages[:4],
                layers, rank=args.offset_rank,
            )
            torch.save(args.offset_basis, pc_path)
            print(f"activation basis: saved to {pc_path.name}")

    # ---- entropy gate threshold, measured once per model ------------------- #
    # In nats, and nats are not comparable across models or tokenisers, so the
    # threshold is a quantile of this model's own decode entropy -- the same
    # reasoning as calibrating the noise scale to the model's own block RMS.
    # Measured unsteered, so it describes the model and not the condition.
    args.gate_thresholds = {"none": 0.0}
    if "gate" in suites_req or args.gate != "none":
        gate_key = f"{args.model}|{lo}-{hi}"
        store = state.setdefault("entropy", {})
        if gate_key not in store:
            print("\nmeasuring the model's own decode entropy ...", flush=True)
            ent = collect_decode_entropies(
                get_model(), messages[0], n_samples=args.gate_samples,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            store[gate_key] = {
                "n_steps": len(ent),
                "quantiles": {lv: gate_threshold(ent, lv) for lv in GATE_LEVELS},
            }
            save_state(state_path, state)
        rec = store[gate_key]
        args.gate_thresholds.update(rec["quantiles"])
        print(f"decode entropy over {rec['n_steps']} unsteered steps: "
              + ", ".join(f"{lv} gate at {v:.3f} nats" for lv, v in
                          sorted(rec["quantiles"].items()) if v))

    # ---- conditions -------------------------------------------------------- #
    suites = suites_req
    items = ["baseline"] if args.with_baseline else []
    for suite in suites:
        built, desc = build_suite(suite, vectors, layers, args.steer_vectors, rms_scale, args)
        items.extend(built)
        print(f"  suite {suite}: {desc} ({len(built)} runs)")

    # A gate asked for on the command line applies to every condition that
    # actually perturbs. --suite gate sets its own per-arm gates and is left alone.
    if args.gate != "none":
        thr = args.gate_thresholds.get(args.gate, 0.0)
        for it in items:
            plan = it.get("plan") if isinstance(it, dict) else None
            if plan is not None and (plan.noise_alpha > 0 or plan.offset_gamma > 0):
                plan.gate_threshold = thr
                plan.gate_level = args.gate
        print(f"  {describe_gate(args.gate, thr)}")

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

    P, K = len(prompts), stories_per_prompt
    total = len(specs) * P * K
    print(f"\n{len(specs)} conditions x {P} prompts x {K} stories = {total} generations")
    for rid in run_ids:
        print(f"  [{len(state['runs'][rid]):>4}/{P * K}] {rid}")

    remaining = sum(
        1 for rid in run_ids for k in range(K) for p_idx in range(P)
        if f"{p_idx}:{k}" not in state["runs"][rid]
    )
    print(f"\n{remaining} of {total} still to generate.")

    # Conditions run one at a time and are scored the moment they finish, so the
    # sweep produces readable results as it goes instead of only at the end.
    scorer = None
    if args.diversity:
        from noiseegra.creativity_metrics import CreativityScorer
        print("loading the embedding model for diversity scoring ...", flush=True)
        scorer = CreativityScorer(
            ["placeholder one", "placeholder two"],
            embedding_model=args.embedding_model,
            truncate_words=args.truncate_words,
            device=args.embedding_device,
        )
        print(f"  embedding model on {scorer.model.device}", flush=True)

    done, t0 = total - remaining, time.time()
    started = done
    rows = []

    live_header, live_row = live_table(checker.constraints, args.diversity)
    print("\n" + live_header)
    print("-" * len(live_header), flush=True)

    for spec, rid in zip(specs, run_ids):
        missing = [(p_idx, k) for k in range(K) for p_idx in range(P)
                   if f"{p_idx}:{k}" not in state["runs"][rid]]
        if missing:
            model = get_model()
            mode = _spec_mode(spec)
            for p_idx, k in missing:
                text = generate_one(model, spec, mode, messages[p_idx],
                                    seed_for(p_idx, k), args.max_new_tokens)
                state["runs"][rid][f"{p_idx}:{k}"] = text
                save_state(state_path, state)
                done += 1
                rate = (time.time() - t0) / max(done - started, 1)
                print(f"  [{done:>4}/{total}] {rid[-34:]:<34} prompt {p_idx:>2} story {k:>2}"
                      f" | {rate:5.1f}s | ETA {(total - done) * rate / 60:6.1f} min", flush=True)
            torch.cuda.empty_cache()

        cells = state["runs"][rid]
        keys = sorted(cells, key=lambda x: tuple(int(i) for i in x.split(":")))
        stories = [cells[key] for key in keys]
        pidx = [int(key.split(":")[0]) for key in keys]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            row = score_condition(stories, pidx, checker, scorer)
        except Exception as exc:
            # Every story is already on disk. Losing the scores for one condition
            # is an inconvenience; losing the run at story 300 of 400 is not.
            print(f"  [warn] scoring {rid} failed ({type(exc).__name__}: {exc}); "
                  "the stories are saved, score them later with "
                  "scripts/score_english.py", flush=True)
            row = score_condition(stories, pidx, checker, None)
        row["run"] = rid
        rows.append(row)
        write_csvs(out, state)
        print(live_row(condition_label(spec), row), flush=True)

    print("\n" + "=" * len(live_header))
    print(f"  {args.model}: {P} prompt{'s' if P != 1 else ''} x {K} stories per condition")
    for line in plan_summary([r["run"] for r in rows]):
        print(f"  {line}")
    print("  perturbation magnitudes are multiples of the model's own activation scale")
    print("=" * len(live_header))
    print(live_header)
    print("-" * len(live_header))
    for spec, row in zip(specs, rows):
        print(live_row(condition_label(spec), row))
    print("\nwhat each requirement column means, and what a story has to do to pass it")
    for line in constraint_legend(checker):
        print("  " + line)

    summary = out / "live_scores.csv"
    with summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "label"] + [k for k in rows[0] if k != "run"])
        w.writeheader()
        for spec, row in zip(specs, rows):
            w.writerow({**row, "label": condition_label(spec)})
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
