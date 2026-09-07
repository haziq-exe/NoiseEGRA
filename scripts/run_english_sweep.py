#!/usr/bin/env python
"""Run the English experiment across several models, one after another.

    python scripts/run_english_sweep.py --suite compare \
        --num-prompts 10 --stories-per-prompt 10

Each model runs in its own subprocess, so its GPU memory is fully released before
the next one loads -- loading several 8B models into one process will not fit on
two T4s. A model that fails (download error, gated repo, out of memory) is
reported and the sweep continues.

Every model keeps its own checkpoint under <out>/<model>/, so re-running the same
command resumes each one exactly where it stopped. Nothing is regenerated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noiseegra.defaults import EN_MODEL_DEPTHS, EN_MODEL_HF_IDS  # noqa: E402

DEFAULT_MODELS = ["Qwen3-8B", "Qwen2.5-7B", "OLMo-2-7B", "Falcon3-7B", "Granite-3.1-8B"]

# Args forwarded verbatim to run_english_experiment.py.
PASSTHROUGH = [
    ("--suite", "suite", True), ("--constraints", "constraints", True),
    ("--num-prompts", "num_prompts", False), ("--stories-per-prompt", "stories_per_prompt", False),
    ("--prompt-seed", "prompt_seed", False), ("--max-words", "max_words", False),
    ("--max-grade", "max_grade", False), ("--alpha", "alpha", False), ("--beta", "beta", False),
    ("--protect-rank", "protect_rank", False), ("--max-new-tokens", "max_new_tokens", False),
    ("--temperature", "temperature", False), ("--out", "out", False),
    ("--alpha-sweep", "alpha_sweep", True), ("--gamma-sweep", "gamma_sweep", True),
    ("--offset-rank", "offset_rank", False),
]


def progress(out: Path, model: str) -> tuple[int, int]:
    """(stories saved, conditions) from a model's checkpoint."""
    sp = out / model / "state.json"
    if not sp.is_file():
        return 0, 0
    try:
        runs = json.loads(sp.read_text(encoding="utf-8")).get("runs", {})
    except Exception:
        return 0, 0
    return sum(len(v) for v in runs.values()), len(runs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(EN_MODEL_HF_IDS))
    ap.add_argument("--suite", nargs="+", default=["compare"])
    ap.add_argument("--constraints", nargs="*", default=None)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--stories-per-prompt", type=int, default=10)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--max-grade", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--protect-rank", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--alpha-sweep", nargs="*", type=float, default=None)
    ap.add_argument("--gamma-sweep", nargs="*", type=float, default=None)
    ap.add_argument("--offset-rank", type=int, default=None)
    ap.add_argument("--no-diversity", dest="diversity", action="store_false")
    ap.add_argument("--out", default="/kaggle/working/english")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop launching new models once this much wall time has passed; "
                         "the model already running is left to finish")
    args = ap.parse_args()

    out = Path(args.out)
    runner = str(ROOT / "scripts" / "run_english_experiment.py")
    per_model = args.num_prompts * args.stories_per_prompt

    print(f"{len(args.models)} models x {args.num_prompts} prompts x "
          f"{args.stories_per_prompt} stories = {per_model} stories per condition\n")
    for m in args.models:
        done, conds = progress(out, m)
        blocks = EN_MODEL_DEPTHS[m]
        print(f"  {m:<18} {blocks:>2} blocks  {EN_MODEL_HF_IDS[m]:<42} "
              f"{'(resuming: %d saved)' % done if done else ''}")
    print()

    results, t_start = {}, time.time()

    for i, model in enumerate(args.models, 1):
        elapsed_h = (time.time() - t_start) / 3600
        if args.max_hours is not None and elapsed_h >= args.max_hours:
            print(f"\n[budget] {elapsed_h:.1f}h elapsed, not starting {model}. "
                  f"Re-run this command to continue.")
            results[model] = "not started (time budget)"
            continue

        cmd = [sys.executable, "-u", runner, "--model", model]
        if not args.diversity:
            cmd.append("--no-diversity")
        for flag, attr, is_list in PASSTHROUGH:
            val = getattr(args, attr)
            if val is None:
                continue
            cmd.append(flag)
            cmd += [str(v) for v in val] if is_list else [str(val)]

        print("=" * 92)
        print(f"[{i}/{len(args.models)}] {model}   ({elapsed_h:.1f}h elapsed)")
        print("=" * 92, flush=True)

        rc = subprocess.call(cmd, cwd=str(ROOT))
        done, conds = progress(out, model)
        expected = conds * per_model
        if rc != 0:
            results[model] = f"FAILED (exit {rc}) — {done} stories saved"
        elif conds == 0:
            results[model] = "no checkpoint written — nothing generated"
        elif done < expected:
            results[model] = f"incomplete — {done}/{expected} stories"
        else:
            results[model] = f"complete — {done} stories across {conds} conditions"
        print(f"\n[{model}] {results[model]}", flush=True)

    print("\n" + "=" * 92)
    print(f"SWEEP SUMMARY  ({(time.time() - t_start) / 3600:.1f}h total)")
    print("=" * 92)
    for m in args.models:
        print(f"  {m:<18} {results.get(m, 'not run')}")
    print("\nScore each with:")
    for m in args.models:
        print(f"  python scripts/score_english.py --input-dir {out / m} --diversity")


if __name__ == "__main__":
    main()
