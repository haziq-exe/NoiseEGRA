#!/usr/bin/env python
"""The headline method on tasks beyond stories (noiseegra/domains.py).

    python scripts/run_domains.py --domains number tests plan poem --stories 50 --out OUT
    python scripts/run_domains.py ... --dry-run          # no model: builds every plan

For each domain, three arms at the same seeds:

- the model as it ships, temperature 1.0;
- top-p 0.95 at temperature 1.8;
- the headline method: the domain's rules steered at a total strength of 2.5 from
  its own contrast pairs, and the per-story random noise sized while it is
  written -- built by suite 'headline' of run_orthosteer_experiment, the same code
  the story runs used, so the method is the one measured there and not a
  re-implementation of it.

The steering directions are extracted per domain, in that domain's own request,
with ``on_task`` shielded from the noise the way ``in_story`` is for stories. Run
ids are prefixed ``dom-<domain>__`` so every domain shares one checkpoint without
collisions, in the story runner's format, so run_sharded merges it and
score_novelty judges it. With ``--shard i/n`` a process takes every n-th domain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from noiseegra import domains as D  # noqa: E402
from noiseegra.defaults import EN_MODEL_HF_IDS  # noqa: E402
from noiseegra.setup_experiment import _spec_mode, _spec_to_run_id, make_specs  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorExtractor, SteeringVectorSet  # noqa: E402
import run_orthosteer_experiment as _ro  # noqa: E402

ARMS = ("untouched", "topp", "method")


def seed_for(prompt_idx: int, story_idx: int, offset: int = 0) -> int:
    """The story runner's seed for (prompt, story): the arms share every seed."""
    base = (42 + prompt_idx * 100003 + story_idx * 7919) % (2 ** 31)
    if offset:
        base = (base + int(offset) * 1000003) % (2 ** 31)
    return base


def suite_args(args) -> types.SimpleNamespace:
    """The settings suite 'headline' reads, at the story runner's defaults, with
    only the noise sized while writing asked for."""
    from noiseegra.defaults import RMS_ALPHA
    return types.SimpleNamespace(
        protect_rank=8, horizon=200, steer_prefill=False, direction_source="extracted",
        alpha=RMS_ALPHA, beta=1.0, noise_horizon=24, baseline_temperature=args.baseline_temperature,
        baseline_top_p=0.95, baseline_top_k=40, offset_basis_kind="prompt", keep_directions={},
        offset_basis=None, steer_budget=args.steer_budget, tail_sweep=[8],
        noise_beta_sweep=[2.0], offset_random_rank=64, front_tokens=40,
        fixed_lengths=[14.83, 18.0], online_start=0.10, headline_arms=["while"],
        headline_budgets=None, output_tilts=None, fixed_base=None)


def set_run_defaults() -> None:
    """What the story runner writes into RUN_DEFAULTS from its own flag defaults."""
    _ro.RUN_DEFAULTS.update({"offset_gamma_spread": 0.0, "offset_taper": 1.0,
                             "noise_fmin_cycles": 0.25, "offset_envelope": "flat"})


def arms_for(dom, vectors, layers, rms_scale, args):
    """{arm: spec item} for one domain."""
    items, _ = _ro.build_suite("headline", vectors, layers, dom.steered, rms_scale,
                               suite_args(args))
    assert len(items) == 1, "suite 'headline' should build the one method arm"
    return {"untouched": "baseline",
            "topp": {"mode": "baseline", "temperature": args.baseline_temperature, "top_p": 0.95},
            "method": items[0]}


def specs_for(dom, vectors, layers, rms_scale, args):
    """[(arm, run id, spec)] in arm order."""
    out = []
    for arm, item in arms_for(dom, vectors, layers, rms_scale, args).items():
        spec = make_specs({"mode": item} if isinstance(item, str) else dict(item))[0]
        out.append((arm, f"dom-{dom.name}__" + _spec_to_run_id(args.model, spec), spec))
    return out


def stand_in_vectors(dom, layers) -> SteeringVectorSet:
    torch.manual_seed(0)
    names = dom.steered + ["on_task"]
    v = SteeringVectorSet(
        vectors={n: {l: torch.randn(64) for l in layers} for n in names},
        components={n: {l: torch.linalg.qr(torch.randn(64, 8))[0] for l in layers} for n in names},
        positives={n: {l: torch.randn(64) for l in layers} for n in names},
        output_profiles={n: torch.randn(32) for n in names})
    v.shield, v.shield_rank = ["on_task"], 0
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=list(D.DOMAINS), choices=list(D.DOMAINS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--stories", type=int, default=50)
    ap.add_argument("--model", default="Qwen3-1.7B", choices=sorted(EN_MODEL_HF_IDS))
    ap.add_argument("--layers", type=int, nargs=2, default=[6, 14])
    ap.add_argument("--steer-budget", type=float, default=2.5)
    ap.add_argument("--baseline-temperature", type=float, default=1.8)
    ap.add_argument("--peek", type=int, default=2)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="experiments/domains")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    layers = list(range(*args.layers))
    set_run_defaults()
    i, n = (int(x) for x in args.shard.split("/"))
    mine = args.domains[i::n]

    if args.dry_run:
        print(f"domains {args.domains}; this shard takes {mine}; arms {args.arms}; "
              f"{args.stories} each")
        total = 0
        for name in args.domains:
            dom = D.get(name)
            specs = [s for s in specs_for(dom, stand_in_vectors(dom, layers), layers, 1.0, args)
                     if s[0] in args.arms]
            ids = {rid for _, rid, _ in specs}
            assert len(ids) == len(specs), f"{name}: two arms share a run id"
            total += len(specs) * args.stories
            print(f"  {name}: steers {dom.steered}; checks {[r.name for r in dom.rules]}")
            for arm, rid, _ in specs:
                print(f"    {arm:9s} {rid[:40]}...{rid[-80:]}")
        print(f"{total} generations")
        return

    from build_steering_vectors import build_model
    from kaggle_orthosteer import generate_one, load_state, save_state
    from noiseegra.RMS_std import RMSCalibrator

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = load_state(state_path)
    state.setdefault("runs", {})
    state.setdefault("rms_scale", {})
    state.setdefault("task", {"task": "domains", "domains": args.domains,
                              "stories": args.stories, "model": args.model})

    model = build_model(EN_MODEL_HF_IDS[args.model], dtype=None,
                        wrapper_for=EN_MODEL_HF_IDS[args.model])
    model.enable_thinking = False
    print("  reasoning blocks: off", flush=True)

    for name in mine:
        dom = D.get(name)
        msgs = dom.messages()
        print(f"\n=== {name} ===\n{msgs[1]['content']}\n", flush=True)
        vec_path = out / f"steering_dom-{name}_{args.model}.pt"
        if vec_path.is_file():
            vectors = SteeringVectorSet.load(vec_path)
        else:
            vectors = SteeringVectorExtractor(model).extract(
                dom.steering_pairs(), layers, system=msgs[0]["content"], user=msgs[1]["content"],
                window="all", window_tokens=4, pca_rank=8,
                only=dom.steered + ["on_task"], verbose=False)
            vectors.save(vec_path)
        vectors.shield, vectors.shield_rank = ["on_task"], 0
        for rule in dom.steered + ["on_task"]:
            cons = [d["consistency"] for d in vectors.diagnostics[rule].values()]
            print(f"  {rule:16s} agreement across pairs: {min(cons):.2f}-{max(cons):.2f}"
                  + ("  (shielded, not pushed)" if rule == "on_task" else ""), flush=True)
        key = f"{name}|{args.model}|{layers[0]}-{layers[-1] + 1}"
        if key not in state["rms_scale"]:
            rms = RMSCalibrator(model).collect_block_rms(msgs, layers=layers)
            state["rms_scale"][key] = float(np.median(list(rms.values())))
            save_state(state_path, state)
        rms_scale = state["rms_scale"][key]
        print(f"  activation scale [{key}] = {rms_scale:.4g}", flush=True)

        for arm, rid, spec in specs_for(dom, vectors, layers, rms_scale, args):
            if arm not in args.arms:
                continue
            cells = state["runs"].setdefault(rid, {})
            todo = [k for k in range(args.stories) if f"0:{k}" not in cells]
            print(f"  [{name}/{arm}] {len(cells)} saved, {len(todo)} to write  {rid[-70:]}",
                  flush=True)
            for k in todo:
                t0 = time.time()
                text = generate_one(model, spec, _spec_mode(spec), msgs, seed_for(0, k),
                                    dom.max_new_tokens, max_words=dom.max_words, story_index=k)
                cells[f"0:{k}"] = text
                save_state(state_path, state)
                ok, why = dom.valid(text)
                if k < args.peek:
                    c = dom.correct(text) if dom.correct else None
                    print(f"  [peek] {name}/{arm} #{k}: {' '.join(text.split())[:200]} | "
                          f"{'valid' if ok else why}, {dom.violations(text)} rules broken"
                          + ("" if c is None else f", correct {c}"), flush=True)
                if (k + 1) % 25 == 0:
                    print(f"  [{name}/{arm}] {k + 1}/{args.stories} | {time.time() - t0:.1f}s",
                          flush=True)
            texts = [cells[f"0:{k}"] for k in range(args.stories) if f"0:{k}" in cells]
            valid = [t for t in texts if dom.valid(t)[0]]
            broken = np.mean([dom.violations(t) for t in valid]) if valid else float("nan")
            line = (f"  [{name}/{arm}] done: {len(valid)}/{len(texts)} valid, "
                    f"{broken:.2f} of {len(dom.rules)} rules broken")
            if dom.correct:
                verdicts = [dom.correct(t) for t in valid]
                line += f", {sum(v is True for v in verdicts)}/{len(valid)} correct"
            print(line, flush=True)


if __name__ == "__main__":
    main()
