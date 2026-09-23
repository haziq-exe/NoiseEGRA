#!/usr/bin/env python
"""Run one suite across every GPU the kernel has.

    python scripts/run_sharded.py <out-dir> <shards> -- <runner args...>

A Kaggle kernel is given two T4s and a single model uses one of them, so half the
hardware sat idle for every run before this. Qwen3-1.7B is about 3.4 GB in
float16, so a second copy fits on the other card comfortably.

Separate processes rather than threads: the generation hooks keep per-call state
on the model object, so two threads sharing one model would interleave their
decode-step counters and corrupt the perturbation schedule without saying so.

Each shard writes its own output directory and extracts its own steering vectors.
That duplicates a couple of minutes of setup and costs no wall clock, because it
happens on both cards at once.

Output from each shard is prefixed with its GPU so the interleaved log stays
readable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_english_experiment.py"


def pump(stream, tag: str) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"[{tag}] {line}")
        sys.stdout.flush()
    stream.close()


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit("usage: run_sharded.py <out-dir> <shards> -- <runner args...>")
    out, shards, rest = argv[0], int(argv[1]), argv[2:]
    # --then-score: judge every arm's stories with NoveltyBench's classifier once
    # the shards are merged, on the GPUs the run already has.
    # Anything between it and "--" is passed to the scorer (e.g. --extra FILE).
    # --runner PATH: another generator taking --shard and --out (scripts/run_domains.py).
    runner = RUNNER
    if len(rest) >= 2 and rest[0] == "--runner":
        runner = ROOT / rest[1]
        rest = rest[2:]
    then_score = bool(rest) and rest[0] == "--then-score"
    score_args = []
    if then_score:
        rest = rest[1:]
        while rest and rest[0] != "--":
            score_args.append(rest.pop(0))
    if rest and rest[0] == "--":
        rest = rest[1:]

    seed_shards(Path(out), shards)

    procs = []
    for i in range(shards):
        # expandable_segments keeps the allocator from fragmenting into a state
        # where a large contiguous block cannot be found. An 8B model on a 16 GB
        # T4 sits close enough to the ceiling that a steered arm died on a 1.16 GB
        # prefill allocation with 0.5 GB free and 1.16 GB reserved-but-unallocated
        # -- exactly the fragmentation this setting addresses.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i),
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-u", str(runner), *rest,
               "--shard", f"{i}/{shards}", "--out", f"{out}/shard{i}"]
        print(f"[gpu{i}] {' '.join(cmd[2:])}", flush=True)
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=pump, args=(p.stdout, f"gpu{i}"), daemon=True)
        t.start()
        procs.append((p, t))

    codes = []
    for p, t in procs:
        p.wait()
        t.join(timeout=30)
        codes.append(p.returncode)

    merge(Path(out))
    print(f"=== {shards} shard(s) finished, exit codes {codes} ===", flush=True)
    if then_score:
        # A scoring failure costs the scores, never the stories: they are merged
        # and checkpointed above, and can be judged again anywhere.
        print("=== judging the stories (NoveltyBench classifier) ===", flush=True)
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "score_novelty.py"),
                             out, *score_args]).returncode
        if rc:
            print(f"=== judging failed (exit {rc}); the stories are unaffected ===", flush=True)
    sys.exit(max(codes) if codes else 0)


def seed_shards(out: Path, shards: int) -> None:
    """Give every shard the history a previous run left behind.

    A finished run is merged into one tree, ``<out>/<model>/``, and that is what
    the checkpoint carries back on a resume. The shards, though, read and write
    ``<out>/shard<i>/<model>/``, so without this they start from nothing and
    regenerate everything the earlier run already produced. That cost one 8B run
    two hours of GPU before it was noticed: the log says "restored 8 checkpoint
    files" and then "resuming: 0 stories already saved" two lines later.

    Every file is copied, not just the state: the steering vectors and the
    activation basis are cached in the same directory, and re-extracting them is
    the other slow part of starting up.
    """
    import shutil

    merged = [p for p in out.glob("*/state.json") if not p.parent.name.startswith("shard")]
    if not merged:
        return
    for sp in merged:
        model_dir = sp.parent
        try:
            n = sum(len(c) for c in json.loads(sp.read_text()).get("runs", {}).values())
        except Exception:
            n = 0
        for i in range(shards):
            dest_dir = out / f"shard{i}" / model_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for f in model_dir.iterdir():
                if not f.is_file():
                    continue
                dest = dest_dir / f.name
                if dest.exists():
                    continue
                shutil.copy2(f, dest)
                copied += 1
            if copied:
                print(f"[seed] shard{i} starts from the merged checkpoint: "
                      f"{n} stories, {copied} files", flush=True)


def merge(out: Path) -> None:
    """Fold the shard directories back into the single tree everything expects.

    Two reasons. Kaggle's output download returned one shard's subtree and
    silently dropped the other, so half a run's conditions never came back; one
    tree with every condition in it is both smaller and the shape the scorer and
    the resume path already read. And a resumed run needs one state file: with
    the results still split by shard, re-running the same command would find no
    record of the conditions the other GPU produced and generate them again.
    """
    import shutil

    states = sorted(out.glob("shard*/*/state.json"))
    if not states:
        print("  nothing to merge", flush=True)
        return
    model_dir = out / states[0].parent.name
    model_dir.mkdir(parents=True, exist_ok=True)

    merged, task = {}, None
    extra = {"rms_scale": {}, "entropy": {}}
    # Start from whatever a previous run left here. Rebuilding this file from the
    # shard states alone discards the history the checkpoint restored, so a
    # resumed run would come back holding only the stories it happened to
    # generate this time -- losing completed conditions from the earlier run.
    existing = model_dir / "state.json"
    if existing.is_file():
        try:
            blob = json.loads(existing.read_text())
        except Exception:
            blob = {}
        task = blob.get("task")
        for key in extra:
            extra[key].update(blob.get(key, {}) or {})
        for rid, cells in blob.get("runs", {}).items():
            merged.setdefault(rid, {}).update(cells)
    for sp in states:
        blob = json.loads(sp.read_text())
        task = task or blob.get("task")
        for key in extra:
            extra[key].update(blob.get(key, {}) or {})
        for rid, cells in blob.get("runs", {}).items():
            merged.setdefault(rid, {}).update(cells)
        for f in sp.parent.iterdir():
            if not f.is_file() or f.name == "state.json":
                continue
            dest = model_dir / f.name
            if f.suffix == ".csv":
                # Per-condition scores: each shard writes only the conditions it
                # ran, so these have to be concatenated. Letting the first one win
                # -- which is what the `not dest.exists()` rule below does, and
                # did -- silently dropped the other GPU's diversity numbers, and a
                # missing row reads exactly like a condition that was never run.
                rows = f.read_text().splitlines()
                if dest.exists():
                    prev = dest.read_text().splitlines()
                    header = prev[0] if prev else (rows[0] if rows else "")
                    body = prev[1:] + rows[1:]
                    dest.write_text("\n".join([header] + body) + "\n")
                else:
                    shutil.copy2(f, dest)
            elif not dest.exists():            # the shards duplicate the vectors
                shutil.copy2(f, dest)

    (model_dir / "state.json").write_text(
        json.dumps({"task": task, "runs": merged, **extra}, indent=0))
    for d in out.glob("shard*"):
        shutil.rmtree(d, ignore_errors=True)
    n = sum(len(v) for v in merged.values())
    print(f"  merged {len(states)} shards -> {len(merged)} conditions, "
          f"{n} stories in {model_dir}", flush=True)


if __name__ == "__main__":
    main()
