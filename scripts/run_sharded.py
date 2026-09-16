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
    if rest and rest[0] == "--":
        rest = rest[1:]

    procs = []
    for i in range(shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        cmd = [sys.executable, "-u", str(RUNNER), *rest,
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
    sys.exit(max(codes) if codes else 0)


def merge(out: Path) -> None:
    """Fold the shard directories back into the single tree everything expects.

    Two reasons. Kaggle's output download returned one shard's subtree and
    silently dropped the other, so half a run's conditions never came back; one
    tree with every condition in it is both smaller and the shape the scorer and
    the resume path already read. And a resumed run needs one state file: with
    the results still split by shard, re-running the same command would find no
    record of the conditions the other GPU produced and generate them again.
    """
    import json
    import shutil

    states = sorted(out.glob("shard*/*/state.json"))
    if not states:
        print("  nothing to merge", flush=True)
        return
    model_dir = out / states[0].parent.name
    model_dir.mkdir(parents=True, exist_ok=True)

    merged, task = {}, None
    extra = {"rms_scale": {}, "entropy": {}}
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
