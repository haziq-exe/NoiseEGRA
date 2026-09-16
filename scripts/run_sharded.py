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

    print(f"=== {shards} shard(s) finished, exit codes {codes} ===", flush=True)
    sys.exit(max(codes) if codes else 0)


if __name__ == "__main__":
    main()
