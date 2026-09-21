#!/usr/bin/env python
"""Run an English experiment on a rented GPU instead of a Kaggle kernel.

Why this exists
---------------
A Kaggle kernel throws its filesystem away when the session ends, so every run
paid the same forty minutes before generating a single story: installing
sentence-transformers and spaCy, then downloading 3.4 GB of Qwen weights. The
weights themselves are ready in five seconds once they are on disk -- the wait
was never the model, it was fetching the same bytes again.

Here the dependencies live in a container image that is built once, and the
model weights live on a network disk that survives between runs. A second run
starts generating in under a minute. The code itself is still cloned fresh from
a commit on every run, so the "commit and push before every run" rule is
unchanged and a new commit never triggers an image rebuild.

Usage
-----
    modal run scripts/modal_run.py --name r114-wholegpu \
        --runner-args "--model Qwen3-1.7B --task generic --constraint-set whole --suite wholefive"

Results land in experiments/<name>/ in the same shape the Kaggle pull produces,
so every scoring script reads them without knowing where they were generated.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

REPO = "https://github.com/haziq-exe/NoiseEGRA.git"
WORK = "/work"
CACHE = "/cache"

# Unpinned, matching requirements.txt and the Kaggle base image: the arms within
# a run are always compared against each other, never against another run's.
# The versions actually used are printed at the top of every log.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "transformers", "accelerate", "hf_transfer",
        "sentence-transformers", "vendi-score", "datasets<4", "textstat",
        "numpy", "pandas", "scikit-learn", "scipy", "spacy",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{CACHE}/hf",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
)

# The model weights. Downloaded by the first run that needs them, then reused.
weights = modal.Volume.from_name("noiseegra-weights", create_if_missing=True)

app = modal.App("noiseegra")


@app.function(image=image, gpu="T4", volumes={CACHE: weights},
              timeout=6 * 60 * 60, retries=0)
def run_shard(commit: str, name: str, runner_args: str,
              index: int, shards: int) -> bytes:
    """Generate one shard's stories and hand back its output directory."""
    import torch
    import transformers

    tag = f"shard{index}"

    def say(line: str) -> None:
        print(f"[{tag}] {line}", flush=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    say(f"{gpu}, torch {torch.__version__}, transformers {transformers.__version__}")

    repo = Path("/tmp/NoiseEGRA")
    if not repo.is_dir():
        subprocess.run(["git", "clone", "--quiet", REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(["git", "-C", str(repo), "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    say(f"code at {head}")

    out = Path(WORK) / name / tag
    out.mkdir(parents=True, exist_ok=True)

    cached = Path(CACHE) / "hf"
    say("weights already on disk" if cached.is_dir() else "first run: downloading weights")

    cmd = [sys.executable, "-u", str(repo / "scripts" / "run_english_experiment.py"),
           *shlex.split(runner_args), "--out", str(out)]
    if shards > 1:
        cmd += ["--shard", f"{index}/{shards}"]
    say(" ".join(cmd[2:]))

    started = time.time()
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(f"[{tag}] {line}", end="", flush=True)
    proc.wait()
    say(f"exit {proc.returncode} after {(time.time() - started) / 60:.1f} min")

    # Keep the downloaded weights for the next run even if generation failed.
    weights.commit()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=tag)
    return buf.getvalue()


@app.local_entrypoint()
def main(name: str, runner_args: str, shards: int = 1) -> None:
    root = Path(__file__).resolve().parents[1]

    # Reject a bad flag here rather than after a GPU has been rented.
    check = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_english_experiment.py"),
         *shlex.split(runner_args), "--dry-run"],
        cwd=root, capture_output=True, text=True)
    print(check.stdout + check.stderr)
    if check.returncode != 0:
        raise SystemExit("ARGS REJECTED")

    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("working tree is dirty; commit and push first -- "
                         "the run clones a commit, not your working tree")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"running {name} at {commit[:8]} on {shards} T4(s)")

    dest = root / "experiments" / name
    dest.mkdir(parents=True, exist_ok=True)
    args = [(commit, name, runner_args, i, shards) for i in range(shards)]
    for blob in run_shard.starmap(args):
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(dest)
    print(f"pulled into experiments/{name}/")

    if shards > 1:
        sys.path.insert(0, str(root / "scripts"))
        from run_sharded import merge
        merge(dest)
        print("shards merged")
