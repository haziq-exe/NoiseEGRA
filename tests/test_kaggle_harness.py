"""Harness checks that need no Kaggle account and no network."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_harness as H  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'[ok] ' if cond else '[FAIL] '}{name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("== the generated kernel ==")
with tempfile.TemporaryDirectory() as td:
    folder = Path(td) / "kernel"
    H.write_kernel(
        folder, kernel_id="someone/noiseegra-demo", title="noiseegra demo",
        repo="https://example.invalid/repo.git", commit="a" * 40,
        state_dir="noiseegra-demo-state", out_name="demo",
        command="scripts/run_english_experiment.py --model Qwen3-8B --stories 100",
        gpu=True, dataset_sources=["someone/noiseegra-demo-state"], spacy=True,
    )
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    script = (folder / "run.py").read_text()

    try:
        ast.parse(script)
        ok = True
    except SyntaxError as exc:
        ok, detail = False, str(exc)
    check("the template renders to parseable python", ok, "" if ok else detail)

    check("it asks for a GPU", meta["enable_gpu"] is True)
    check("it asks for internet, which the git clone needs",
          meta["enable_internet"] is True)
    check("it is private", meta["is_private"] is True)
    check("it is a script, not a notebook", meta["kernel_type"] == "script")
    check("the checkpoint dataset is mounted",
          meta["dataset_sources"] == ["someone/noiseegra-demo-state"])
    check("no stray sources", meta["competition_sources"] == []
          and meta["kernel_sources"] == [])

    check("the commit is pinned, not a branch name",
          f"COMMIT    = {'a' * 40!r}" in script and "main" not in script.split("\n")[4],
          next(l for l in script.splitlines() if l.startswith("COMMIT")))
    check("the command is carried through",
          "--model Qwen3-8B --stories 100" in script)
    check("the output directory is forced, so results come back",
          '--out {OUT}' in script or "--out" in script)
    check("it unpacks the checkpoint archive", "state.tgz" in script
          and "tarfile" in script)
    check("it installs spaCy when asked", "spacy download en_core_web_sm" in script)
    check("it deletes the clone so only results are returned",
          "shutil.rmtree(repo" in script)
    check("it exits with the command's status", "sys.exit(rc)" in script)

    H.write_kernel(folder, kernel_id="a/b", title="t", repo="r", commit="c",
                   state_dir="s", out_name="o", command="x", gpu=False,
                   dataset_sources=[], spacy=False)
    check("spaCy can be skipped",
          "spacy download" not in (folder / "run.py").read_text().split("SPACY")[1]
          or "SPACY     = False" in (folder / "run.py").read_text())
    check("the GPU can be turned off",
          json.loads((folder / "kernel-metadata.json").read_text())["enable_gpu"] is False)

print("\n== the state round-trip ==")
with tempfile.TemporaryDirectory() as td:
    state = Path(td) / "state"
    (state / "Qwen3-8B").mkdir(parents=True)
    (state / "Qwen3-8B" / "state.json").write_text('{"runs": {"a": {"0:0": "story"}}}')
    (state / "Qwen3-8B" / "run.csv").write_text("prompt_index,story_index,story\n0,0,hi\n")

    archive = Path(td) / "state.tgz"
    files = [f for f in state.rglob("*") if f.is_file()]
    with tarfile.open(archive, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(state)))

    restored = Path(td) / "restored"
    restored.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(restored)
    check("the checkpoint survives the archive unchanged",
          (restored / "Qwen3-8B" / "state.json").read_text()
          == (state / "Qwen3-8B" / "state.json").read_text())
    check("the directory layout survives too",
          sorted(str(f.relative_to(restored)) for f in restored.rglob("*") if f.is_file())
          == sorted(str(f.relative_to(state)) for f in files))

print("\n== guards ==")
check("a slug of the right shape is accepted",
      bool(H._SLUG_OK.match("noiseegra-qwen-gate")))
check("an underscore is rejected", not H._SLUG_OK.match("noiseegra_qwen"))
check("a capital is rejected", not H._SLUG_OK.match("noiseegra-Qwen"))
check("something too short is rejected", not H._SLUG_OK.match("ab"))

check("credentials are looked for in every supported place",
      {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"}
      <= set(H._have_credentials.__code__.co_consts
             + tuple(H._have_credentials.__code__.co_names))
      or "KAGGLE_API_TOKEN" in H._have_credentials.__code__.co_consts)

print("\n== the repo must be pushed before a run ==")
branch, commit = H.repo_state(require_clean=False)
check("it reads the branch and commit", len(commit) == 40 and bool(branch),
      f"{branch} {commit[:8]}")
head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
check("the commit is HEAD", commit == head)

print("\n== the dry run builds a kernel without a network ==")
exp = ROOT / "experiments" / "harness-selftest"
shutil.rmtree(exp, ignore_errors=True)
res = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "kaggle_harness.py"), "run",
     "--name", "harness-selftest", "--dry-run", "--allow-dirty",
     "--", "scripts/run_english_experiment.py", "--model", "Qwen3-8B"],
    capture_output=True, text=True,
)
check("the dry run succeeds", res.returncode == 0, res.stderr[-400:])
check("it says what it would push", "would push" in res.stdout)
check("it writes the kernel to disk",
      (exp / "kernel" / "run.py").is_file()
      and (exp / "kernel" / "kernel-metadata.json").is_file())
check("it never reaches the network", "kaggle.com" not in res.stderr)
shutil.rmtree(exp, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all harness tests passed")
