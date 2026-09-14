"""Harness checks that need no Kaggle account and no network."""

from __future__ import annotations

import ast
import inspect
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
        expect_state=True,
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
    check("a command that does not mention the output directory gets --out appended",
          'command += f" --out {OUT}"' in script)
    check("and {OUT} is substituted for commands that place it themselves",
          'COMMAND.replace("{OUT}", str(OUT))' in script)
    check("it unpacks the checkpoint archive", "state.tgz" in script
          and "tarfile" in script)
    # A mount that silently fails would make a resumed run regenerate everything.
    check("a promised checkpoint that fails to mount is an error, not a fresh start",
          "EXPECT_STATE = True" in script and "sys.exit(2)" in script)
    check("it installs spaCy when asked", "spacy download en_core_web_sm" in script)
    check("it deletes the clone so only results are returned",
          "shutil.rmtree(repo" in script)
    check("it exits with the command's status", "sys.exit(rc)" in script)

    # GPU quota is spent by a session being alive, so a runaway must not be able
    # to run to Kaggle's nine-hour cap just because nobody was watching.
    check("the kernel enforces a wall-clock budget on itself",
          "MAX_MIN" in script and "TimeoutExpired" in script)
    check("it kills the whole process group, not just the shell",
          "start_new_session=True" in script and "killpg" in script)
    check("it leaves a marker so a truncated run is not read as a finished one",
          "BUDGET_REACHED" in script)

    H.write_kernel(folder, kernel_id="a/b", title="t", repo="r", commit="c",
                   state_dir="s", out_name="o", command="x", gpu=False,
                   dataset_sources=[], spacy=False, expect_state=False)
    check("a first run with no checkpoint starts fresh instead",
          "EXPECT_STATE = False" in (folder / "run.py").read_text())
    check("spaCy can be skipped",
          "spacy download" not in (folder / "run.py").read_text().split("SPACY")[1]
          or "SPACY     = False" in (folder / "run.py").read_text())
    check("the GPU can be turned off",
          json.loads((folder / "kernel-metadata.json").read_text())["enable_gpu"] is False)

# A scorer takes --input-dir and --out-dir, not --out, so the harness has to be
# able to run something other than the generator.
with tempfile.TemporaryDirectory() as td:
    folder = Path(td) / "k"
    H.write_kernel(folder, kernel_id="a/b", title="t", repo="r", commit="c",
                   state_dir="s", out_name="demo",
                   command="scripts/score_diversity.py --input-dir {OUT}/Qwen3-8B "
                           "--out-dir {OUT}/diversity",
                   gpu=False, dataset_sources=[], spacy=False)
    sc = (folder / "run.py").read_text()
    ns = {}
    exec(compile(ast.parse("\n".join(l for l in sc.splitlines()
                                      if l.startswith("COMMAND"))), "<t>", "exec"), ns)
    check("the placeholder survives into the kernel verbatim",
          "{OUT}" in ns["COMMAND"] and "--out-dir" in ns["COMMAND"], ns["COMMAND"])

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

# Credential discovery is delegated to the client rather than guessed from
# filenames: OAuth login writes ~/.kaggle/credentials.json, the legacy path is
# ~/.kaggle/kaggle.json, and both have moved between versions. A harness that
# guesses reports "no credentials" to someone who is plainly logged in.
src = inspect.getsource(H._authenticate)
check("authentication asks the client instead of guessing file paths",
      "api.authenticate()" in src and "credentials.json" not in src)
check("the client's own help is captured, not printed over ours",
      "redirect_stdout" in src)
check("an anonymous fallback is not mistaken for being logged in",
      "username" in src and "return None" in src)

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

print("\n== live log streaming ==")

import io as _io                                                  # noqa: E402
import contextlib as _ctx                                         # noqa: E402


class StreamStub:
    """A Kaggle client whose log stream drops partway through, as the real one
    does when the load balancer cuts an idle connection. On reconnect the server
    replays from the beginning."""

    def __init__(self, lines, drop_after=None, drops=1):
        self.lines = lines
        self.drop_after = drop_after
        self.drops_left = drops
        self.attach_count = 0

    def kernels_logs_stream(self, kernel):
        self.attach_count += 1
        for i, line in enumerate(self.lines):
            if (self.drop_after is not None and i == self.drop_after
                    and self.drops_left > 0):
                self.drops_left -= 1
                raise ConnectionError("load balancer cut the connection")
            yield {"data": line}


H._LOG_RETRY_DELAY = 0
LINES = [f"story {i}" for i in range(6)]

with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "log.txt"
    api = StreamStub(LINES)
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        H.follow_logs(api, "me/k", log)
    check("every line is printed as it arrives",
          buf.getvalue().splitlines() == LINES, repr(buf.getvalue()[:80]))
    check("and mirrored to the log file", log.read_text().splitlines() == LINES)

with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "log.txt"
    api = StreamStub(LINES, drop_after=3, drops=1)
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        H.follow_logs(api, "me/k", log)
    printed = [l for l in buf.getvalue().splitlines() if l.startswith("story")]
    check("a dropped connection is reconnected", api.attach_count == 2,
          str(api.attach_count))
    check("the replay is not printed twice", printed == LINES, str(printed))
    check("the log file has no duplicates either",
          log.read_text().splitlines() == LINES)


class AlwaysDrops:
    attach_count = 0

    def kernels_logs_stream(self, kernel):
        AlwaysDrops.attach_count += 1
        raise ConnectionError("nope")
        yield  # pragma: no cover


with tempfile.TemporaryDirectory() as td:
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        H.follow_logs(AlwaysDrops(), "me/k", Path(td) / "log.txt")
    check("it gives up rather than reconnecting forever",
          AlwaysDrops.attach_count == H._LOG_MAX_SILENT_FAILURES,
          str(AlwaysDrops.attach_count))
    check("and says why", "falling back to status polling" in buf.getvalue())


class NoStreamSupport:
    pass


with tempfile.TemporaryDirectory() as td:
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        H.follow_logs(NoStreamSupport(), "me/k", Path(td) / "log.txt")
    check("an older client degrades to quiet waiting",
          "cannot stream logs" in buf.getvalue())


class Unauthorised:
    def kernels_logs_stream(self, kernel):
        raise ValueError("Permission 'kernels.get' was denied")
        yield  # pragma: no cover


with tempfile.TemporaryDirectory() as td:
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        H.follow_logs(Unauthorised(), "me/k", Path(td) / "log.txt")
    check("an unexpected error does not take the run down with it",
          "log stream unavailable" in buf.getvalue(), buf.getvalue()[:90])

check("terminal states cover what Kaggle reports",
      {"complete", "error"} <= set(H.TERMINAL))


print("\n== GPU time is not left running by accident ==")

import inspect as _insp  # noqa: E402


class FakeKernel:
    def __init__(self, ref): self.ref = ref


class SessionStub:
    def __init__(self, states): self.states = states

    def kernels_list(self, **kw):
        return [FakeKernel(r) for r in self.states]

    def kernels_status(self, ref):
        return {"status": self.states[ref]}


buf = _io.StringIO()
with _ctx.redirect_stdout(buf):
    live = H.running_sessions(SessionStub({
        "me/noiseegra-a": "KernelWorkerStatus.COMPLETE",
        "me/noiseegra-b": "KernelWorkerStatus.RUNNING",
        "me/noiseegra-c": "KernelWorkerStatus.QUEUED",
    }))
check("running and queued sessions are both reported",
      sorted(live) == ["me/noiseegra-b", "me/noiseegra-c"], str(live))
check("and reported loudly", "STILL USING GPU TIME" in buf.getvalue())

buf = _io.StringIO()
with _ctx.redirect_stdout(buf):
    live = H.running_sessions(SessionStub({"me/noiseegra-a": "KernelWorkerStatus.COMPLETE"}))
check("a quiet account says so plainly",
      live == [] and "nothing is consuming GPU time" in buf.getvalue())

check("enum statuses are reduced to bare names",
      H._norm_status("KernelWorkerStatus.COMPLETE") == "complete"
      and H._norm_status("complete") == "complete"
      and H._norm_status("KernelWorkerStatus.RUNNING") == "running")

src = _insp.getsource(H.cmd_run)
check("interrupting the watcher warns that the kernel is still running",
      "STILL RUNNING" in src and "KeyboardInterrupt" in src)
check("stop exists and says what it costs",
      "kernels_delete" in _insp.getsource(H.cmd_stop)
      and "no cancel endpoint" in _insp.getsource(H.cmd_stop))


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all harness tests passed")
