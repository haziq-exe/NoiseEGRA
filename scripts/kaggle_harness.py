#!/usr/bin/env python
"""Run experiments on Kaggle's GPUs from here, without the browser.

Kaggle will run a script headless on two T4s and hand back whatever it wrote to
``/kaggle/working``. This drives that loop: it packages the checkpoint, pushes a
kernel pinned to an exact commit of this repo, waits, and pulls the results and
the log back to disk. One command per experiment, and the answer lands locally.

    python scripts/kaggle_harness.py check
    python scripts/kaggle_harness.py run --name qwen-gate -- \\
        scripts/run_english_experiment.py --model Qwen3-8B --task generic \\
        --stories 100 --suite gate --alpha 0.4

Everything about one experiment lives under ``experiments/<name>/``:

    state/     the checkpoint, uploaded before a run and refreshed after it
    output/    what the kernel wrote to /kaggle/working
    log.txt    the kernel's stdout
    kernel/    the generated script and metadata, regenerated on every push

State survives between runs because it round-trips through a private Kaggle
dataset: uploaded before the run, mounted read-only by the kernel, copied into
the working directory, and downloaded again afterwards. So a run that hits the
session limit is resumed by re-issuing the same command, exactly as on the
notebook.

Credentials come from ``~/.kaggle/kaggle.json`` (Kaggle account page -> Settings
-> API -> Create New Token), or ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.

The kernel's stdout streams back live while it runs, so a long sweep is watchable
rather than opaque, and every line is mirrored to ``log.txt`` as it arrives. Use
``follow`` to reattach after closing the terminal.

The one limit left is the session cap: about nine hours on GPU against a weekly
quota, so a sweep that exceeds it has to be resumed -- which the checkpoint
already handles.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
VENV_PY = Path.home() / ".cache" / "noiseegra-harness" / "venv" / "bin" / "python"

DEFAULT_REPO = "https://github.com/haziq-exe/NoiseEGRA.git"
KERNEL_PREFIX = "noiseegra"
POLL_SECONDS = 60

# Kaggle slugs: lowercase letters, digits and hyphens, 5-50 characters.
_SLUG_OK = re.compile(r"^[a-z0-9-]{5,50}$")


# --------------------------------------------------------------------------- #
#  Kaggle client                                                               #
# --------------------------------------------------------------------------- #

USER_CACHE = EXPERIMENTS / ".kaggle-username"

CRED_HELP = """could not authenticate with Kaggle.

  Easiest, opens a browser once and caches the result:

      {venv} -m kaggle auth login

  Or make a token at kaggle.com/settings/api ("Create New Token"), which
  downloads kaggle.json, and put it where the client looks:

      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json

  Either grants full access to the account, so keep it out of the repo.
  KAGGLE_USERNAME with KAGGLE_KEY, or KAGGLE_API_TOKEN, work too.

  If you have logged in and still see this, the login may have gone to a
  different interpreter's kaggle install. Check which one holds it:

      {venv} -m kaggle auth login --force"""


def _authenticate():
    """Return an authenticated client, or None.

    The client supports several credential stores -- an OAuth file, a legacy
    kaggle.json, an access token, environment variables -- and where each lives
    has changed between versions. Rather than guess filenames, this asks the
    client to authenticate and believes the answer. The library prints its own
    help and calls exit() when it cannot, so its output is captured and swapped
    for ours.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
    except ImportError:
        raise SystemExit(
            "the kaggle package is not importable from this interpreter.\n"
            f"  Use the harness venv:  {VENV_PY} scripts/kaggle_harness.py ...\n"
            "  Or install it here:     pip install kaggle"
        )
    except SystemExit:
        return None
    except Exception:
        return None
    # An anonymous fallback authenticates without identifying anyone, which fails
    # later on anything that needs an account.
    values = getattr(api, "config_values", {}) or {}
    if not (values.get("username") or values.get("token") or values.get("key")
            or os.environ.get("KAGGLE_USERNAME")):
        return None
    return api


def _api():
    """An authenticated Kaggle client, with a readable error if it is not set up."""
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.is_file() and (token.stat().st_mode & 0o077):
        token.chmod(0o600)
    api = _authenticate()
    if api is None:
        raise SystemExit(CRED_HELP.format(venv=VENV_PY))
    return api


def _username(api, override: Optional[str] = None) -> str:
    """The account the kernels and datasets belong to.

    There is no whoami endpoint, and OAuth login leaves no username in the config,
    so this tries the obvious sources and then caches whatever it finds. The
    username is not a secret; it is half of every kernel and dataset id.
    """
    if override:
        USER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        USER_CACHE.write_text(override.strip(), encoding="utf-8")
        return override.strip()
    for candidate in (
        os.environ.get("KAGGLE_USERNAME"),
        (getattr(api, "config_values", {}) or {}).get("username"),
        USER_CACHE.read_text(encoding="utf-8").strip() if USER_CACHE.is_file() else None,
    ):
        if candidate:
            return candidate
    try:  # last resort: read it off anything already on the account
        for listing in (api.kernels_list(mine=True, page_size=1),
                        api.dataset_list(mine=True, page_size=1)):
            for item in listing or []:
                ref = str(getattr(item, "ref", item))
                if "/" in ref:
                    name = ref.split("/")[0]
                    USER_CACHE.parent.mkdir(parents=True, exist_ok=True)
                    USER_CACHE.write_text(name, encoding="utf-8")
                    return name
    except Exception:
        pass
    raise SystemExit(
        "could not work out the Kaggle username. Pass it once and it is cached:\n"
        "    python scripts/kaggle_harness.py check --user <your-kaggle-username>"
    )


# --------------------------------------------------------------------------- #
#  Repo state                                                                  #
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def repo_state(require_clean: bool = True):
    """(branch, commit) that the kernel will clone, after checking it is pushed.

    The kernel clones a commit, not a working directory, so an unpushed change is
    a change the run will not see. Better to refuse than to run yesterday's code
    and spend an hour finding out.
    """
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    if dirty and require_clean:
        raise SystemExit(
            "the working tree has uncommitted changes, which the kernel will not see:\n"
            + "\n".join("    " + l for l in dirty.splitlines()[:20])
            + "\n\n  Commit and push them, or pass --allow-dirty to run the last "
              "pushed commit anyway."
        )
    try:
        _git("merge-base", "--is-ancestor", commit, f"origin/{branch}")
    except subprocess.CalledProcessError:
        if require_clean:
            raise SystemExit(
                f"HEAD ({commit[:8]}) is not on origin/{branch}. Push first:\n"
                f"    git push origin {branch}"
            )
    return branch, commit


# --------------------------------------------------------------------------- #
#  Generated kernel                                                            #
# --------------------------------------------------------------------------- #

KERNEL_TEMPLATE = '''"""Generated by scripts/kaggle_harness.py. Do not edit here; edit the harness."""
import os, shutil, subprocess, sys, time
from pathlib import Path

REPO      = {repo!r}
COMMIT    = {commit!r}
STATE_DIR = {state_dir!r}
OUT_NAME  = {out_name!r}
COMMAND   = {command!r}
SPACY     = {spacy}

WORK = Path("/kaggle/working")
OUT  = WORK / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def sh(cmd, **kw):
    print(f"$ {{cmd}}", flush=True)
    return subprocess.run(cmd, shell=True, check=kw.pop("check", True), **kw)


print("=" * 70, flush=True)
print(f"commit {{COMMIT[:8]}}   out {{OUT}}", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true", check=False)
print("=" * 70, flush=True)

# ---- restore the checkpoint -------------------------------------------------
src = Path("/kaggle/input") / STATE_DIR
archive = src / "state.tgz"
if archive.is_file():
    import tarfile
    with tarfile.open(archive) as tar:
        tar.extractall(OUT)
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"restored {{n}} checkpoint files from {{archive}}", flush=True)
elif src.is_dir():
    n = 0
    for item in src.rglob("*"):
        if item.is_file():
            dest = OUT / item.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            n += 1
    print(f"restored {{n}} loose checkpoint files from {{src}}", flush=True)
else:
    print(f"no checkpoint mounted at {{src}}; starting fresh", flush=True)

# ---- code -------------------------------------------------------------------
repo = WORK / "NoiseEGRA"
if not repo.is_dir():
    sh(f"git clone --quiet {{REPO}} {{repo}}")
sh(f"git -C {{repo}} fetch --quiet --all && git -C {{repo}} checkout --quiet {{COMMIT}}")
print("code at", sh(f"git -C {{repo}} log --oneline -1",
                    capture_output=True, text=True).stdout.strip(), flush=True)

# ---- dependencies -----------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
sh("pip install -q hf_transfer vendi-score sentence-transformers 'datasets<4' "
   "textstat 2>&1 | tail -2", check=False)
if SPACY:
    sh("python -m spacy download en_core_web_sm -q 2>&1 | tail -2", check=False)

# ---- run --------------------------------------------------------------------
cmd = f"cd {{repo}} && python -u " + COMMAND + f" --out {{OUT}}"
print("=" * 70, flush=True)
rc = subprocess.run(cmd, shell=True).returncode
print("=" * 70, flush=True)
print(f"exit code {{rc}} after {{(time.time() - t0) / 60:.1f}} min", flush=True)

# ---- leave only the results in /kaggle/working ------------------------------
shutil.rmtree(repo, ignore_errors=True)
for junk in WORK.glob("*.log"):
    junk.unlink(missing_ok=True)
total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"returning {{total / 1e6:.1f}} MB from {{OUT}}", flush=True)
sys.exit(rc)
'''


def write_kernel(folder: Path, *, kernel_id: str, title: str, repo: str, commit: str,
                 state_dir: str, out_name: str, command: str, gpu: bool,
                 dataset_sources: List[str], spacy: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(
        KERNEL_TEMPLATE.format(repo=repo, commit=commit, state_dir=state_dir,
                               out_name=out_name, command=command, spacy=spacy),
        encoding="utf-8",
    )
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id,
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  State dataset                                                               #
# --------------------------------------------------------------------------- #

STATE_ARCHIVE = "state.tgz"


def sync_state_up(api, state_dir: Path, dataset_id: str, title: str) -> bool:
    """Upload the checkpoint as a private dataset version. False if there is none.

    The checkpoint goes up as a single tar.gz rather than as loose files. Kaggle's
    own handling of directories and archives in a dataset varies with the upload
    mode and with whether it decides to unzip for you; a tarball is one file that
    comes back exactly as it went, and the kernel unpacks it itself.
    """
    files = [f for f in state_dir.rglob("*") if f.is_file()
             and f.name != "dataset-metadata.json"]
    if not files:
        return False

    staging = state_dir.parent / "upload"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    archive = staging / STATE_ARCHIVE
    with tarfile.open(archive, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(state_dir)))
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": dataset_id, "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")

    raw = sum(f.stat().st_size for f in files) / 1e6
    print(f"  checkpoint: {len(files)} files, {raw:.1f} MB -> "
          f"{archive.stat().st_size / 1e6:.1f} MB compressed", flush=True)
    try:
        api.dataset_create_version(str(staging), version_notes="harness sync",
                                   quiet=True)
    except Exception as exc:
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise
        print(f"  creating {dataset_id}", flush=True)
        api.dataset_create_new(str(staging), public=False, quiet=True)
        time.sleep(15)  # a new dataset takes a moment to become mountable
    shutil.rmtree(staging, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
#  Commands                                                                    #
# --------------------------------------------------------------------------- #

def cmd_check(args) -> None:
    api = _api()
    user = _username(api, args.user)
    print(f"authenticated as {user}")
    try:
        branch, commit = repo_state(require_clean=False)
        print(f"repo: {branch} at {commit[:8]}")
    except Exception as exc:
        print(f"repo: {exc}")
    mine = api.kernels_list(mine=True, search=KERNEL_PREFIX, page_size=20)
    print(f"{len(mine)} harness kernel(s) on the account:")
    for k in mine:
        print(f"  {getattr(k, 'ref', k)}")
    print("\nGPU quota is not exposed by the API; check it at "
          "kaggle.com/settings -> Accelerators.")


def cmd_run(args) -> None:
    api = None if args.dry_run else _api()
    user = args.user or (os.environ.get("KAGGLE_USERNAME") if args.dry_run else None) \
        or (USER_CACHE.read_text(encoding="utf-8").strip()
            if args.dry_run and USER_CACHE.is_file() else None)
    if not args.dry_run:
        user = _username(api, args.user)
    user = user or "USERNAME"
    name = args.name
    if not _SLUG_OK.match(f"{KERNEL_PREFIX}-{name}"):
        raise SystemExit(f"--name {name!r} must make a slug of lowercase letters, "
                         "digits and hyphens, 5-50 characters in total")

    branch, commit = repo_state(require_clean=not args.allow_dirty)
    exp = EXPERIMENTS / name
    state_dir, out_dir = exp / "state", exp / "output"
    state_dir.mkdir(parents=True, exist_ok=True)

    kernel_id = f"{user}/{KERNEL_PREFIX}-{name}"
    dataset_slug = f"{KERNEL_PREFIX}-{name}-state"
    dataset_id = f"{user}/{dataset_slug}"

    sources = []
    if args.dry_run:
        if any(f.is_file() for f in state_dir.rglob("*")):
            sources.append(dataset_id)
    elif not args.fresh and sync_state_up(api, state_dir, dataset_id,
                                          f"{KERNEL_PREFIX} {name} state"):
        sources.append(dataset_id)
    elif not args.dry_run:
        print("  no checkpoint to upload; the kernel starts fresh")

    command = " ".join(shlex.quote(a) for a in args.command)
    write_kernel(exp / "kernel", kernel_id=kernel_id,
                 title=f"{KERNEL_PREFIX} {name}", repo=args.repo, commit=commit,
                 state_dir=dataset_slug, out_name=name, command=command,
                 gpu=not args.no_gpu, dataset_sources=sources, spacy=not args.no_spacy)

    if args.dry_run:
        print(f"would push {kernel_id}")
        print(f"  commit  {commit[:8]} ({branch})")
        print(f"  command {command}")
        print(f"  sources {sources or 'none'}")
        print(f"\n--- {exp / 'kernel' / 'kernel-metadata.json'} ---")
        print((exp / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
        print(f"--- {exp / 'kernel' / 'run.py'} ---")
        print((exp / "kernel" / "run.py").read_text(encoding="utf-8"))
        return

    print(f"\npushing {kernel_id}")
    print(f"  commit  {commit[:8]} ({branch})")
    print(f"  command {command}")
    api.kernels_push(str(exp / "kernel"))
    if args.no_wait:
        print("pushed. Watch it with:\n"
              f"    python scripts/kaggle_harness.py follow --name {name}")
        return
    _wait_and_pull(api, kernel_id, exp, out_dir, state_dir, args.timeout)


TERMINAL = ("complete", "error", "cancelacknowledged", "cancelled",
            "cancelrequested", "failed", "unknown")
_LOG_RETRY_DELAY = 5
_LOG_MAX_SILENT_FAILURES = 6


def _stream_exceptions():
    """Connection faults worth reconnecting through, as classes to catch.

    OSError is in the list deliberately: requests wraps its connection errors in
    an IOError subclass, the standard library raises OSError-family errors from
    sockets, and both mean the same thing here -- the connection went away, try
    again.
    """
    out = [OSError]
    try:
        import requests.exceptions as rex

        out += [rex.ChunkedEncodingError, rex.ConnectionError, rex.Timeout]
    except Exception:
        pass
    try:
        from urllib3 import exceptions as uex

        out.append(uex.ProtocolError)
    except Exception:
        pass
    return tuple(out) or (OSError,)


def follow_logs(api, kernel_id: str, log_path: Path) -> None:
    """Print the kernel's stdout as it is produced, mirroring it to ``log_path``.

    Kaggle proxies a running session's output as server-sent events and switches
    to the persisted blob once the session ends, so the same call covers both. The
    load balancer drops idle connections every few minutes; on reconnect the
    server replays from the start, so events are counted and the ones already
    printed are skipped.
    """
    if not hasattr(api, "kernels_logs_stream"):
        print("  (this kaggle client cannot stream logs; waiting quietly)", flush=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    seen, silent_failures = 0, 0
    retryable = _stream_exceptions()
    with log_path.open("w", encoding="utf-8") as fh:
        while True:
            before = seen
            try:
                for index, event in enumerate(api.kernels_logs_stream(kernel_id)):
                    if index < seen:
                        continue
                    seen = index + 1
                    data = event.get("data")
                    if data is None:
                        continue
                    line = data.rstrip("\n")
                    print(line, flush=True)
                    fh.write(line + "\n")
                    fh.flush()
                return
            except retryable:
                silent_failures = 0 if seen > before else silent_failures + 1
                if silent_failures >= _LOG_MAX_SILENT_FAILURES:
                    print("  (log stream kept dropping with no new output; "
                          "falling back to status polling)", flush=True)
                    return
                if silent_failures > 1:
                    print("  (log stream dropped, reconnecting)", flush=True)
                time.sleep(_LOG_RETRY_DELAY)
            except Exception as exc:
                print(f"  (log stream unavailable: {type(exc).__name__}: {exc}; "
                      "falling back to status polling)", flush=True)
                return


def _norm_status(value) -> str:
    """A bare lowercase state name.

    The client returns an enum whose str() is "KernelWorkerStatus.COMPLETE", and
    older versions returned the plain string "complete". Comparing against the
    raw value silently never matches, so a finished run looks like it is still
    going until the timeout fires.
    """
    text = str(value if not hasattr(value, "name") else value.name)
    return text.rsplit(".", 1)[-1].strip().lower()


def _status(api, kernel_id: str):
    """(normalised state, failure message or None)."""
    raw = api.kernels_status(kernel_id)
    if isinstance(raw, dict):
        return _norm_status(raw.get("status", "unknown")), raw.get("failureMessage")
    return (_norm_status(getattr(raw, "status", "unknown")),
            getattr(raw, "failureMessage", None))


def _wait_and_pull(api, kernel_id, exp: Path, out_dir: Path, state_dir: Path,
                   timeout_min: int) -> None:
    print(f"\nwatching kaggle.com/{kernel_id}\n")
    t0 = time.time()

    # Attach only once the session is actually running: a queued kernel has no
    # stream yet, and attaching early would replay the previous run's log.
    last = None
    while True:
        status, failure = _status(api, kernel_id)
        if status != last:
            print(f"  [{(time.time() - t0) / 60:5.1f} min] {status}", flush=True)
            last = status
        if status == "running" or status in TERMINAL:
            break
        if (time.time() - t0) / 60 > timeout_min:
            raise SystemExit(f"still {status} after {timeout_min} min")
        time.sleep(15)

    if status not in TERMINAL:
        print("-" * 70, flush=True)
        follow_logs(api, kernel_id, exp / "log.txt")
        print("-" * 70, flush=True)

    # The stream ends slightly before the session is marked finished.
    while True:
        status, failure = _status(api, kernel_id)
        if status in TERMINAL:
            break
        if (time.time() - t0) / 60 > timeout_min:
            raise SystemExit(f"still {status} after {timeout_min} min; "
                             f"pull it later with --name {exp.name}")
        time.sleep(POLL_SECONDS)

    mins = (time.time() - t0) / 60
    print(f"\n{status} after {mins:.1f} min" + (f"\n  {failure}" if failure else ""))
    _pull(api, kernel_id, exp, out_dir, state_dir)


def _pull(api, kernel_id: str, exp: Path, out_dir: Path, state_dir: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel_id, path=str(out_dir), quiet=True)

    streamed = exp / "log.txt"
    streamed_len = len(streamed.read_text(encoding="utf-8")) if streamed.is_file() else 0
    logs = list(out_dir.glob("*.log"))
    if logs:
        text = logs[0].read_text(encoding="utf-8", errors="replace")
        try:  # Kaggle logs are JSON records, not plain text
            text = "\n".join(r.get("data", "").rstrip()
                             for r in json.loads(text) if isinstance(r, dict))
        except json.JSONDecodeError:
            pass
        if len(text) >= streamed_len:
            streamed.write_text(text, encoding="utf-8")
            print(f"  log     -> {streamed} ({len(text.splitlines())} lines)")
        logs[0].unlink()

    # The results become the checkpoint for the next run.
    inner = out_dir / exp.name
    payload = inner if inner.is_dir() else out_dir
    files = [f for f in payload.rglob("*") if f.is_file()]
    if files:
        shutil.rmtree(state_dir, ignore_errors=True)
        shutil.copytree(payload, state_dir)
        (state_dir / "dataset-metadata.json").unlink(missing_ok=True)
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"  results -> {payload}  ({len(files)} files, {size:.1f} MB)")
        print(f"  state   -> {state_dir}  (the next run resumes from here)")
    else:
        print("  the kernel returned no files")


def cmd_status(args) -> None:
    api = _api()
    kernel_id = f"{_username(api)}/{KERNEL_PREFIX}-{args.name}"
    status, failure = _status(api, kernel_id)
    print(f"{kernel_id}: {status}" + (f"\n  {failure}" if failure else ""))


def cmd_pull(args) -> None:
    api = _api()
    exp = EXPERIMENTS / args.name
    kernel_id = f"{_username(api)}/{KERNEL_PREFIX}-{args.name}"
    _pull(api, kernel_id, exp, exp / "output", exp / "state")


def cmd_wait(args) -> None:
    api = _api()
    exp = EXPERIMENTS / args.name
    kernel_id = f"{_username(api)}/{KERNEL_PREFIX}-{args.name}"
    _wait_and_pull(api, kernel_id, exp, exp / "output", exp / "state", args.timeout)


def cmd_follow(args) -> None:
    api = _api()
    exp = EXPERIMENTS / args.name
    kernel_id = f"{_username(api)}/{KERNEL_PREFIX}-{args.name}"
    status, _ = _status(api, kernel_id)
    print(f"{kernel_id}: {status}")
    if status in TERMINAL:
        print("that session has finished; showing the persisted log")
        text = api.kernels_logs(kernel_id) if hasattr(api, "kernels_logs") else ""
        if text:
            (exp / "log.txt").parent.mkdir(parents=True, exist_ok=True)
            (exp / "log.txt").write_text(text, encoding="utf-8")
            print(text)
        return
    _wait_and_pull(api, kernel_id, exp, exp / "output", exp / "state", args.timeout)


def cmd_log(args) -> None:
    log = EXPERIMENTS / args.name / "log.txt"
    if not log.is_file():
        raise SystemExit(f"no log at {log}; run `pull --name {args.name}` first")
    lines = log.read_text(encoding="utf-8").splitlines()
    for line in (lines if args.all else lines[-args.tail:]):
        print(line)


def _reexec_in_harness_venv() -> None:
    """Re-run under the venv that has the kaggle client, if this one does not.

    The client is a heavier dependency than the rest of the repo needs, so it
    lives in its own venv. Rather than make the caller remember that path, hop
    there once and carry on.
    """
    if os.environ.get("NOISEEGRA_HARNESS_REEXEC"):
        return
    try:
        import kaggle  # noqa: F401
        return
    except Exception:
        pass
    if not VENV_PY.is_file():
        raise SystemExit(
            "the kaggle client is not installed. Create its venv once:\n"
            f"    python3 -m venv {VENV_PY.parent}\n"
            f"    {VENV_PY} -m pip install kaggle"
        )
    os.environ["NOISEEGRA_HARNESS_REEXEC"] = "1"
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify credentials and list harness kernels")
    c.add_argument("--user", help="Kaggle username; cached after the first time")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="push an experiment, wait for it, pull the results")
    r.add_argument("--name", required=True, help="experiment name, lowercase and hyphens")
    r.add_argument("--repo", default=DEFAULT_REPO)
    r.add_argument("--user", help="Kaggle username; cached after the first time")
    r.add_argument("--fresh", action="store_true",
                   help="ignore the local checkpoint and start over")
    r.add_argument("--no-gpu", action="store_true")
    r.add_argument("--no-spacy", action="store_true",
                   help="skip the spaCy model, leaving tense and names on the "
                        "approximate backends")
    r.add_argument("--no-wait", action="store_true", help="push and return immediately")
    r.add_argument("--dry-run", action="store_true",
                   help="build the kernel and print it, without touching the network")
    r.add_argument("--allow-dirty", action="store_true",
                   help="run the last pushed commit even with local changes")
    r.add_argument("--timeout", type=int, default=600, help="minutes before giving up")
    r.add_argument("command", nargs=argparse.REMAINDER,
                   help="after --, the command to run inside the repo")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="is it still running")
    s.add_argument("--name", required=True)
    s.set_defaults(func=cmd_status)

    w = sub.add_parser("wait", help="wait for a running kernel, then pull")
    w.add_argument("--name", required=True)
    w.add_argument("--timeout", type=int, default=600)
    w.set_defaults(func=cmd_wait)

    f = sub.add_parser("follow", help="attach to a running kernel's live output")
    f.add_argument("--name", required=True)
    f.add_argument("--timeout", type=int, default=600)
    f.set_defaults(func=cmd_follow)

    p = sub.add_parser("pull", help="download results and log")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_pull)

    l = sub.add_parser("log", help="show the downloaded log")
    l.add_argument("--name", required=True)
    l.add_argument("--tail", type=int, default=60)
    l.add_argument("--all", action="store_true")
    l.set_defaults(func=cmd_log)

    args = ap.parse_args()
    if not getattr(args, "dry_run", False):
        _reexec_in_harness_venv()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if getattr(args, "command", None) == []:
        raise SystemExit("give the command to run after --")
    args.func(args)


if __name__ == "__main__":
    main()
