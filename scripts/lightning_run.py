#!/usr/bin/env python
"""Run an English experiment on a Lightning AI Studio instead of a Kaggle kernel.

Why this exists
---------------
A Kaggle kernel throws its filesystem away when the session ends, so every run
paid the same forty minutes before generating a single story: installing
sentence-transformers and spaCy, then downloading 3.4 GB of Qwen weights. The
weights load in a few seconds once they are on disk -- the wait was never the
model, it was fetching the same bytes again.

A Lightning Studio keeps its disk. The home directory, and everything installed
into it, survives a stop, a restart, and a move from a processor to a GPU. So
the install happens on the first run and never again, the weights are fetched
once into a cache that stays put, and a second run starts generating in about a
minute. The code is still cloned at an exact commit on every run, so the rule
that a run reflects a pushed commit rather than a working tree is unchanged.

A Studio rather than a batch job
--------------------------------
Lightning will also run a command as a job, which needs no machine to be started
or stopped by hand. A job is given a copy of the Studio's environment, but what
it writes is thrown away when it ends, so the weights it downloads are gone by
the next job and the one thing this was built for is lost. The Studio disk is
the point, so a Studio is what this drives.

What the free plan gives you
----------------------------
One Studio running for free, a monthly credit allowance worth roughly twenty
hours on a T4, and a single GPU: a request for a two-card machine is refused on
that plan. So shards run one after another on the one card. If the rented
machine does turn out to have several cards they run at the same time, one card
each, the way they already do in a Kaggle kernel. A Studio falls asleep after
ten minutes of nothing happening, which this plan cannot change, but a running
generation counts as something happening and the run is left alone.

Usage
-----
    python scripts/lightning_run.py --name r114-wholegpu \
        --runner-args "--model Qwen3-1.7B --task generic --constraint-set whole --suite wholefive"

Results land in experiments/<name>/ in the same shape the Kaggle pull produces,
so every scoring script reads them without knowing where they were generated.

The run is detached on the Studio and this terminal only follows its log, so
closing the terminal does not kill it. Re-attach with the same name and
--attach, and the results are pulled when it ends. The machine is stopped once
the results are down, because credits are spent on the time a machine is held
rather than the time it is busy.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/haziq-exe/NoiseEGRA.git"

# The teamspace every Lightning call has to name. It is a constant with a flag
# to override rather than a default the SDK is left to guess, because the guess
# fails with a message about an unresolvable name that says nothing useful.
TEAMSPACE = "haziqkhalid04/default-project"

# lightning-sdk lives in its own virtualenv. Installing it beside the project's
# own packages upgrades scikit-learn underneath them, and the command it
# installs crashes on a version conflict with what is already there.
VENV_PY = Path.home() / ".cache" / "noiseegra-lightning" / "venv" / "bin" / "python"

# Everything under the Studio home persists; everything outside it does not.
HOME = "/teamspace/studios/this_studio"
CODE = f"{HOME}/noiseegra"
CACHE = f"{HOME}/hf"

DEPS = ("torch transformers accelerate hf_transfer sentence-transformers "
        "vendi-score 'datasets<4' textstat numpy pandas scikit-learn scipy spacy")


def under_the_lightning_venv() -> None:
    """Re-run this script with the interpreter that has the SDK.

    The virtualenv holding lightning-sdk has none of the project's own packages,
    so the argument check that runs before anything is rented still has to use
    the interpreter this was called with. That one is passed along.
    """
    try:
        import lightning_sdk  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("NOISEEGRA_LIGHTNING_REEXEC"):
        raise SystemExit(f"{VENV_PY} does not have lightning-sdk either:\n"
                         f"  {VENV_PY} -m pip install lightning-sdk")
    if not VENV_PY.is_file():
        raise SystemExit(
            "lightning-sdk is not installed. It needs a virtualenv of its own, "
            "because installing it beside this project's packages upgrades them:\n"
            f"  python -m venv {VENV_PY.parents[1]}\n"
            f"  {VENV_PY} -m pip install lightning-sdk\n"
            f"  {VENV_PY.parent / 'lightning'} login")
    env = dict(os.environ, NOISEEGRA_LIGHTNING_REEXEC="1",
               NOISEEGRA_LOCAL_PY=sys.executable)
    os.execve(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()),
                             *sys.argv[1:]], env)


def local_python() -> str:
    """The interpreter that has this project's packages, for the argument check."""
    return os.environ.get("NOISEEGRA_LOCAL_PY") or sys.executable


def driver_script(commit: str, name: str, runner_args: str,
                  shards: int, gpus: int, setup_only: bool) -> str:
    """The shell script the Studio runs, written for one run and uploaded.

    A file rather than a command string, because the runner arguments carry
    quoting of their own and a file goes across untouched.
    """
    args = " ".join(shlex.quote(a) for a in shlex.split(runner_args))
    out = f"{HOME}/runs/{name}"

    body = [
        "#!/usr/bin/env bash",
        "# Written by scripts/lightning_run.py for one run and uploaded. Editing",
        "# it here has no effect: the next run overwrites it.",
        "set -uo pipefail",
        "",
        f'OUT="{out}"',
        f'REPO="{CODE}"',
        f'export HF_HOME="{CACHE}"',
        "# The Hub client on the Studio fetches through Xet and warns on every",
        "# import if the older hf_transfer switch is set, so this is the one.",
        "export HF_XET_HIGH_PERFORMANCE=1",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        "",
        'mkdir -p "$OUT"',
        "# A machine that is stopped mid-run still leaves an exit code behind, so",
        "# the terminal following this log learns that it stopped instead of",
        "# waiting for output that is never coming.",
        "FAIL=70",
        "trap 'echo \"$FAIL\" > \"$OUT/exit\"' EXIT",
        "",
        'echo "=== machine ==="',
        'nvidia-smi -L 2>/dev/null || echo "no GPU visible"',
        "",
        'echo "=== code ==="',
        "# git writes its progress to the log even when told to be quiet, and a",
        "# carriage-return progress bar is noise in a file, so it goes aside and",
        "# is shown only if something failed.",
        'if [ ! -d "$REPO/.git" ]; then',
        f'  if ! git clone --quiet {REPO} "$REPO" > "$OUT/git.txt" 2>&1; then',
        '    cat "$OUT/git.txt"; echo "clone failed"; FAIL=64; exit 1',
        "  fi",
        "fi",
        'if ! git -C "$REPO" fetch --quiet --all > "$OUT/git.txt" 2>&1; then',
        '  cat "$OUT/git.txt"; echo "fetch failed"; FAIL=64; exit 1',
        "fi",
        'rm -f "$OUT/git.txt"',
        f'if ! git -C "$REPO" checkout --quiet --detach {commit}; then',
        f'  echo "commit {commit[:8]} is not on the remote; push it and run again"',
        "  FAIL=64; exit 1",
        "fi",
        'git -C "$REPO" log --oneline -1',
        "",
        'echo "=== dependencies ==="',
        "# Asking the interpreter beats a stamp file: it also catches an install",
        "# left half finished by a run that was interrupted.",
        "if python - >/dev/null 2>&1 <<'CHECK'",
        "import spacy",
        "import torch, transformers, accelerate, hf_transfer, sentence_transformers",
        "import vendi_score, datasets, textstat, numpy, pandas, sklearn, scipy",
        'spacy.load("en_core_web_sm")',
        "CHECK",
        "then",
        '  echo "already installed on this Studio"',
        "else",
        '  echo "first run here: installing, which happens once because the disk persists"',
        f"  if ! python -m pip install --quiet {DEPS}; then",
        '    echo "install failed"; FAIL=65; exit 1',
        "  fi",
        "  python -m spacy download en_core_web_sm",
        "fi",
        "python -c \"import torch, transformers; print('torch', torch.__version__, "
        "'transformers', transformers.__version__, "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')\"",
        "# The install can happen on a processor, where pip is still free to hand",
        "# over a build that cannot use a card. Finding that out from the timings",
        "# three hours into a rental is the failure this prevents.",
        "if nvidia-smi -L >/dev/null 2>&1 && ! python -c "
        "\"import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)\"; then",
        '  echo "the card is here but torch cannot see it; reinstall torch on this Studio"',
        "  FAIL=66; exit 1",
        "fi",
        "",
        'echo "=== weights ==="',
        'if [ -d "$HF_HOME/hub" ]; then',
        '  echo "model cache already on disk: $(du -sh "$HF_HOME/hub" | cut -f1)"',
        "else",
        '  echo "first run here: the weights are fetched once and kept"',
        "fi",
    ]

    if setup_only:
        # The weights too, not only the packages: fetched here on a processor,
        # so a GPU rental spends none of its time downloading them.
        body += [
            'echo "=== fetching weights ==="',
            "if ! python - <<'FETCH'",
            "from huggingface_hub import snapshot_download",
            'for m in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-Embedding-0.6B"):',
            "    snapshot_download(m)",
            '    print("on disk:", m, flush=True)',
            "FETCH",
            "then",
            '  echo "fetching the weights failed"; FAIL=67; exit 1',
            "fi",
            "", 'echo "=== set up, nothing to generate ==="', "FAIL=0"]
        return "\n".join(body) + "\n"

    body += ["", 'echo "=== generating ==="', "FAIL=0"]

    runner = '"$REPO/scripts/run_english_experiment.py"'
    shard_flag = [f"--shard {i}/{shards}" if shards > 1 else "" for i in range(shards)]

    if gpus >= shards > 1:
        # Several cards, so the shards run at once with a card each, the way
        # run_sharded.py runs them inside a Kaggle kernel. The exit code goes
        # through a file because waiting on a pipeline returns the status of the
        # thing prefixing the output, not the status of the generation.
        for i in range(shards):
            body += [
                f'( CUDA_VISIBLE_DEVICES={i} python -u {runner} {args} '
                f'{shard_flag[i]} --out "$OUT/shard{i}"; echo "$?" > "$OUT/rc{i}" ) '
                f'2>&1 | sed -u "s/^/[gpu{i}] /" &',
                f"P{i}=$!",
            ]
        body += [f"wait $P{i}" for i in range(shards)]
        for i in range(shards):
            body += [
                f'rc=$(cat "$OUT/rc{i}" 2>/dev/null || echo 1)',
                f'echo "shard {i} exit $rc"',
                '[ "$rc" -ne 0 ] && FAIL="$rc"',
            ]
    else:
        # One card, so one shard at a time. Sharding buys no wall clock here; it
        # only cuts the work into pieces that checkpoint separately.
        for i in range(shards):
            body += [
                f'echo "--- shard {i} of {shards} ---"',
                f'python -u {runner} {args} {shard_flag[i]} --out "$OUT/shard{i}"',
                "rc=$?",
                f'echo "shard {i} exit $rc"',
                '[ "$rc" -ne 0 ] && FAIL="$rc"',
            ]

    body += ["", 'echo "=== finished with exit $FAIL ==="']
    return "\n".join(body) + "\n"


def probe(log: str, sentinel: str, seen: int) -> str:
    """One read of the remote log: how many whole lines it holds, whether the run
    has stopped, and every line written since the last read.

    Counting lines rather than bytes holds back a line that is half written when
    the read lands, instead of delivering it in two pieces.
    """
    return (
        "n=$(wc -l < " + log + " 2>/dev/null || echo 0); "
        "s=$(cat " + sentinel + " 2>/dev/null || echo running); "
        'echo "__AT__ $n $s"; '
        'if [ "$n" -gt ' + str(seen) + " ]; then "
        'sed -n "' + str(seen + 1) + ',${n}p" ' + log + "; fi"
    )


def follow(studio, name: str, poll: float) -> int:
    """Print the run's output as it appears and return the exit code.

    The SDK's own run() collects everything and hands it over when the command
    ends, which for a three-hour sweep is three hours of silence. So the run is
    detached and its log is read instead. The reads double as a keepalive: each
    one is activity on a Studio that would otherwise start counting idle minutes.
    """
    log = f"{HOME}/runs/{name}/log.txt"
    sentinel = f"{HOME}/runs/{name}/exit"
    seen, misses = 0, 0

    while True:
        try:
            out, _ = studio.run_with_exit_code(probe(log, sentinel, seen))
            misses = 0
        except Exception as exc:              # a dropped read is not a dead run
            misses += 1
            if misses >= 20:
                print(f"lost contact with the Studio: {exc}")
                print(f"the run carries on; follow it again with "
                      f"--name {name} --attach")
                return 75
            time.sleep(poll)
            continue

        head, _, chunk = out.partition("\n")
        if not head.startswith("__AT__"):
            time.sleep(poll)
            continue
        fields = head.split()
        seen, state = int(fields[1]), fields[2]
        if chunk:
            print(chunk, flush=True)
        if state != "running":
            return int(state) if state.isdigit() else 1
        time.sleep(poll)


def require_credentials() -> None:
    """Stop before anything opens a browser or rents a machine."""
    from lightning_sdk.lightning_cloud.login import Auth

    auth = Auth()
    if auth.api_key or auth.auth_token or auth.load():
        return
    raise SystemExit(
        "no Lightning credentials found.\n"
        "  sign up at https://lightning.ai, then\n"
        f"    {VENV_PY.parent / 'lightning'} login\n"
        "  or export LIGHTNING_USER_ID and LIGHTNING_API_KEY from the account's\n"
        "  API-key page and run this again.")


def main() -> None:
    # Line by line even when this is piped into tee or a file, where Python
    # would otherwise hold its output back in a block until the buffer fills.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True,
                    help="what to call this run; results land in experiments/<name>/")
    ap.add_argument("--runner-args", default="",
                    help="arguments for scripts/run_english_experiment.py, as one string")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the suite's conditions into this many pieces")
    ap.add_argument("--machine", default="T4",
                    help="machine to rent: T4, L4, L40S. The free plan is one "
                         "GPU, so the multi-card names are refused on it")
    ap.add_argument("--studio", default="noiseegra",
                    help="which Studio to run in; made on first use and reused after")
    ap.add_argument("--teamspace", default=TEAMSPACE,
                    help="owner/teamspace the Studio belongs to")
    ap.add_argument("--interruptible", action="store_true",
                    help="rent a cheaper machine that can be taken back mid-run. "
                         "The checkpoint survives it, but the run has to be "
                         "issued again to carry on")
    ap.add_argument("--max-hours", type=float, default=8.0,
                    help="how long the machine is allocated for")
    ap.add_argument("--poll", type=float, default=8.0,
                    help="seconds between reads of the remote log")
    ap.add_argument("--keep-running", action="store_true",
                    help="leave the machine rented when the run ends, for a "
                         "second run that should not pay the startup again")
    ap.add_argument("--resume-from", default=None, metavar="EXPERIMENT_DIR",
                    help="a pulled run to continue: its saved stories and steering "
                         "directions are placed where this run looks for them")
    ap.add_argument("--attach", action="store_true",
                    help="follow a run already going under this name and pull it "
                         "when it ends, without starting anything")
    ap.add_argument("--prewarm", action="store_true",
                    help="clone and install on a processor and stop, so the GPU "
                         "run does not pay for the install. Use --machine CPU "
                         "with it; the weights are still fetched by the first "
                         "real run and kept from then on")
    args = ap.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards wants at least 1")

    commit = ""
    if not args.attach:
        if args.runner_args:
            # Reject a bad flag here rather than after a GPU has been rented.
            check = subprocess.run(
                [local_python(), str(ROOT / "scripts" / "run_english_experiment.py"),
                 *shlex.split(args.runner_args), "--dry-run"],
                cwd=ROOT, capture_output=True, text=True)
            print(check.stdout + check.stderr)
            if check.returncode != 0:
                raise SystemExit("ARGS REJECTED")
        elif not args.prewarm:
            raise SystemExit("--runner-args is required unless --attach or --prewarm")

        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            raise SystemExit("working tree is dirty; commit and push first -- "
                             "the run clones a commit, not your working tree")
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        # The Studio clones from GitHub, so a commit that exists only here is not
        # one it can check out, and finding that out costs a rental.
        on_remote = subprocess.run(
            ["git", "-C", str(ROOT), "branch", "-r", "--contains", commit],
            capture_output=True, text=True).stdout.strip()
        if not on_remote:
            raise SystemExit(f"{commit[:8]} has not been pushed; git push first -- "
                             "the Studio clones from GitHub")

    require_credentials()

    from lightning_sdk import Machine, Status, Studio

    Studio.show_progress = True
    studio = Studio(name=args.studio, teamspace=args.teamspace, create_ok=True)
    print(f"studio {args.studio} in {args.teamspace}: {studio.status}")

    if not args.attach:
        want = Machine.from_str(args.machine)
        if studio.status == Status.Running and studio.machine != want:
            print(f"switching from {studio.machine} to {want}")
            studio.switch_machine(want, interruptible=args.interruptible)
        elif studio.status != Status.Running:
            print(f"starting on {want}")
            studio.start(want, interruptible=args.interruptible,
                         max_runtime=int(args.max_hours * 3600))
        else:
            print(f"already on {want}")

        # accelerator_count on a processor machine counts cores, not cards.
        gpus = 0 if studio.machine.is_cpu() else studio.machine.accelerator_count
        print(f"running {args.name} at {commit[:8]} on {args.machine}, "
              f"{gpus} GPU(s), {args.shards} shard(s)")
        if args.shards > 1 and gpus < args.shards:
            print("  one card, so the shards run one after another")

        script = driver_script(commit, args.name, args.runner_args,
                               args.shards, gpus, args.prewarm)
        # Written through the same command channel that starts it, and checked
        # before starting. A separate file upload failed once without raising --
        # the Lightning API was answering 500 at the time -- and the launch then
        # ran a script that was not there, left a one-line error in the log, and
        # never wrote the exit marker, so the follow loop waited on nothing
        # while the GPU sat rented.
        import base64
        rdir = f"{HOME}/runs/{args.name}"
        if args.resume_from:
            # A run that stopped part-way -- a machine that ran out of credit --
            # carries on from its checkpoint rather than generating its stories
            # again: the saved stories and the extracted steering directions go
            # where this run will look for them. Each file is checked by size,
            # because an upload has failed silently here before.
            src = Path(args.resume_from)
            model_dir = next(src.glob("shard0/*/state.json")).parent
            dest = f"runs/{args.name}/shard0/{model_dir.name}"
            studio.run_with_exit_code(f"mkdir -p {HOME}/{dest}")
            for f in sorted([model_dir / "state.json", *model_dir.glob("*.pt")]):
                for attempt in range(3):
                    studio.upload_file(str(f), remote_path=f"{dest}/{f.name}",
                                       progress_bar=False)
                    out, _ = studio.run_with_exit_code(
                        f"stat -c %s {HOME}/{dest}/{f.name} 2>/dev/null || echo 0")
                    if out.strip().splitlines()[-1] == str(f.stat().st_size):
                        break
                else:
                    raise SystemExit(f"could not place {f.name} on the Studio; "
                                     "nothing was started")
            print(f"resuming from {src}: {len(list(model_dir.glob('*.pt'))) + 1} files placed")
        b64 = base64.b64encode(script.encode()).decode()
        out, code = studio.run_with_exit_code(
            f"mkdir -p {rdir} && rm -f {rdir}/exit {rdir}/log.txt",
            f"echo {b64} | base64 -d > {rdir}/driver.sh",
            f"test -s {rdir}/driver.sh && echo DRIVER_OK")
        if "DRIVER_OK" not in (out or ""):
            raise SystemExit(f"the driver script did not reach the Studio "
                             f"(exit {code}): {out!r}. Nothing was started.")

        # Detached, so the run outlives this terminal and the connection that
        # follows it. Nothing on this path holds output back until the end.
        studio.run_with_exit_code(
            f"cd {HOME} && setsid nohup bash {rdir}/driver.sh "
            f"> {rdir}/log.txt 2>&1 < /dev/null &")
        print(f"launched; following runs/{args.name}/log.txt")

    try:
        code = follow(studio, args.name, args.poll)
    except KeyboardInterrupt:
        print(f"\nstopped watching. The run carries on and the machine is still "
              f"rented; follow it again with:\n"
              f"  python scripts/lightning_run.py --name {args.name} --attach")
        return

    print(f"=== run exited {code} ===")

    if not args.prewarm:
        dest = ROOT / "experiments" / args.name
        dest.mkdir(parents=True, exist_ok=True)
        studio.download_folder(f"runs/{args.name}", str(dest))
        states = sorted(dest.glob("shard*/*/state.json"))
        print(f"pulled {len(states)} shard checkpoint(s) into experiments/{args.name}/")
        if not states:
            print("  nothing came back; the log above says why")
        elif args.shards > 1:
            sys.path.insert(0, str(ROOT / "scripts"))
            from run_sharded import merge
            merge(dest)

    if not args.keep_running:
        # Credits are spent on the time the machine is held, not the time it is
        # busy, and a Studio keeps its disk while it is stopped.
        print(f"stopping {args.studio}")
        try:
            studio.stop()
        except Exception as exc:
            print(f"  could not stop it: {exc}")

    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    under_the_lightning_venv()
    main()
