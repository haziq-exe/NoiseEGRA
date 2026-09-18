# Working on this repo

Noise steering for controlled text generation: inject calibrated perturbations
into a language model's residual stream at inference so it produces varied text
that still satisfies hard constraints. The Arabic study is published (arXiv
2604.03380); the current work generalises it to English on the
`english-generalization` branch.

**Start with `RESEARCH_STATE.md`.** It says what the project is trying to show,
what is already known, what the current best result is, and which measurement
traps have already produced wrong numbers here. `METHODS_TRIED.md` lists every
condition ever run — check it before proposing a mechanism, because one was
nearly re-run thirteen rounds after it had been shown to be a null.

## Ground rules

**Never `git add -A`.** `.gitignore` in this repo is *untracked* — it is a local
file that git has no copy of. It keeps `.env`, `.history/`, `experiment_results/`,
`paper/` and `scripts/oneoff/` out of the repo. A blanket add sweeps secrets into
a commit; it has happened. Stage files by name, and check `git show --name-only
HEAD` before pushing.

Do not overwrite `.gitignore` with `>`. Append with `>>` if you must add a rule.

**Run the tests before pushing.** They need no GPU, no network and no model
downloads; every expensive dependency is stubbed.

    for t in tests/*.py; do python "$t" || break; done

**Report what happened.** If a check fails, say so with the output. The results
here decide what goes in a paper, so a number that is wrong is worse than no
number.

**Write for a reader who has not read the code or the earlier updates.** Every
message, log entry and doc section the human reads must stand on its own. Never
put a code-internal short name in that prose: not suite names (`frontier`,
`spread`, `select`, `controls`, `sampling`), not run-id tags (`obstory`, `bud3`,
`g0p15`, `ponly`), not variable or flag names. Say what the condition does to the
model in ordinary words — "the constraint push held to a fixed total strength of
3 with a per-story perturbation of size 0.15 added at the prompt", not the tag
that encodes it. Do not refer back to "the method" or "the best arm" as if the
reader remembers it; restate what it is each time. Say what every number
measures. This is a hard requirement from the human, not a style preference.

## Layout

    noiseegra/            the library
      subspace.py         steering geometry: orthogonalisation, projections, plans
      steering_vectors.py contrastive extraction of a direction per constraint
      entropy_gate.py     applies the perturbation only where the model is unsure
      EGRA_functions.py   generation, with the hooks that inject perturbations
      constraint_metrics_en.py   twelve exact English constraint checks
      coherence.py        repetition, junk, perplexity, sentence-to-sentence checks
      diversity.py        length-matched Vendi, distinct-k, rarefaction
      run_labels.py       run ids -> readable condition names
    scripts/
      run_english_experiment.py   the runner: generates and scores conditions
      score_english.py            constraint tables from saved stories
      score_diversity.py          diversity tables from saved stories
      kaggle_harness.py           runs all of the above on Kaggle's GPUs
    tests/                        all offline, all runnable in seconds

## Running experiments on Kaggle

Local machines here have no usable GPU. Real runs go to Kaggle, which gives two
T4s and about 30 GPU-hours a week. The harness drives that from the command line.

### Once

Credentials, which only the human can create. Either form works; the client is
asked whether it is authenticated rather than being checked for a filename, so
an OAuth login and a legacy token are equally fine:

    ~/.cache/noiseegra-harness/venv/bin/python -m kaggle auth login

or a token from kaggle.com/settings/api saved to `~/.kaggle/kaggle.json`. Then:

    python scripts/kaggle_harness.py check

`check` prints the account, the branch and commit that would be used, and any
existing harness kernels.

Before trusting the loop after a change to the harness, run the end-to-end check.
It needs no GPU and no model, takes about two minutes, and verifies the three
things that matter: live output, the checkpoint round-trip, and the exit status.

    python scripts/kaggle_harness.py run --name harness-check --no-gpu --no-spacy \
        -- scripts/harness_selftest.py --seconds 40

Run it twice. The "N run(s) recorded" count must grow, which is only possible if
the checkpoint made the trip to Kaggle and back.

### Each experiment

    python scripts/kaggle_harness.py run --name qwen-gate -- \
        scripts/run_english_experiment.py --model Qwen3-8B --task generic \
        --stories 100 --suite gate --alpha 0.4

That packages the checkpoint, pushes a script kernel pinned to the current
commit, streams the kernel's stdout back live, and pulls the results into
`experiments/qwen-gate/`:

    log.txt      the kernel's output, written as it arrives
    output/      everything the kernel left in /kaggle/working
    state/       the checkpoint the next run resumes from

**Commit and push before running.** The kernel clones a commit, not your working
tree, so the harness refuses to start with uncommitted or unpushed changes.
Otherwise it runs yesterday's code and you lose an hour finding out.
`--allow-dirty` overrides this; it almost always means you forgot to push.

Other subcommands:

    status --name X     is it still running
    follow --name X     reattach to the live output after closing the terminal
    pull   --name X     download results without waiting
    log    --name X     show the log already on disk
    run ... --no-wait   push and return immediately
    run ... --dry-run   build the kernel and print it, touching no network
    run ... --fresh     ignore the local checkpoint and start over

### What to expect

A run takes a few minutes to start (queueing, then a 16 GB model download), then
roughly 10 seconds per story. One hundred stories per condition is about 17
minutes, so a four-condition sweep is a little over an hour.

A session is capped near nine hours, but the kernel stops itself at
`--max-minutes` (default 240) well before that, because GPU quota is spent by a
session being alive and cannot be got back. Either way, reissuing the same
command resumes from the checkpoint: every story is saved as it is generated.

**GPU time is only spent while a session is alive.** Watch for this:

    python scripts/kaggle_harness.py sessions     # anything still running?
    python scripts/kaggle_harness.py stop --name X

Interrupting `run` stops the *watching*, not the kernel; it says so. Check
`sessions` before finishing a piece of work, and again if a run was interrupted.
Kaggle has no cancel endpoint, so `stop` deletes the kernel to end the session --
results already checkpointed survive, but that run's output is not published. The
gentler route is the Stop Session button on the kernel's page.

### Resuming, and the traps around it

A finished run is folded into one tree, `<out>/<model>/`, and that is what the
checkpoint carries back. The shards read and write `<out>/shard<i>/<model>/`, so
`run_sharded.py` copies the merged state, the steering vectors and the activation
basis into each shard before launching, and folds the new work back into the
existing merged state rather than rebuilding it from the shards alone. Both were
bugs: without the first a resumed run regenerated everything it already had (two
hours of GPU on an 8B run, with the log saying "restored 8 checkpoint files" and
"resuming: 0 stories already saved" two lines apart), and without the second the
merge overwrote conditions the checkpoint held with only what the retry produced.
`tests/test_shard_resume.py` fails on either.

A resume only works when the task setup matches. The runner refuses to mix
stories generated under different settings -- it will tell you which value
changed, e.g. `max_new_tokens: 400 -> 300` -- because those values reach the
prompt or the generation and the stories are then not comparable. Either restore
the old value or point `--out` somewhere fresh; `--allow-task-change` exists but
mixes them.

Two more, both cheap to trip over:

* **`/tmp/<name>.log` is appended across attempts.** A watcher that greps it for
  "pulling results" fires immediately on the previous attempt's text. Poll
  `kaggle_harness.py status --name <run>` instead. `/tmp/<name>.live` is
  truncated per attempt and is safe to tail.
* **`stop` asks for confirmation.** Pass `--yes` when running it without a
  terminal, or it dies on an `EOFError` having stopped nothing.

### Stale watchers, and the 429 they cause

`kaggle_run.sh` leaves a loop polling `status` every three minutes until the run
ends. Backgrounded, that loop outlives the terminal, and it does not stop when
the run finishes if the finish was never seen -- a killed foreground, a pulled
result, a relaunch under the same name all leave one behind.

They accumulate silently and then break everything at once:

    429 Client Error: Too Many Requests for url: .../GetKernelSessionStatus

on *every* status call, including for runs that are alive and healthy, and
including new launches. Nine of them had built up over one long session, each
polling on its own three-minute cycle, and the reading was that Kaggle had
started throttling a normal workload. It had not.

    pgrep -fl kaggle_run.sh        # how many are still looping
    pkill -f kaggle_run.sh         # stop all of them

Do that before concluding anything from a 429, and after any run that was
interrupted or relaunched. The rate limit is a rolling window, so it takes some
minutes to decay after the pollers are gone.

### GPU quota, and what running out looks like

Each account gets roughly 30 GPU-hours a week, and a heavy day spends it. When
it is gone the failure is not a message about quota. The kernel pushes, the
harness reports "pushed", and then every status call answers

    404 Client Error: Not Found for url: .../GetKernelSessionStatus

because the kernel exists and no session was ever started for it. `sessions`
says nothing is running, which is true and not the point. `check` still lists
the account's kernels, so credentials are fine.

Two accounts hit this on 2026-09-18 within an hour of each other, and the first
reading was a transient API fault; it is not. Treat a push that never produces a
live log, with a 404 on status, as that account being out of GPU for the week,
and move the run to another profile.

### Running on more than one account at once

`--profile` picks the account, but the Kaggle client hard-codes its credentials
path, so `kaggle_token.py` swaps the file in place. Launching three runs on three
profiles in quick succession therefore leaves whichever went last as the account
`--profile default` resolves to. It is not cosmetic: a watcher then polls the
wrong account and reports

    Cannot access kernel 'haziqaus/noiseegra-r27-headtohead'
    (Permission 'kernels.get' was denied)

for a run that is alive and well on another account. The run itself is
unaffected -- it was launched with the right token and keeps going -- but its
results are never pulled and the log fills with permission errors.

Check with `scripts/kaggle_token.py --profile <name> --whoami` before believing
any status. The three accounts are reachable by their own names regardless of
what `default` currently points at: `coauth1` is haziqexe, `coauth2` is
haziqaus, `coauth3` is haziqcsv. Use those three and never `default` when more
than one run is in flight.

### Watching a run as it generates

`--peek-stories N` prints the opening of each condition's first N stories to the
live log as they are produced, and `--abort-broken-arms` runs the coherence
checks on a condition's first twelve stories and skips the rest of it when nearly
all of them fail. A dead condition then costs twelve stories instead of a
hundred; the threshold is 90% broken, so ordinary imperfect conditions are never
touched. Both have paid for themselves: the 14-22 layer band was diagnosed from
two aborts, and reading peeked stories is what caught the leaked planning
monologue.

### Memory on a T4

Qwen3-8B in half precision sits close enough to a 16 GB T4's ceiling that a
steered arm can die on a ~1 GB allocation with fragmentation to spare. Shards run
with `PYTORCH_ALLOC_CONF=expandable_segments:True`, which is what the error itself
recommends and what let the combined arm finish. It also costs about 140 seconds
a story there against 5 on Qwen3-1.7B, so size samples accordingly: 30 stories a
condition is about two hours for four conditions across two GPUs.

### If something goes wrong

The log is in `experiments/<name>/log.txt` whether the run succeeded or not, and
every story generated before a failure is in the checkpoint. Scoring failures in
particular do not lose work: score the saved stories afterwards with
`scripts/score_english.py --input-dir <dir> --diversity`.

## Reading results

Conditions are named from their run id, so `label_run` in `noiseegra/run_labels.py`
is the single source of truth. Do not write a second naming function; one existed
and drifted, and a three-arm sweep printed the same name three times.

Diversity numbers are only comparable within one embedding model, and only at
matched length. `Vendi` over full text partly measures story length — in one
sweep, mean word count and Vendi correlated at +0.94 — so quote `Vendi@N` and
`distinct@m`, and read the length-sensitivity table that `score_diversity.py`
prints underneath.

`distinct/k` rises on its own when stories are dropped, because the group gets
smaller. Use `distinct@m`, which rarefies every condition to the same group size.
