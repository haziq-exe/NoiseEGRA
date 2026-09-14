# Working on this repo

Noise steering for controlled text generation: inject calibrated perturbations
into a language model's residual stream at inference so it produces varied text
that still satisfies hard constraints. The Arabic study is published (arXiv
2604.03380); the current work generalises it to English on the
`english-generalization` branch.

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
