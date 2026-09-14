# English generalisation: what was tried, and what it did

One row per change. Numbers are from `scripts/score_english.py` (requirements)
and `scripts/score_diversity.py` (diversity), both re-run locally over the saved
stories, so every row in a table was scored by the same code.

Model: Qwen3-8B. Task: one generic instruction with twelve requirements, no
scenario, every story in a single group. Steering along four directions
(closure, present tense, simple register, dialogue) at strength 1, layers 14-22,
symmetric orthogonalisation, 36 protected dimensions.

`broken` is the mean number of the twelve requirements a story breaks, so lower
is better and the baseline is the number to match. `Vendi@N` is the effective
number of distinct stories in the group after every story is cut to its first N
words, so a group of 100 identical stories scores 1 and a group of 100 unrelated
ones scores 100; higher is better. The cut matters because longer stories embed
further apart whatever they say.

## Starting point (commit 8025cbb, 100 stories per condition)

| condition | broken | words |
|---|---|---|
| baseline | 2.02 | 44 |
| steering only, no perturbation | 2.26 | 47 |
| per-token noise a=0.4 | 4.34 | 182 |
| per-token noise a=0.4, most uncertain tenth of steps | 2.38 | 49 |
| per-token noise a=0.4, more uncertain half of steps | 2.66 | 60 |

The ungated arm collapses: stories become `She runs. She runs. She runs.` for
hundreds of tokens. Requirements broken nearly doubles and mean length goes from
44 words to 182 against a 60-word limit.

Diagnosis: the perturbation is added to the block output at the last position
and that position is written to the key/value cache, so every later step reads a
perturbed state. Over 400 decode steps the error compounds. Gating the
perturbation to uncertain steps stops the collapse but only by applying much
less of it.

## Changes

### 1. Per-story constant offsets, done properly (commit 47bf76d)

Three fixes to machinery that existed but had never been run in English.

*Magnitude.* The drawn offset was used at its natural length, so `gamma` meant a
perturbation `sqrt(dim/rank)` times weaker than the same number asked of the
noise arm: a factor of 8 at rank 64 in a 4096-wide stream. Now scaled to
`gamma * rms * sqrt(dim)`, the length of an isotropic draw at `alpha = gamma`, so
`gamma` and `alpha` are the same dimensionless quantity -- the perturbation's
length as a fraction of the hidden state's own length.

*Direction.* `collect_block_pcs` takes the principal components of individual
decode-step activations, whose leading directions describe token position (inside
a word, start of a sentence) rather than story content. `collect_story_pcs`
summarises each sampled story by the mean of its block outputs and takes the
components across stories, so the offset moves along an axis the model's own
stories already vary on.

*Site.* An offset is one fixed vector, so unlike per-token noise it has a
well-defined value during prefill. `--offset-basis story` with prefill on adds it
to the prompt positions, changing how the model reads the instruction before it
writes a token.

### 2. Perturbation confined to the opening (commit 47bf76d)

New `prefix` schedule plus a `noise_horizon` separate from the constraint
schedules' `horizon`. Per-token noise runs for the first N generated tokens and
then stops. The premise is chosen early; past that the perturbation cannot change
which story is being told and can only cost grammar.

Results: pending, sweep 1.
