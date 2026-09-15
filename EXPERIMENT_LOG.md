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

### Sweep 1 results (40 stories per condition, commit 47bf76d)

Cut to 40 stories so the new arms could be compared against the existing ones at
the same group size -- Vendi grows with group size, so 40 against 100 would not
have been a comparison. `Vendi` here is over the full story.

| condition | broken | words | Vendi |
|---|---|---|---|
| baseline | 2.02 | 44 | 3.78 |
| steering only, no perturbation | 2.26 | 47 | 3.52 |
| per-token noise a=0.4, most uncertain tenth of steps | 2.38 | 49 | 3.50 |
| per-token noise a=0.4, first 24 tokens only | 2.65 | 59 | 4.83 |
| per-story offset g=0.05, from the first generated token | 2.35 | 46 | 4.08 |
| per-story offset g=0.05, from the prompt | 2.30 | 48 | 4.60 |
| per-story offset g=0.15, from the first generated token | 2.70 | 51 | 4.69 |
| **per-story offset g=0.15, from the prompt** | **2.55** | **50** | **8.57** |
| per-story offset g=0.35, from the first generated token | 4.38 | 168 | 10.11 |
| per-story offset g=0.35, from the prompt | 5.62 | 181 | 15.91 |

Four things this settles.

*The published method buys no diversity here.* Entropy-gated per-token noise
scores 3.50 against a baseline of 3.78. It is not a small gain, it is no gain.
The only arm that ever produced a large number was the ungated one, and it did so
by breaking the text.

*Where the perturbation is applied matters more than how strong it is.* At the
same magnitude, an offset that also shifts the prompt scores 8.57 against 4.69 for
one that starts at the first generated token. The instruction is where the model
decides what story to tell.

*Above g=0.15 the text breaks.* Both g=0.35 arms run to 168-181 words against a
60-word limit and break 4.4-5.6 requirements. Their Vendi of 10-16 is the same
artefact as the old ungated noise arm: broken text embeds far apart. Any Vendi
quoted next to a mean length three times the limit is measuring degradation.

*Confining per-token noise to the opening helps a little and costs a little.*
4.83 against 3.50 for the gated version, at 2.65 broken against 2.38. Worth
keeping as a comparison, not as the method.

Standing at g=0.15 from the prompt: Vendi 2.3x the baseline, at a cost of 0.53
extra requirements broken per story (2.55 against 2.02). The diversity target is
met; the constraint cost is not yet acceptable.

Reading the stories at that setting confirms the number is real. The baseline
writes "Lila runs through the park. She sees a red ball." forty times. This arm
writes a lost puppy, a sand castle washed away by the tide, slipping on a floor
next to a heavy box, chasing a butterfly. The failures are visible too: one story
loops on "Mila is named Mila", one is written entirely in lower case, one drifts
above the grade-3 register, and two slip into first person.

### 3. Prompt-only offsets and amplification (commit 9b1fa3c)

Both follow from the sweep-1 finding, and neither has been run yet.

*Prompt-only offsets.* If the gain comes from shifting the instruction and the
cost comes from perturbing while the model writes, then do only the first: hold
the offset at prefill and leave every decode step untouched. The shift can then
be much larger than 0.15 without costing fluency.

*Amplification.* Adds nothing random. At each site the component of the current
state lying in the between-story subspace is multiplied by lambda:

    h -> h + (lambda - 1) * (h - mu) B B^T

Each story is pushed further along the direction it was already taking, so two
stories that had begun to diverge are driven apart rather than jointly displaced
the way one shared offset would. B has the constraint directions projected out.

Results: pending, sweep 2.
