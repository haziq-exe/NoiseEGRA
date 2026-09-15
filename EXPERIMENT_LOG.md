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

#### The gain survives length matching

Vendi partly measures story length, so the live numbers above were re-scored
locally with every story cut to its first 44 words, the baseline's mean length.

| condition | words | Vendi | Vendi@44 |
|---|---|---|---|
| baseline | 43 | 3.78 | 3.89 |
| steering only | 48 | 3.52 | 3.66 |
| per-token noise a=0.4, most uncertain tenth | 49 | 3.50 | 3.71 |
| per-token noise a=0.4, uncertain half | 63 | 3.95 | 4.17 |
| per-token noise a=0.4, first 24 tokens only | 59 | 4.81 | 4.88 |
| per-story offset g=0.05, from first token | 46 | 4.07 | 4.10 |
| per-story offset g=0.05, from the prompt | 48 | 4.58 | 4.81 |
| per-story offset g=0.15, from first token | 51 | 4.66 | 4.68 |
| **per-story offset g=0.15, from the prompt** | 50 | 8.52 | **8.52** |
| per-story offset g=0.35, from first token | 171 | 10.06 | 9.81 |
| per-story offset g=0.35, from the prompt | 191 | 15.81 | 16.49 |

The winning arm holds at 8.52 after matching, against a baseline of 3.89: 2.2x,
and not a length effect -- it is 50 words against 43. The two g=0.35 arms are the
opposite case and should not be quoted: their length is three times the limit.

#### Where the constraint cost is, requirement by requirement

Pass rates over the same 40 stories.

| the story must ... | baseline | steering only | offset g=0.15 from the prompt |
|---|---|---|---|
| be at most 60 words | 100% | 95% | 82% |
| be in the present tense | 100% | 100% | 100% |
| read at grade 3 or below | 98% | 100% | 98% |
| contain a line of speech | 100% | 100% | 100% |
| open with at most 8 words | 100% | 100% | 100% |
| keep every sentence under 15 words | 100% | 100% | 100% |
| have 5 to 9 sentences | 35% | 10% | 20% |
| use no word over 3 syllables | 100% | 100% | 95% |
| spell out any number | 100% | 100% | 100% |
| name exactly one character, twice | 75% | 68% | 50% |
| start no two sentences alike | 0% | 0% | 0% |
| be one paragraph | 100% | 100% | 100% |
| **mean broken** | **1.93** | **2.27** | **2.55** |

Nine of the twelve are untouched. The whole cost is three requirements:

  - **one named character** 75% -> 50%. This is the method working as intended
    and the requirement objecting: varied stories bring in a second character, a
    Mom or a Sam or a named puppy. There is no steering direction for it -- only
    closure, present tense, simple register and dialogue are steered -- so nothing
    is holding it.
  - **at most 60 words** 100% -> 82%. Stories run longer. Closure is steered, and
    its strength is the obvious lever.
  - **5 to 9 sentences** 35% -> 20%, against 10% for steering alone. The offset is
    not the problem here; steering already is, and the offset partly recovers it.

Against steering alone rather than against the raw baseline, the offset costs
0.28 requirements, not 0.53.

Standing at g=0.15 from the prompt: length-matched Vendi 2.2x the baseline, at a
cost of 0.62 extra requirements broken per story (2.55 against 1.93). The
diversity target is met; the constraint cost is not yet acceptable, and the
breakdown says exactly which three knobs to try.

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

## Next

In priority order, from what the breakdown above says:

1. Prompt-only offsets at g = 0.15, 0.35, 0.7. If the gain comes from shifting the
   instruction and the cost from perturbing while the model writes, this separates
   them.
2. Amplification at small lambda. It compounds across the nine steered layers and
   again through the key/value cache, so start at 1.1-1.6, not 2-3.
3. A steering direction for "one named character". The single largest constraint
   loss is a requirement nothing is steering.
4. More closure. Raising the closure strength alone should buy back the 60-word
   limit without touching the diversity the offset produces.
