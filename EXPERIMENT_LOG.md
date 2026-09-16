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

## Novelty assessment (2026-09-15)

64 papers read at method-section depth across 8 search angles, then five novelty
claims attacked by adversarial reviewers instructed to refute. Every paper cited
below was verified to exist by fetching it directly; one agent's verdict (SKOP)
was discarded because it admitted arguing from recall after exhausting its search
budget, and one author list it gave (MuCoLa) was wrong even though the paper and
arXiv id were right.

### Every mechanism part is individually anticipated

| part of the method | closest published work | verdict |
|---|---|---|
| one perturbation drawn per generation, not per token | RSP, arXiv 2605.11936 | anticipated; RSP makes this its headline design choice, in autoregressive LLMs, for diversity |
| perturbation restricted to a data-estimated activation subspace, "on the manifold" | STRIDE, arXiv 2605.11494 | anticipated; same construction, same manifold rationale, same Vendi metric, in diffusion transformers |
| a protected subspace projected out of the perturbation | InterFaceGAN (CVPR 2020) conditional manipulation; LEACE/INLP; ORBIT arXiv 2606.22357 | the operator is textbook; see below for what is not |
| applied at prompt positions during prefill, for diversity | NC-GRPO, arXiv 2608.21595 | anticipated; prompt-position hidden-state noise at prefill, inert at decode, per rollout, norm-relative scaling |
| constraint control and a diversity perturbation in one inference-time intervention, both scored | MuCoLa, arXiv 2205.12558 (EMNLP 2022) | anticipated four years ago |

There is also a 2026 line of work on exactly this objective: STARS
(arXiv 2601.22010, ICLR 2026) steers activations at inference to maximise the
geometric volume of concurrent generations -- that is, it optimises a Vendi-like
quantity directly, and gets orthogonality between its steering directions from
the Stiefel manifold. Any submission here now has to beat STARS, RSP and an
LLM port of STRIDE as baselines, not the entropy-gated noise arm.

### The one measurement that is not in any of them

The adversarial reviewer's sharpest objection was that projecting a 36-dimensional
protected span out of a perturbation in a 4096-dimensional stream removes under 1%
of it and is therefore close to a no-op. That assumes the two subspaces are
unrelated. Measured on the actual saved bases, they are not:

| offset subspace | share of its energy inside the protected span | chance level |
|---|---|---|
| between-story axes (32 sampled stories) | 8.2% | 0.88% |
| prompt-position axes (one forward pass) | 6.2% | 0.88% |

Seven to nine times chance, with a largest principal cosine of 0.59 -- there is a
direction in the diversity subspace more than half aligned with the constraint
span. The directions along which generations differ from one another are
substantially the same directions the constraints live on. That is a concrete
claim no read paper makes, and it is the reason the projection is load-bearing
rather than decorative.

It is not yet established causally. The experiment that would establish it is the
one ablation never run in English: the same offset at the same magnitude, drawn
with and without the constraint span removed (`offset_mode` orth against free). If
removing an 8% overlap buys back compliance at no cost in Vendi, that is a finding.
If it changes nothing, the projection is decoration and should be dropped.

### Score

**2.5 out of 10** against the bar of A* main-track methodological novelty, as the
method currently stands. Not because any part is wrong, but because all five parts
are individually published, three of them in 2026 papers a reviewer will know, and
the remaining contribution is a combination in which each part plays its original
role. The honest framing today is a strong empirical study, not a new mechanism.

What would move it, in order of expected effect:

1. The entanglement result above, with the causal ablation attached. A measured
   claim that diversity axes and constraint axes overlap far above chance, and
   that removing the overlap separates the two controls, is new. Would plausibly
   reach 5.
2. Steering the requirements that actually fail. Sentence count and the word cap
   are the two that break, and neither is steered; three of the four steered
   directions target requirements already passing at 98-100%.
3. Baselines against STARS, RSP and STRIDE-ported-to-an-LLM. Without them the
   submission is not reviewable.

---

## Round 2: a task the model cannot already do, and a different place to put the noise

### Why the constraints had to change

Ten of the twelve requirements in round 1 were satisfied by unmodified Qwen3-8B at
98-100%. A requirement passed at 100% cannot record what a perturbation costs,
because there is no headroom below it to lose; a requirement passed at 0% (the old
`varied_openers`) cannot either, for the same reason from the other side. Only two
of the twelve carried any signal, so "mean requirements broken" was, in effect, a
two-item scale with ten constants added to it.

Every rule is now a band the model has to land inside rather than a ceiling it has
to stay under. Landing inside a band requires planning the whole story, which is
what a perturbation disturbs.

| requirement | round 1 | round 2 | baseline pass, round 1 |
|---|---|---|---|
| length | at most 60 words | between 50 and 65 words | 100% |
| present_tense | 80% of verbs present | every verb present | 100% |
| simple_register | grade 3.0 or easier | grade 2.5 or easier | 98% |
| dialogue | at least one quoted line | exactly two quoted lines | 100% |
| easy_opening | first sentence at most 8 words | at most 5 words | 100% |
| short_sentences -> sentence_band | no sentence over 15 words | every sentence 4 to 10 words | 100% |
| sentence_count | 5 to 9 sentences | 6 to 8 sentences | 35% |
| short_words | no word over 3 syllables | no word over 2 syllables | 100% |
| one_name | one name, used twice | one name, used three times | 75% |
| varied_openers | no repeated sentence opener | no word begins more than two sentences | 0% |
| single_paragraph -> plain_punctuation | one paragraph | one paragraph, and no `;` `:` `--` `(` `)` | 100% |
| no_digits -> spelled_number | no digits | a number of two or more, written as a word | 100% |

None of them constrains the content, so the diversity measurement is not fighting
a requirement that dictates what the story is about.

Scoring round 1's baseline stories against the round 2 rules -- stories written to
the old prompt, so this is a floor and not a prediction -- gives 6.25 of 12 broken,
against 1.93 under the old rules.

### Why the steering directions had to change

Two faults, both diagnosed in round 1 and both fixed here.

**The steered set did not overlap the failing set.** Of four directions, `closure`
was not one of the twelve scored requirements at all, and the other three targeted
requirements already passing at 98-100%. The new set is the five requirements whose
violation is a *local* property of the text: present tense, simple register, quoted
speech, short sentences (`terse`), varied sentence openings. The counting rules
(total words, sentence count, one name) are asked for in the prompt and scored but
not steered, because there is no token-level direction that means "stop at 65".

**Three of four contrast sets were length-confounded**: the positive side was 11 to
14 words shorter than the negative, so "simple register" was partly "say less".
Fixed in two independent ways, because either alone can be argued with:

* every pair is now written to the same word count (the builder refuses a gap above
  one word, and a test asserts it);
* the extractor reads both sides over the *same number of token positions*,
  truncating to the shorter continuation, so even a residual wording imbalance
  cannot reach the difference vector. Mean word gap is now reported per direction
  in the extraction table.

### The new method: f(S_c)

Everything tried so far adds a perturbation *beside* the constraint push and then
works to keep the two apart -- the noise is projected out of the constraint
subspace so it cannot move a requirement. This inverts that. The perturbation is
applied *to* the constraint vector, and one vector is added to the residual
stream, not a sum of a signal and a disturbance. Three forms, one draw per
generation held fixed for the whole story:

| form | f(S) | what is preserved |
|---|---|---|
| `perp` | `S + kappa * rms * sqrt(dim) * j`, `j` unit and perpendicular to `S` | the push along `S` exactly; the step is free to lie inside the constraint subspace |
| `rotate` | `\|S\| * (S/\|S\| + kappa*j) / sqrt(1+kappa^2)` | the *length* of the push; the aim turns by atan(kappa) |
| `gain` | `sum_c beta_c * exp(kappa z_c - kappa^2/2) * s_c` | the span: nothing leaves the constraint subspace at all |

`perp` is the direct test of the round 1 entanglement measurement. The protected
subspace forbids the perturbation from touching the constraint directions; `perp`
lets it live there and holds the *net* constraint push fixed instead. If the
entanglement finding is causal, this should buy diversity that the orthogonal
offset cannot reach, at the same compliance.

`gain` is the extreme of the same idea: every story is written under a different
emphasis of the same requirements, and the perturbation never leaves the
constraint span.

kappa is read on the same scale as the offset's gamma and the noise's alpha -- the
perturbation's length as a fraction of the hidden state's own length -- so the
three families are comparable at matched dose.

### What round 2 runs

Thirteen conditions, 40 stories each. Every perturbed arm has an unperturbed
control at the same injection site, so a difference cannot be read as "the prompt
is a better place to push" when it is really "pushing harder helps".

| # | condition |
|---|---|
| 1 | baseline |
| 2 | constraint vector at the decode steps, no perturbation |
| 3 | constraint vector at the prompt only, no perturbation |
| 4 | per-token noise a=0.4, flat |
| 5 | per-token noise a=0.4, cosine decay over 64 tokens |
| 6 | per-story offset g=0.15, story basis, at the decode steps |
| 7 | per-story offset g=0.15, story basis, at the prompt only |
| 8, 9 | f(S_c) `perp` kappa=0.15, at the decode steps / at the prompt |
| 10, 11 | f(S_c) `rotate` kappa=1.0, at the decode steps / at the prompt |
| 12, 13 | f(S_c) `gain` spread 0.5, at the decode steps / at the prompt |

4 and 5 settle the question round 1 left open: cosine decay was only ever tried at
alpha 0.8 and 1.6, never at the 0.4 the flat schedule was run at, so "decay does
not help" was never actually tested at matched strength.

Results below once the run lands.

### Round 2, as it landed: the first three rows

| condition | broken /12 | words | sents | Vendi |
|---|---|---|---|---|
| baseline | 4.55 | 43 | 10.6 | 4.03 |
| the constraint vector at the decode steps | 5.05 | 48 | 13.0 | 3.49 |
| the constraint vector at the prompt only | 4.95 | 49 | 12.3 | 4.19 |

**The task is a task now.** 4.55 of 12 broken against 1.93 under the round-1
rules, with headroom in both directions. That part worked.

**Steering still hurts, and this time the cause is visible in one column.**
Sentences: 10.6 unsteered, 13.0 steered, against a rule asking for six to eight.
`terse` is the direction that ends a sentence and starts another, and it was
signed positively to serve "no sentence runs over ten words". Measured over 24
unsteered stories, the longest sentence in any of them is **seven words** -- that
ceiling is never approached. What every one of those stories breaks is the
four-word floor. So the coefficient was pushing the model further into the
violation it was already committing, and it took the other requirements with it:
shorter sentences mean fewer words, and the word count was already below the
50-word minimum.

A fixed positive coefficient assumes the model errs on one particular side of
every rule. With two-sided rules that is wrong half the time, and being wrong is
worse than not steering at all.

`--beta-calibration auto` (`noiseegra/beta_calibration.py`) states the rule
instead of guessing the sign: **steer a direction only if its requirement
actually fails, and in the direction of the side that is failing.** Measured on
unsteered generations only, so nothing about the conditions being compared enters
the coefficients. On the 24 stories above it gives

    present_tense     8% of stories want more of it   beta +1
    simple_register   8% of stories want more of it   beta +1
    dialogue         92% of stories want more of it   beta +1
    terse           100% of stories want less of it   beta -1
    varied_openers  100% of stories want more of it   beta +1

Prediction, recorded before the run that tests it: flipping `terse` alone should
move `length`, `sentence_band` and `sentence_count` together, because longer
sentences are also more words and fewer sentences. If the steered arm still
breaks more requirements than the baseline after the flip, the problem is not the
sign and constraint steering is the wrong tool for this constraint set.

**The injection site matters on its own.** The same constraint vector at the same
magnitude costs 3.49 Vendi at the decode steps and 4.19 at the prompt, with no
perturbation anywhere. Round 1 read "the offset works better from the prompt" as
a fact about the offset. Part of it is a fact about the site: pushing once while
the model reads the instruction disturbs the output distribution less than
pushing at every step, whatever is being pushed. Every perturbed arm in this
round therefore has an unperturbed control at its own site, and has to be read
against that control rather than against the baseline.

### Reproducing the subspace overlap

`scripts/subspace_overlap.py` rebuilds the protected span exactly as
`SteeringPlan.build` does and reports the overlap per layer. Run against round
1's saved bases it reproduces the hand-computed numbers -- story basis 8.22%
inside the span against 0.88% chance (9.4x), prompt basis 6.22% (7.1x), largest
principal cosine 0.588 -- and adds what the hand computation did not have: the
overlap is flat across every layer from 14 to 22 (7.7% to 8.5%), so it is not an
artefact of one layer.

### Round 2 results, all thirteen conditions

40 stories each, Vendi length-matched to 43 words (the baseline mean). Each
perturbed arm is read against the unperturbed control at *its own* injection
site, not against the baseline, because the site alone moves diversity.

| condition | broken /12 | words | Vendi@43 | vs its control | | per broken |
|---|---|---|---|---|---|---|
| baseline | 4.55 | 43 | 4.02 | | | |
| the constraint vector at the decode steps | 5.05 | 48 | 3.68 | *control* | | |
| the constraint vector at the prompt | 4.95 | 49 | 4.11 | *control* | | |
| per-token noise a=0.4, flat | 7.28 | 159 | 5.24 | +1.56 | +2.23 | 0.70 |
| per-token noise a=0.4, cosine decay over 24 | 5.50 | 44 | 4.88 | +1.20 | +0.45 | **2.67** |
| per-story offset g=0.15, at the decode steps | 5.83 | 55 | 5.62 | +1.94 | +0.78 | 2.49 |
| per-story offset g=0.15, at the prompt | 5.75 | 49 | 6.33 | +2.22 | +0.80 | **2.78** |
| f(S_c) perp k=0.15, at the decode steps | 5.75 | 56 | 4.67 | +0.99 | +0.70 | 1.41 |
| f(S_c) perp k=0.15, at the prompt | 5.95 | 53 | **6.54** | +2.43 | +1.00 | 2.43 |
| f(S_c) rotate k=1, at the decode steps | 5.03 | 47 | 4.10 | +0.42 | -0.02 | free but tiny |
| f(S_c) rotate k=1, at the prompt | 5.12 | 45 | 4.23 | +0.12 | +0.17 | 0.7 |
| f(S_c) gain 0.5, at the decode steps | 5.15 | 47 | 3.66 | -0.02 | +0.10 | ~0 |
| f(S_c) gain 0.5, at the prompt | 5.08 | 48 | 4.13 | +0.02 | +0.13 | ~0 |

**Cosine decay at matched alpha is a large win, and the earlier "decay does not
help" was wrong because it was never run at matched strength.** Round 1 jumped
straight to alpha 0.8 and 1.6 and concluded the schedule bought nothing. At the
alpha the flat schedule was actually run at, the same noise costs 0.45 broken
instead of 2.23 and still buys 1.20 Vendi instead of 1.56. The flat arm is
degenerate on inspection -- 159 words and 78.7 sentences against a 50-to-65-word,
six-to-eight-sentence rule, which is fragments, not prose -- and it generates
about six times slower for it, roughly 43 seconds a story against 6.6, because
the perturbation stops the model terminating and every sample runs to the
400-token cap. For a method whose selling point is that it is cheap at inference,
that is a cost worth stating.

**f(S_c) `perp` does not beat the orthogonal offset.** It reaches the highest
absolute diversity in the round, 6.54 against 6.33, and costs proportionally more
for it, 1.00 broken against 0.80, so per requirement spent it is slightly worse.
`perp` is the arm that is allowed to put its perturbation inside the constraint
subspace while holding the net push along the summed constraint vector exactly
fixed. If the round-1 entanglement measurement were causal -- if the axes stories
differ along overlapping the constraint span at nine times chance meant the
projection was throwing away real diversity -- this is the arm that should have
shown it, and it did not.

**`gain` is a clean null**: +0.02 Vendi. It never leaves the constraint span at
all, and it buys nothing. Taken with `perp`, that is the same conclusion from two
directions: the diversity is not in the constraint span, and the projection that
keeps the perturbation out of it is not costing anything worth recovering.

The honest reading of the entanglement result is therefore **weaker than round 1
claimed**. The overlap is real and reproducible (9.4x chance, stable across
layers 14-22). It does not follow that removing it costs diversity, and two arms
designed to test exactly that came back negative. What the `ablate` suite can
still settle is the other half: whether the projection *buys* compliance. If it
does not, the projection is decoration either way.

**`rotate` is nearly free and too small to matter**: +0.42 Vendi at -0.02 broken
at the decode steps. That is a dose limit and not a mechanism limit, as predicted
before the run -- at beta 1 a 45-degree turn of the summed constraint vector
displaces about 2.6 activation units against the offset's 14.6, and rotation is
bounded above by |S| * sqrt(2) whatever kappa is. It cannot be scaled without
scaling beta.

**Where the cost falls.** Per-requirement pass rates say the twelve are not
twelve: `simple_register`, `easy_opening` and `plain_punctuation` sit at 100% for
every condition and `varied_openers` at 5% for most, so the effective scale is
about eight. The largest single cost in the best arm is specific and
interpretable: `spelled_number` falls from 60% at baseline to 12% under f(S_c)
`perp` at the prompt, against 42% under the orthogonal offset. "Mention a number"
is not one of the five steered directions, so it is not in the protected span;
`perp` is free to move it and the offset is not. The projection protects what it
was built to protect and nothing else.

**Nothing preserves compliance.** Every arm that buys diversity costs about one
requirement of twelve. That, not the diversity, is now the open problem.

### What round 3 runs, and why

The largest untested lever is that **every arm in round 2 sits on top of steering
that was signed wrongly** and is therefore worse than no steering at all: 5.05
broken at the decode steps and 4.95 at the prompt against a 4.55 baseline. The
perturbation results are all measured from a starting point below where they
should start. `--beta-calibration auto` fixes the sign by measurement rather than
by assumption.

`--suite pareto`, eleven conditions, 440 generations:

* both unperturbed siting controls, with calibrated signs;
* the per-story offset at the prompt at gamma 0.05, 0.10, 0.15 and 0.25 -- the
  method that bought the most per requirement, swept over dose so the trade is a
  curve rather than one point;
* f(S_c) `perp` at the prompt at kappa 0.10 and 0.15, to see whether the two
  families cross anywhere on that curve;
* cosine-decayed noise at alpha 0.4 and 0.8, which was the surprise of round 2
  and has never had a dose sweep of its own.

Dropped: flat noise (degenerate and six times slower), `rotate` and `gain` (null).
The gamma 0.15 and kappa 0.15 points are directly comparable to round 2, so the
effect of the sign fix is readable off the same two rows.

## Round 4: does each direction move anything, and in which direction?

Steering had been run only ever as a block of five directions at one coefficient,
and the block lost on both axes: 4.65 requirements broken against a 4.55 baseline,
with diversity 3.27 against 4.00. A block that loses on both cannot be repaired by
tuning what is added on top of it, and "the block does not work" does not say
which of the five is at fault.

So: one direction at a time, pushed hard both ways (beta +/-3), 24 stories each,
scored on the **whole** requirement list rather than only on the requirement the
direction was extracted to serve. Directions re-extracted in the real task
context, which is a fix in its own right -- they had been measured under "You are
a creative writer / write a short story" and applied under the twelve-requirement
children's-reading prompt, which is exactly the out-of-distribution use the
extraction module's own docstring warns against.

| direction | push | broken /12 | change | what moved by 8 points or more |
|---|---|---|---|---|
| baseline | | 4.42 | | |
| present tense | +3 | 5.00 | +0.58 | words +38, tense -21, syllables -25, number -21 |
| present tense | -3 | 5.12 | +0.71 | tense -92, syllables -33, number +21 |
| **simple register** | **+3** | **3.88** | **-0.54** | **words +21, number +29** |
| simple register | -3 | 6.25 | +1.83 | easy opening -100, syllables -92, varied +38 |
| dialogue | +3 | 5.88 | +1.46 | quotes -54, opening -21 |
| dialogue | -3 | 5.00 | +0.58 | words +38, quotes -38, syllables -25 |
| short sentences | +3 | 5.92 | +1.50 | words +62, opening -83, count -21 |
| short sentences | -3 | 5.25 | +0.83 | quotes -29, band -17, number -37 |
| varied openings | +3 | 4.71 | +0.29 | words +54, band +25, syllables -88 |
| **varied openings** | **-3** | **4.21** | **-0.21** | words +17, tense +8, count -17 |

The whole block at the calibrated signs: 4.96 at 1x, 5.92 at 2x, 8.62 at 4x (204
words, degenerate). Worse at every size.

**The directions work. Three of them are harmful.** This is not a sign problem
and not a dose problem. `present_tense` at -3 drops present-tense compliance from
92% to 0% -- the stories come back in the past tense, and they are perfectly
readable while doing it -- so the direction is a strong, clean controller of the
property it was extracted for. It simply does not follow that controlling that
property helps, and for three of the five it does not help at either sign.

**Both of the directions that do help were mis-set by the previous rule.**

* `simple_register` was switched **off** by the calibration, because its own
  requirement already passes at 100% and the rule was "steer only what fails". It
  is the single most useful direction in the set: at +3 it takes the total from
  4.42 to 3.88, the first time anything has beaten the baseline. It does that
  almost entirely through requirements that are not its own -- word count +21
  points and the spelled-number rule +29 -- while its own requirement drifts down
  four points.
* `varied_openers` was set to +1 by the rule, and -3 is what helps.

So the rule was wrong, and wrong in an instructive way. **A direction's value is
not the requirement it was extracted for.** Constraint directions have large
cross-effects, and the only thing worth selecting on is the measured change in
the whole violation count. "Steer what fails, in the direction it fails" sounds
principled and is not: it cannot see that pushing an already-satisfied property
harder is what buys the two hardest requirements in the list.

### Round 5

Keep what is measured to help, at the coefficient that helps most, drop the rest.
`--suite select`, twelve conditions, 24 stories each:

* `simple_register` alone at +1.5, +3, +4.5, +6;
* `varied_openers` alone at -1.5, -3, -4.5, -6;
* the two together;
* the two together plus the per-story offset at the prompt at gamma 0.15 and 0.25.

The last two are the point of the exercise: for the first time the perturbation
is added on top of a steering configuration that is better than the baseline
rather than worse.

## Round 5: a coefficient curve, and the shape of the problem

The two directions round 4 measured as helpful, swept over coefficient, then
combined, then with the per-story perturbation on top. 24 stories per condition,
Qwen3-8B, directions extracted in the task context.

| condition | broken /12 | vs baseline | words | Vendi |
|---|---|---|---|---|
| baseline | 4.42 | | 41 | 3.61 |
| simple language +1.5 | 4.21 | -0.21 | 44 | 3.33 |
| **simple language +3** | **3.88** | **-0.54** | 47 | 3.12 |
| simple language +4.5 | 4.25 | -0.17 | 50 | 2.89 |
| simple language +6 | 4.71 | +0.29 | 51 | 3.84 |
| varied openings -1.5 | 4.71 | +0.29 | 46 | 3.38 |
| varied openings -3 | 4.21 | -0.21 | 46 | 3.00 |
| varied openings -4.5 | 5.96 | +1.54 | 149 | 2.54 |
| varied openings -6 | 6.29 | +1.88 | 184 | 2.54 |
| both together at 3 | 4.33 | -0.08 | 53 | 3.07 |
| both + per-story offset 0.15 | 5.62 | +1.21 | 74 | 5.25 |
| both + per-story offset 0.25 | 7.88 | +3.46 | 137 | 8.95 |

**The curve for `simple_register` is a clean inverted U**: -0.21, -0.54, -0.17,
+0.29 at 1.5, 3, 4.5 and 6. A smooth optimum is far better evidence that the
effect is real than the single point round 4 had. This is the first steering
configuration in the project that beats the baseline on constraint following.

**Two good directions together are worse than the better one alone**, 4.33 against
3.88. They interfere. "Steer every constraint that has a direction" was wrong in
principle and not only in its signs; what works is one direction at its own
optimum.

**`varied_openers` is fragile.** It helps at -3 and destroys the text at -4.5 and
-6, where the stories run to 149 and 184 words against a 50-to-65-word rule. The
useful band is narrow.

**And the two levers pull against each other from both ends.** Steering buys
compliance and spends diversity: 4.42 -> 3.88 broken, 3.61 -> 3.12 Vendi. The
perturbation buys diversity and spends compliance. Read off the round-3 dose
curve, the best available combination -- steering at its optimum plus the
smallest useful perturbation, gamma about 0.10 -- lands near 4.3 broken and 4.6
Vendi: constraint following level with the baseline, diversity up by about a
quarter. That is "as good as baseline on constraints and clearly better on
diversity", which is not yet "better on both".

Closing the gap needs a steering gain larger than 0.54 of a requirement. That is
what round 6 tests, on a model where the steered requirements are not already
satisfied.

## Round 7: the steering bottleneck, and when a direction can work at all

Three runs in parallel, one per account, each across both of a kernel's T4s.
Qwen3-1.7B, 24 stories a condition, reasoning blocks switched off. Thirteen
minutes a run, against two and a half hours before.

### The dose was the confound

Summing k directions at coefficient beta gives a push of length beta*sqrt(k)*rms.
"Steer one more constraint" therefore meant "push harder", and the two-direction
arm was never compared against the one-direction arm at the same strength.

| condition | broken /12 | vs baseline | Vendi |
|---|---|---|---|
| baseline | 8.08 | | 6.46 |
| all five summed at 1, as before | 7.21 | -0.87 | 5.05 |
| all five, total push held at 2 | 7.21 | -0.87 | 5.58 |
| all five, total push held at 3 | 6.83 | **-1.25** | 4.91 |
| all five, total push held at 4.5 | 6.75 | **-1.33** | 3.83 |
| **two** directions, total push held at 3 | 7.75 | -0.33 | 5.26 |
| all five, budget 3, reallocated per story | 6.92 | -1.16 | **5.65** |
| all five, budget 3, reallocated, decayed | 6.88 | -1.20 | 5.48 |

Steering now removes 1.33 requirements of twelve, a sixth of the violations. And
**at the same total push, five directions beat two by 0.92 requirements**. The
answer to a multi-constraint method that stops working is not to steer fewer
constraints; it is to stop letting the number of constraints set the strength.

Reallocating the budget per story -- same total push, drawn differently for each
story, never leaving the constraint subspace -- buys **+0.74 Vendi for +0.09
requirements** against the constant push at the same budget. That is the best
exchange rate in the project so far. It is still below the baseline's diversity,
so steering continues to homogenise; it just homogenises less.

### Why some directions cannot work, and it is not the extraction

One direction at a time, on a model that actually fails these requirements:

| direction | push | its own requirement | baseline | steered | total broken |
|---|---|---|---|---|---|
| **present tense** | +3 | present tense | **0%** | **96%** | -0.21 |
| **simple language** | +3 | grade 2.5 or easier | 71% | **100%** | -0.42 |
| **varied openings** | +3 | no opener used three times | 4% | **17%** | +0.92 |
| dialogue | +3 | exactly two quoted lines | 4% | 0% | +1.08 |
| short sentences | +3 | every sentence four to ten words | 8% | 0% | +0.42 |

Present tense goes from 0% to 96%. The extraction was never the problem: on
Qwen3-8B that requirement already passed at 92%, so the direction had nothing to
win and only side effects to show. Three of five directions clearly control the
requirement they were extracted for.

The two that fail share a property. `dialogue` asks for **exactly two** quoted
lines and `sentence_band` for sentences **between four and ten** words. Both are
two-sided. A constant push has no notion of *enough*: it keeps pushing after the
constraint is satisfied and straight out the other side, which is why pushing
"more dialogue" takes the exactly-two rule from 4% to 0%. The three that work are
all monotone -- more present tense, simpler language, more varied openings are
never wrong.

**So constant steering is the right tool for monotone constraints and the wrong
tool for banded ones**, and that is a property of the mechanism rather than of
the vectors. It is also exactly what a feedback correction fixes: a push
proportional to the shortfall saturates when the shortfall reaches zero.

### What is running

* `assemble` -- budget steering plus the per-story perturbation, the first time
  the perturbation starts from a steering configuration that is winning rather
  than losing. Steering has 1.25 requirements of slack for it to spend.
* `feedback` -- steering that reads where the story already sits on each
  constraint axis and closes only its own shortfall. Verified on CPU: a
  compliant story receives nothing at all, and two stories failing different
  constraints are corrected in orthogonal directions, against cosine 1.0 between
  any two stories under the constant push.

## Rounds 9 to 11: why steering only works one way, and what counts

### The split that was hiding inside the aggregate

Forty stories a condition, thirteen requirements, Qwen3-1.7B. Separating the
requirements by *shape* rather than by name:

* **monotone** -- present tense, simple vocabulary, varied sentence openings.
  More of the property is always better.
* **banded** -- fifty to sixty-five words, six to eight sentences, every sentence
  four to ten words, exactly two quoted lines. Compliance is an interval.

| condition | broken /13 | loops | monotone | banded |
|---|---|---|---|---|
| baseline | 8.05 | 5% | **25%** | **23%** |
| constant steering, budget 3 | 7.50 | 43% | **60%** | 6% |
| feedback steering, gain 0.5 | 7.67 | **12%** | 57% | 6% |

Steering more than doubles compliance on the monotone requirements and cuts the
banded ones to a quarter. The two move in opposite directions and nearly cancel,
which is why eight rounds of aggregate scores showed a gain of half a requirement
and no sign that anything interesting was happening underneath.

A steering vector adds a fixed displacement whose first-order effect is a
monotone tilt: more of property P. For a monotone requirement more-P *is*
more-compliant, so a tilt is the right control signal. For a band, compliance is
an interval in P and a constant tilt cannot say *stop when you get there*.
Pushing "more dialogue" takes the exactly-two rule from 4% to 0% -- the mechanism
working exactly as designed and the requirement failing anyway.

**Feedback on activations does not fix it, and the reason is instructive.** The
correction closes the gap between where the story sits on a constraint axis and
where the positive contrast examples sit. Those examples are local text
containing quoted speech, so the correction saturates at "quote-like enough". It
never saturates at "I have written two". The direction encodes a local property;
the requirement is a global count. Measured: banded 6% under feedback, the same
as under the constant push.

### Feedback steering earns its place on a different axis

| method | broken /13 | loops | words |
|---|---|---|---|
| baseline | 8.05 | 5% | 55 |
| constant, one direction | 7.67 | **10/24 = 42%** | 82 |
| constant, five under a budget | 7.50 | 43% | 68 |
| **feedback** | 7.67 | **12%** | **50** |

Steering has been buying part of its compliance with broken text, and the twelve
requirements could not see it: within-story repetition was not among them, so
only the word cap caught any of it. Feedback loops three to four times less than
either constant arm at the same compliance, and is the only steered arm that does
not inflate length. That is the mechanism behaving as intended -- it stops
pushing a story that is already compliant, so it never drives one off a cliff.
`no_repetition` is now the thirteenth scored requirement.

### Diversity, established

Length matched at 55 words, degenerate stories filtered by the same rule in every
arm, equal group size, twelve paired random draws:

| condition | Vendi | beats baseline |
|---|---|---|
| baseline | 5.64 | |
| constant steering | 4.49 | 0/12 |
| feedback steering alone | 4.94 | 0/12 |
| **feedback + per-story perturbation 0.10** | **6.17** | **12/12 (+0.54)** |
| **feedback + per-story perturbation 0.15** | **6.60** | **12/12 (+0.96)** |

Neither component alone beats the baseline on diversity -- both lose, 0 of 12.
Only the combination wins, and it wins every draw. A first raw reading of this
was inflated by degeneration: the best-looking arm contained "I am going to the
store" eight times. The numbers above survive removing it.

### Closure: a brake, not a counter

A direction meaning "bring it to an end", on a schedule silent for eighty tokens
and rising after -- a legal story is about seventy-five tokens, so a compliant
one is never touched.

| condition | broken /13 | loops | banded | words | sentences |
|---|---|---|---|---|---|
| baseline | 8.05 | 5% | 23% | 55 | 11.3 |
| closure alone | 8.00 | 3% | 24% | 54 | 10.9 |
| five directions | 7.60 | **48%** | 7% | **73** | **41.9** |
| five + closure | **7.28** | **15%** | 6% | 48 | 20.8 |

The prediction was wrong: closure alone does nothing, and giving the *schedule* a
clock does not fix a counting requirement. What it does instead is undo the
damage steering causes. Five directions send the model to 73 words and 41.9
sentences against a six-to-eight rule, looping in 48% of stories; adding closure
brings that to 48 words, 20.8 sentences and 15%, and improves compliance further.
A principled brake on steering-induced degeneration, which is worth more than the
post-hoc n-gram filter it replaces.

The sharper statement about counting requirements is not that they cannot be
steered. Unsteered, the model already manages length 55% and sentence band 28%.
Steered, those fall to 25% and 0%. **Steering destroys counting requirements the
model could otherwise satisfy.** The exactly-two-quotes rule sits at 0% even
unsteered, carries no signal at all, and should be loosened or dropped.

### What the literature does, read critically

*Closing the Loop: PID Feedback Control for Activation Steering* (2606.18790)
drives its controller from "the mean magnitude of the top-N target features" --
an activation proxy for whether the steering survives sparsification. It
regulates the actuator, not the output. Both controlled properties, pitch and
duration, are monotone continuous quantities. Its headline is 72.65 semitones
against 72.30 static at n=40 with no error bars, which is not a demonstrated
effect; the credible result is efficiency, average lambda 1.15 against 3.0. It
concedes "sample sizes (n=40) are modest, and perceptual validation is absent".

In-Distribution Steering and DIRECTER adapt steering strength from model-internal
plausibility or token probability, not from a measured constraint error. Length
steering exists as a separate vector per length target, still open loop.

Nothing found closes the loop on a deterministic checker run over the partial
decoded text, which is what `noiseegra/constraint_control.py` does: each decode
step the story so far is decoded and the same checks that score the finished text
are run over it, giving a signed error per requirement. Inside its band a
requirement gets a coefficient of zero and the model writes unsteered. `--suite
control` is the test.

## Round 12 - each mechanism on the requirements it suits, and the dose confound

`--suite control`, Qwen3-1.7B, 40 stories per arm, thirteen mixed requirements.

| condition | broken/13 | vs baseline | looping | monotone | banded |
|---|---|---|---|---|---|
| constant push, coefficient 1, budget 2 | 7.53 | -0.53 | 5% | **64%** | 3% |
| baseline | 8.05 | - | 5% | 25% | 23% |
| error-driven, gain 0.5 | 8.32 | +0.27 | 12% | 28% | 14% |
| error-driven, gain 2 | 8.85 | +0.80 | 57% | 25% | 9% |

Two clean facts. A constant push takes the monotone requirements from 25% to 64%
and the banded ones from 23% to 3%. The error-driven controller does the reverse
in miniature: banded recovers to 14%, monotone falls back to 28%.

The controller's monotone collapse was a bug, not a finding. It watches three
requirements -- closure, terse, dialogue -- and scales each direction by its
measured error. The other three directions have no probe, so their error read as
zero and they were never pushed at all. The error-driven arm was silently
steering three directions instead of six.

## Round 13 - the hybrid, and one budget shared by two halves

Fix: a direction the controller watches is scaled by its error; one it has no
probe for keeps its constant coefficient. One push, each requirement controlled
the way its shape allows.

| condition | broken/13 | monotone | banded |
|---|---|---|---|
| constant push, coefficient 1 | 7.53 | 64% | 3% |
| baseline | 8.05 | 25% | 23% |
| hybrid, coefficient 2 | 8.10 | 35% | 12% |
| hybrid, coefficient 1 | 8.12 | 35% | 12% |

Both halves present, both weak: monotone 35% against the 64% a constant push
reaches alone. The cause was one ceiling covering the sum of the two halves. A
story forty words over its limit makes a large closure error, the combined vector
exceeds the cap, and the whole thing is scaled down -- constant components with
it. The dose on "more present tense" moved inversely with how badly the word
count happened to be doing.

## Round 14 - two budgets

The constant half is renormalised to its own fixed budget; the error-driven half
is capped at its own. Neither can rob the other. The probe also seeds a zero for
every requirement it watches before the first token, so the plan can tell a
watched direction from an unwatched one at step one.

| condition | broken/13 | vs baseline | looping | monotone | banded |
|---|---|---|---|---|---|
| **hybrid, coefficient 1** | **7.38** | **-0.68** | 32% | **66%** | 9% |
| constant push, coefficient 1 | 7.53 | -0.53 | 5% | 64% | 3% |
| hybrid, coefficient 2 + offset 0.15 | 7.60 | -0.45 | 30% | 68% | 6% |
| hybrid, coefficient 2 | 7.85 | -0.20 | 60% | 66% | 5% |
| baseline | 8.05 | - | 5% | 25% | 23% |

The fix worked as diagnosed: monotone 35% -> 66%, and the hybrid now beats the
constant push on both families at once (66% against 64%, 9% against 3%). Best
aggregate on the mixed set so far.

Two things it does not do. Banded is still far below the 23% the model reaches
unsteered -- **across all thirty-six steered conditions run to date, none has
improved a banded requirement over leaving the model alone.** And the hybrid buys
part of its win with degenerate text: 32% of its stories loop against 5% for the
constant push. That cost is already priced into the 7.38, because repetition is
one of the thirteen scored requirements, but it is the next thing to fix on this
line.

## Round 15 - a requirement set where steering and the requirement agree

A negative result about banded requirements is not a paper. The main comparison
moves to thirteen requirements that are all one-sided, so a push along a
direction and the requirement it serves agree about which way is better.

Seven are new or reshaped: every sentence at most eight words; at least three
lines of speech (not exactly two); at most two adverbs; at least two words saying
how something looks, sounds, feels, smells or tastes; at most one subordinate
clause; no word leaned on more than three times; a named character. Each is a
rule someone writing for children would actually be given rather than one
reverse-engineered from a steering direction.

Every threshold was set by measuring the candidate on forty stories written
*without* being asked for it and keeping the level that passed between 5% and
50%:

| requirement | unprompted | requirement | unprompted |
|---|---|---|---|
| present tense | 2% | at most 2 adverbs | 50% |
| grade 2.5 or easier | 72% | 2+ sensory words | 18% |
| no word over 2 syllables | 50% | at most 1 subordinate clause | 80% |
| first sentence at most 5 words | 38% | no word used over 3 times | 42% |
| every sentence at most 8 words | 65% | a name, used 3+ times | 48% |
| 3+ quoted lines | 48% | no repeated five-word run | 95% |
| no opener used over 3 times | 15% | | |

Nothing sits at 0% or 100%, so every requirement has room to be lost and room to
be won.

Four new directions come with it -- plain verbs over adverbs, sensory detail,
simple syntax, a named character -- twelve length-matched pairs each, each
checked to move its own metric and not the others:

| direction | positive side | negative side |
|---|---|---|
| plain verbs over adverbs | 0.25 adverbs | 2.33 |
| sensory detail | 4.00 sensory words | 0.08 |
| simple syntax | 0.00 subordinate clauses | 2.75 |
| a named character | 2.33 name uses | 0.08 |

Two conflicts inside the set had to be settled. Using the character's name three
times against using no word more than three times left only the single value
three, so names are exempt from the second. And spaCy parses `"Come here," Mira
says` as a complement clause, which made every line of dialogue count as
subordination and made the speech and syntax rules jointly unsatisfiable; quoted
speech is now removed before that count, and a test asserts a real story can
satisfy all thirteen at once.

## Round 15 - the monotone set, measured

Three runs: the decoding curve at 100 stories, each direction probed alone, and
the dose curve.

### The decoding curve, 100 stories

| condition | broken/13 | Vendi | looping |
|---|---|---|---|
| temperature 1.8, top-k 40 | 7.21 | 10.65 | 6% |
| temperature 1.3, top-p 0.95 | 7.68 | 7.96 | 28% |
| the model as it ships | 7.78 | 6.94 | 32% |

Two things came out of this. **The model as it ships loops on 32% of stories**
for this prompt, which inflates its broken count and holds its Vendi down. And
raising the temperature fixes the looping, so it beats the default on both axes
at once. The reference a method has to beat is therefore not the default.

Vendi here is untruncated and is *not* comparable with the round 16 numbers.

Also found: Qwen3 ships top_p in its generation config, so the top-p 0.95 arm at
temperature 1.0 came back identical to the baseline in every digit. Cut-offs are
only varied where the temperature is raised now.

### Each direction alone, pushed at 3, 30 stories

| direction | its requirement, unsteered | pushed | looping |
|---|---|---|---|
| present tense | 20% | 100% | 53% |
| dialogue | 3% | 100% | 93% |
| short sentences | 47% | 97% | 83% |
| a named character | 13% | 80% | 70% |
| simple register | 60% | 80% | 43% |
| sensory detail | 50% | 77% | 3% |
| simple syntax | 57% | 70% | 43% |
| varied openers | 27% | 40% | 3% |
| **plain words (adverbs)** | 63% | 67% | 73% |

Eight of nine control their own requirement. The adverb direction does not: it
moves its requirement by four points pushed one way, and its apparent 87% pushed
the other way is collapsed text -- *"The people did. The children did. The people
did."* -- at 70% looping. It is scored but no longer steered.

Strength 3 is a probe, not an operating point; most directions degenerate the
text there. The budget is what keeps the working dose below this.

### The dose curve, 100 stories

| condition | broken/13 | looping | words |
|---|---|---|---|
| the model as it ships | 7.78 | 32% | 85 |
| push held at 1.5 | 5.99 | 32% | 64 |
| push held at 2 | 5.55 | 24% | 56 |
| push held at 3 | 4.71 | 24% | 53 |
| **push held at 4.5** | **3.83** | **21%** | 52 |

On a requirement set where the directions and the requirements agree, steering
**more than halves** the broken count and *reduces* looping. This is the payoff
for moving off the mixed set.

Re-weighting the requirements per story at push 3 gave 5.33 against plain push
3's 4.71 -- worse, and with more looping. Consistent with round 2.

## Round 16 - the two halves together, and a negative result

100 stories per arm, eight directions, diversity on the first 40 words.

| condition | broken/13 | Vendi | looping |
|---|---|---|---|
| the model as it ships | 7.78 | 6.93 | 32% |
| constraint push at 3, alone | 3.94 | 5.67 | 11% |
| perturbation 0.1, alone | 7.08 | 9.94 | 20% |
| **push at 3 + perturbation 0.1** | **3.72** | **8.26** | **15%** |
| push at 3 + perturbation 0.15 | 3.84 | 9.56 | 19% |

**The first arm to beat the unmodified model on compliance, diversity and
degeneracy at once.** Each half alone wins one axis and loses the other; together
they win both, and the perturbation costs almost nothing in compliance on top of
the push (3.94 -> 3.72, i.e. it did not cost, it helped slightly).

### The negative result: choosing the perturbation set does not help

The idea: diversity is a property of the *set* of stories, and every round so far
drew each story's perturbation independently and hoped the set spread out. In the
low-rank subspace the perturbation is drawn from that hope fails -- sixty
independent draws from a rank-8 basis contain a pair 90% alike. So the whole set
was laid out in advance by repulsion, same length and same subspace, worst pair
0.90 -> 0.67 with mean similarity unchanged.

| pair | drawn independently | set chosen together |
|---|---|---|
| perturbation 0.1 alone | 9.94 | 9.68 |
| perturbation 0.15 alone | 12.71 | 11.94 |
| push 3 + perturbation 0.1 | 8.26 | 7.95 |
| push 3 + perturbation 0.15 | 9.56 | 8.81 |

**Four matched pairs, all four slightly worse.** Covering the offset subspace
evenly does not cover the output space evenly: distance in the subspace the
perturbation is drawn from does not predict distance between the stories that
come out. The mechanism stays in the code as `offset_draw="spread"` and is off.

## Round 17 - the headline run, and what filtering does to it

`r17-headline`: the decoding grid and the method's frontier in one command, so
every number shares one prompt (all fifteen requirements asked for), one scorer,
and one truncation (Vendi over the first 40 words). Qwen3-1.7B, layers 10-18,
100 stories per condition, eight directions. The push is the constraint sum held
at a total budget; the perturbation is the per-story offset at the prompt
positions only. (Correction added after r17-controls: the offset in this run drew
from an *isotropic* direction, not the story-difference basis its run ids claim.
The basis was never built -- see the measurement-fault note in RESEARCH_STATE and
round 17-controls below. The compliance column is unaffected; the diversity is
that of an isotropic perturbation.)

### The raw table, as the kernel scored it

| condition | broken/15 | Vendi@40 | words |
|---|---|---|---|
| the model as it ships | 8.43 | 6.64 | 83 |
| temperature 1.3, top-p 0.95 | 8.45 | 8.40 | 80 |
| temperature 1.6, top-p 0.95 | 8.18 | 9.20 | 81 |
| temperature 1.8, top-p 0.95 | 7.65 | 9.52 | 78 |
| temperature 1.8, top-k 40 | 8.03 | 10.69 | 80 |
| push 3 alone | 4.79 | 5.32 | 48 |
| push 4.5 alone | 4.25 | 6.29 | 50 |
| push 3 + perturbation 0.1 | 4.57 | 8.15 | 48 |
| push 3 + perturbation 0.15 | 4.94 | 9.73 | 58 |
| push 3 + perturbation 0.25 | 6.15 | 14.51 | 73 |
| push 4.5 + perturbation 0.1 | 4.51 | 7.79 | 58 |
| push 4.5 + perturbation 0.15 | 4.49 | 9.33 | 61 |
| push 4.5 + perturbation 0.25 | 5.01 | 12.74 | 62 |
| perturbation 0.1 alone | 7.88 | 10.47 | 82 |
| perturbation 0.15 alone | 8.22 | 12.67 | 83 |
| perturbation 0.25 alone | 8.99 | 18.29 | 89 |

Read raw, this looks like the answer: push 4.5 + perturbation 0.25 beats every
decoding arm on both axes at once, and push 3 + perturbation 0.25 does too.

### The raw winners are broken text

The same day's coherence checks (the four added after reading r16's stories:
windowed vocabulary entropy, near-duplicate sentences, tiny-sentence runs,
quote density, plus the existing heuristics) flag **every single story in every
push-4.5 arm** - 100 of 100, in all four. Reading confirms it; a typical
"winning" story is

    Mia runs. She runs. Mia jumps. Mia laughs. Mia! Mia. Mia.
    Lilly looks. Lily. Lily. Lily. "Look at the tree!" Call. "Tree." "Tree."

35-43 "sentences" in fifty-odd words. The 4.25-5.01 broken counts are collapse
satisfying one-sided requirements, and the Vendi is differently-broken stories
embedding far apart. Push 4.5 is dead at temperature 1.0, and the two arms the
raw table crowns are fake. The push-3 arms are half-broken (48-58 of 100
flagged); the decoding arms lose 19-32 of 100.

### Scored only over stories that pass the checks

`scripts/score_structure.py`, structural diversity = Vendi over each story's set
of content lemmas (names excluded - what happens in it), syntactic = Vendi over
part-of-speech trigram profiles (how its sentences are shaped); both over the
first 40 words of coherent stories, every condition rarefied to 30. Compliance
re-scored over the same kept stories.

| condition | kept/100 | broken/15, clean | structural | syntactic |
|---|---|---|---|---|
| the model as it ships | 72 | 7.93 | 16.4 | 5.5 |
| temperature 1.6, top-p 0.95 | 81 | 7.91 | 20.4 | 6.3 |
| temperature 1.8, top-p 0.95 | 78 | 7.38 | 21.5 | 6.8 |
| temperature 1.8, top-k 40 | 79 | 7.72 | 23.3 | 6.8 |
| push 3 alone | 51 | 4.86 | 13.9 | 6.1 |
| push 3 + perturbation 0.1 | 51 | 4.55 | 16.0 | 8.8 |
| push 3 + perturbation 0.15 | 48 | 4.31 | 16.8 | 8.1 |
| push 3 + perturbation 0.25 | 42 | 5.52 | 21.6 | 11.2 |
| perturbation 0.1 alone | 75 | 7.24 | 20.3 | 10.3 |
| perturbation 0.15 alone | 75 | 7.83 | 21.4 | 12.5 |
| perturbation 0.25 alone | 73 | 8.89 | 24.3 | 17.8 |

Four things this settles.

**The compliance win is real, not collapse.** Among stories that read fine, the
push-3 arms break 4.3-5.5 of fifteen against 7.4-7.9 for every decoding arm.
Filtering barely moves the pushed arms' broken counts (4.94 -> 4.31), so the
advantage was never carried by the broken half.

**The method does not beat tuned decoding on clean structural diversity.**
Temperature 1.8 reaches 21.5-23.3 effective structurally-distinct stories;
the best method arm reaches 21.6 (push 3 + perturbation 0.25) and the headline
arm 16.8. Raw Vendi said the method wins; the win was degenerate outliers plus
phrasing variety (syntactic 8-11 against the temperatures' 6.3-6.8).

**The perturbation works, and it is the push that degenerates.** The offset
alone lifts structural diversity 16.4 -> 20-24 at a keep rate no worse than the
temperatures'. The push alone drops it to 13.9 and halves the keep rate. Their
sum inherits both.

**Temperature buys structure too.** The claim from the diversity-injection
literature that temperature only buys token-level variety is not what this
measures: 16.4 -> 23.3 structural is a real gain in what happens in the
stories. What temperature does not buy is compliance (7.4-7.9 everywhere) or
syntactic variety (flat at 5.5-6.8).

### Verdict on the round's question, and what it opens

Against the unmodified default the method still wins both axes filtered or not
(4.31 broken against 7.93; embedding Vendi 9.73 against 6.64 unfiltered,
structural 16.8 against 16.4 - level - with syntactic 8.1 against 5.5). Against
the decoding curve it wins compliance by three requirements and loses clean
structural diversity. Neither dominates: the honest statement is a trade, plus
a cost column the tables did not have - the method keeps 42-51 coherent stories
of 100 where the temperatures keep 73-81.

The obvious untested cell: **the method at a working temperature.** Every push
arm above ran at temperature 1.0, the regime where this model loops on a third
of unmodified stories and the push turns prose to staccato. Temperature 1.6-1.8
is known to stop the looping (r15, and the keep rates above); the push is known
to restore the compliance temperature spends. Each half fixes the other's
failure mode, and no run has ever combined them.

## Round 17-controls - which parts of the perturbation earn their place, and the basis bug that made the question answerable

Nine conditions on the second account, added to test three surgeries on the
perturbation: the constraint-span projection removed, the sampled story-difference
basis replaced by an isotropic draw, and the offset kept on while the model
writes. Adding the suite to the runner's basis-construction guard set is what
made this the first run since r15 to actually build the story-difference basis --
and comparing it against r17-headline is what exposed that r16-spread and
r17-headline never built one at all (the `spread` and `frontier` suites were
missing from the guard, so their "story-basis" offsets were isotropic noise with
a mislabelled run id). Details and the fix are in RESEARCH_STATE's measurement-
fault section; `tests/test_suites_build.py` now catches a fourth affected suite,
`select`, before it can run.

Scored over coherent stories only (`scripts/score_structure.py`), gamma 0.15,
rarefied to 40 per condition, 40 draws:

| arm | kept/100 | broken/15 | structural | syntactic |
|---|---|---|---|---|
| story-difference basis, alone | 70 | 8.25 | 27.70 | 12.62 |
| isotropic draw, alone | 75 | 8.22 | 25.93 | 14.08 |
| projection removed, alone | 78 | 8.57 | 27.07 | 14.00 |
| story-difference basis + push 3 | 52 | 4.84 | 20.41 | 9.92 |
| isotropic draw + push 3 | 48 | 4.94 | 20.34 | 8.90 |
| projection removed + push 3 | 48 | 5.32 | 22.37 | 9.76 |

**The sampled basis does not beat an isotropic draw.** +1.8 structural alone at
a worse keep rate and lower syntactic diversity, and dead level under the push.
The between-generation basis is the part of the method that is not already
published; on the first run that genuinely uses it, plain noise of the same
length does as well.

**The projection buys compliance, not diversity.** Removing it holds or raises
diversity (22.37 against 20.41 structural under the push) while compliance falls
0.5 of a requirement (4.84 to 5.32). It protects what it was built to protect and
is not charging diversity for it.

The offset-kept-on-while-writing arm (13.81 raw Vendi, 5.83 broken, 67% of
stories flagged by the coherence checks) is the worst of the three sites, as
prior rounds found; it is not worth a filtered breakdown.

## Round 18 - the method at a higher randomness setting (temperature 1.6 part)

### What this run was

The method has two parts, applied while the model writes a story: a *rule-nudge*
that pushes the model's internal state toward obeying the 15 writing rules (this
improves rule-following), and a *per-story random shove* that starts each story
in a slightly different place (this adds variety). Every earlier test of the
method ran with the model's randomness dial (the sampling temperature) at its
normal setting of 1.0, where this model tends to loop and repeat on its own. This
run raised the dial to 1.6 to see whether the method works better when the model
is not already looping. The comparison, all in one run of 100 stories per
setting: the randomness dial alone at several settings (the free alternative to
the method), and the method at rule-nudge strengths 3 and 4.5 combined with a
random shove of size 0.1 and 0.15. The stories were then filtered to keep only
the coherent ones, and variety was measured two ways: *variety of what happens*
(are the events and things mentioned genuinely different across stories?) and
*variety of wording* (are the sentences phrased and shaped differently?). Both
are on a scale where the number is roughly how many completely-unrelated stories
the batch is worth.

### The randomness setting stopped the looping but the rule-nudge still breaks the text

At the raised randomness setting the exact-repetition rate of the steered stories
was tiny (1-5%, against 29% for the untouched model), which looked like the
rule-nudge no longer degrades the text. It does. Reading the stories, the
rule-nudge collapses them into disconnected two-word fragments instead -- "Tom
Yaks. Cat Dances. Tom Yaks. Cat Dancer." -- with broken and non-English words. The
fragments vary slightly, so the old repeated-phrase check missed them (it reports
2% broken) while the new checks catch them (about 63% broken). Without the new
degeneration checks this run would have looked like a clean win and been wrong.

### The decisive comparison, over coherent stories only

| setting | coherent stories kept /100 | rules broken /15 | variety of what happens | variety of wording |
|---|---|---|---|---|
| method: rule-nudge 3 + random shove 0.15, randomness 1.6 | 50 | 4.48 | 27.2 ±0.5 | 11.1 ±0.3 |
| free: randomness 1.8, nucleus cutoff 0.95 | 78 | 7.38 | 26.4 ±0.8 | 7.6 ±0.7 |
| free: randomness 1.8, top-40 cutoff | 79 | 7.72 | 29.3 ±0.6 | 7.6 ±0.6 |
| the untouched model | 72 | 7.93 | 20.0 | 6.0 |

(The two "free" rows are two ways of trimming the least-likely words while
sampling; both are the plain randomness-dial alternative to the method.)

Reading this honestly:

- **Rule-following: the method wins clearly.** Among coherent stories it breaks
  4.48 of the 15 rules against about 7.4-7.9 for every randomness-dial setting --
  roughly three fewer rules broken.
- **Variety of wording: the method wins clearly.** 11.1 against 7.6.
- **Variety of what happens: the method is competitive but not ahead of the best
  free option.** It beats one randomness-dial setting (27.2 against 26.4) and
  loses to the best (29.3 for the top-40 cutoff at randomness 1.8). So on genuine
  story-to-story difference the method is in the same range as tuned randomness,
  not better than it.
- **The method wastes more stories.** It keeps 50 coherent stories of 100 against
  about 78 for the randomness dial, so in practice you would generate more to get
  the same number of usable ones.

Compared with the same method at the normal randomness setting of 1.0 (from the
previous run), raising the dial to 1.6 helped: at 1.0 the method clearly lost on
variety of what happens, and at 1.6 it draws level with tuned randomness. But the
strongest rule-nudge (strength 4.5) is dead even at the raised setting -- 0 to 2
coherent stories of 100 -- so it is too strong at any randomness.

### Still pending

The randomness dial at 1.8 with the method (a separate run on the second account)
is the more promising setting still: more randomness should mean less
fragmentation from the rule-nudge, and possibly more coherent stories kept.
Results when it lands.
