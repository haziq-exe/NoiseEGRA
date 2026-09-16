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
