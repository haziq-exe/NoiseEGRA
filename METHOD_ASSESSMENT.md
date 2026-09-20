# Every mechanism tried, what it bought, and whether it is publishable

One assessment per mechanism rather than per run. Numbers are the honest ones:
non-stories counted as broken stories, preambles and headings counted as broken
requirements, variety with 95% intervals from subsampling without replacement,
200 stories a condition against baselines also at 200.

Three axes are scored for each:

**Result** — what it actually bought, in numbers.

**Novelty** — is the mechanism new, or a renaming of something published? Judged
against what a reviewer would know: contrastive activation addition (Rimsky et
al., ACL 2024), representation engineering (Zou et al., 2023), and the decoding
literature (Fan et al. ACL 2018; Holtzman et al. ICLR 2020; Meister et al. TACL
2023; Su et al. NeurIPS 2022; Hewitt et al. EMNLP Findings 2022).

**Generality** — would this mechanism mean anything on another task, or is it a
knob fitted to *this* instruction with *this* model? The harshest question in
the list, and the one that sinks several otherwise interesting results.

---

## Tier 1 — publishable as a contribution in its own right

### 1. The diversity measure is fooled by fluent non-stories

**Result.** An arm scored 200 of 200 coherent and beat both raised-temperature
baselines on every variety measure. Reading its stories showed a quarter were
not stories: numbered lists of drawing tips, chat with the reader, descriptions
with no character and nothing happening. Every check passed them — they are
fluent, varied, not repetitive, not degenerate. Rescored, it was 173 of 200 and
its variety wins were ties. The untouched model and both baselines produce **0%**
of this; arms that displace the hidden state produce up to 24%.

**Novelty.** High, and not about our method. The standard coherence filters in
this literature ask whether text is *degenerate*. None asks whether it is the
thing that was requested. And the failure does not merely evade the filter — it
**inflates the diversity score**, because a list of drawing tips is enormously
unlike a story about a girl at a bus stop, and a set-diversity measure rewards
exactly that. Any paper reporting Vendi or self-BLEU gains from an intervention
that perturbs the model is exposed to this.

**Generality.** Very high. It is a property of measuring open-ended generation
diversity, not of this task. The detector needs care — a story may be told in
the second person, and three of the first eight that a bare "you" rule flagged
were stories — but the hazard is universal.

**Publishability: 1st.** This is the strongest thing produced. It is a negative
result about measurement that applies to a whole class of papers, it is cheap to
demonstrate, and it comes with a worked example where it changed a headline
result. Would make a solid short paper on its own, or a prominently placed
section here.

---

### 2. Requirement compliance from constraint steering

**Result.** 2.75 to 3.06 requirements broken of twelve against 3.92 for
raised-temperature nucleus sampling, 3.95 for top-k and 4.30 for the untouched
model — intervals well clear of zero, at 190–193 of 200 coherent. Won at every
displacement setting tried. Survived all four measurement corrections, because
the failures those corrections exposed were being scored against the
requirements too.

**Novelty.** Low to moderate. The push is contrastive activation addition
(Rimsky et al., ACL 2024) applied to writing requirements rather than to
behavioural traits. What is ours: Löwdin-orthogonalising the direction set and
holding a **fixed total budget**, so "steer one more requirement" does not
silently mean "push harder" — which we measured mattering, twice.

**Generality.** High. Nothing in it is fitted to this instruction. The
directions are extracted from contrast pairs in the task's own context; the same
recipe would transfer to any set of checkable writing requirements.

**Publishability: 2nd.** The result is large, repeatable, and has survived more
adversarial rescoring than most published numbers. The mechanism is not new, so
it is a solid empirical contribution rather than a methodological one.

---

### 3. Where in the prompt the perturbation sits decides *which* diversity you get

**Result.** Holding the displacement fixed at two stories out and varying only
how much of the prompt tail is spared:

| positions spared | coherent | not a story | what happens | wording | commonest opener |
|---|---|---|---|---|---|
| 8 | 170/200 | 10% | **+4.8** | **+5.2** | 35% |
| 16 | 177/200 | 3% | +1.9 tie | +0.3 tie | 44% |
| 32 | 183/200 | **0%** | +1.5 tie | −0.3 tie | 45% |
| faded to 0.15 | **193/200** | **0%** | tie | loss | 38% |

The perturbation near the end of the prompt is what varies the **wording** and
what makes the model answer the reader instead of telling a story. The
perturbation early in the prompt is what varies **what happens** and is safe: at
2.5 stories out with 48 spared there is not one non-story in 200 and what
happens is still won, while wording becomes a loss.

**Novelty.** Moderate to high. Prompt-position-dependent steering is not itself
new, but the finding that *different positions control different diversity
axes*, and that one of them carries the failure mode, is a specific empirical
result I have not seen stated.

**Generality.** Moderate. The mechanism is general; the specific positions are
not, and a different chat template would move them. Stated as "late-prompt
perturbation varies surface form and risks register collapse; early-prompt
perturbation varies content" it should transfer.

**Publishability: 3rd.** A genuinely interesting analysis result, and the kind
of thing that makes a method paper look like it understands its own mechanism.

---

## Tier 2 — worth a paragraph, not a section

### 4. The story-register shield

**Result.** A contrast direction separating telling a story from talking about
the task, extracted like any other and then never pushed — only removed from the
subspace the perturbation may move along. Took refusals and leaked planning from
9 and 9 to **0 and 0** at 1.5 stories out, at 99–100 of 100 coherent, while the
constraint push stays byte-identical.

**Novelty.** Moderate. Orthogonalising noise against constraint directions is in
the original workshop paper. Extracting a direction *purely to forbid it*, for a
failure mode rather than a requirement, is a small but real extension.

**Generality.** Moderate. The idea — name the failure's direction and remove it
from whatever you perturb along — is general. But it **does not generalise even
within our own task**: a purpose-built direction for assistant helpfulness moved
that failure from 23% to 20%, i.e. nothing. So the honest claim is narrow:
refusal is a direction, helpfulness is not.

**Publishability: 4th.** A clean mechanism with a clean result and an honest
limit. The limit is arguably the more interesting half.

### 5. Measuring displacement in units of the data

**Result.** The perturbation size was a fraction of the hidden state's own norm,
which has no relation to how far the model's stories sit from one another. A
real story sits 6.9 from their average at layer 6; the size every good
configuration used works out at 15.0 — more than twice as far out as any story
the model wrote. Rescaling to "one story's distance" made the whole sweep
interpretable and opened a range nothing had been run in.

**Novelty.** Low as an idea, high as a correction. Nobody would defend the old
unit once it is stated.

**Generality.** High, and free.

**Publishability: 5th.** A methods footnote, not a contribution. But it should
be *in* the paper, because a reviewer asking "what does γ=0.15 mean?" has a good
question and this is the answer.

### 6. Steering an event, not just a style

**Result.** The commonest content words in our stories were *scent* 29%, *air*
32%, *sun* 30%, where the baseline's include *backpack*, *window*, *stand*,
*look*. The push asked for present tense, sensory detail, a named character and
format — **nothing asked for anything to happen**. Adding a direction contrasting
an event with a description moved the commonest words to *run*, *grab*, *hit*
and won variety of what happens at +4.9 [+2.5, +7.5] where four rounds of
perturbation tuning had only tied.

**Novelty.** Low as a mechanism (it is another CAA direction). Moderate as a
diagnosis: *the steered requirement set determines the ceiling on content
diversity, and a set of purely stylistic requirements caps it.*

**Generality.** The diagnosis is general and rather useful. The specific
direction is task-shaped.

**Publishability: 6th.** Worth reporting as analysis — it explains why
perturbation tuning was hitting a ceiling — rather than as a method.

---

## Tier 3 — record as negative results, do not lead with them

### 7. Per-story gain on a steered direction (f(S_c))

Varying how hard one direction is pushed, drawn afresh per story, lognormal with
mean one. **Completely inert**: 183/200 at both spreads, identical compliance,
identical variety, and doubling the spread changed nothing. Tried twice — once
when every direction was stylistic, once with a direction that demonstrably
controls content. The second run killed the explanation offered for the first.
**Generality:** the negative result is general and worth one sentence.

### 8. Aiming the displacement at one of the model's own stories

Appeared to be the breakthrough: +3.5 on what happens, +9.0 on wording. Then the
non-story check showed 23–25% of its output was not stories, against 0% for
drawn perturbation at the same distance. **The win and the failure were the same
thing.** Honest rescoring leaves it a tie. **Novelty** would have been high;
**publishability** is now as a cautionary example inside contribution 1.

### 9. Calibrating which sampled stories are dangerous, and shortening toward them

Eight of 32 sampled stories caused 62 of 76 rejections across six runs; a
hundred-story probe finds them without being told. Removing them reached the
baselines' coherence and lost the variety; shortening instead kept both, which
was the first move *off* the frontier. But it only exists because of mechanism 8,
which is itself unsound, and it is **heavily fitted to this task** — 32 stories
from one instruction on one model. **Publishability: low.** Elegant, and I would
not put it in front of a reviewer.

### 10. Everything that was simply a null

Wider shields; a second shield for helpfulness; register-matched contrast pairs;
richer perturbation bases; a larger total push (3 changes nothing, 4 is much
worse); repulsive/spread draws; entropy gating; per-token noise (best compliance
of anything, 1.67, and *negative* variety); the wandering aim; per-story rotation;
the opening-decode window; weighting the format direction; quietening the sensory
direction; and pushing the story-register direction (which collapsed coherence to
109/200). **One sentence each in a limitations table.**

---

## What I would actually submit

**Title claim:** constraint steering makes a small model follow a dozen writing
requirements markedly better than raised-temperature decoding, at the same
coherence — and the diversity gains that representation perturbation appears to
buy are largely an artefact of the measurement.

That is two real results, one positive and one negative, both robust. It is
honest about the frontier rather than hiding it, and the measurement finding is
the kind of thing reviewers remember.

**What is missing before submission:** comparisons against published decoding
methods beyond nucleus and top-k. Typical sampling (Meister et al., TACL 2023),
η-sampling (Hewitt et al., EMNLP Findings 2022), contrastive search (Su et al.,
NeurIPS 2022) and min-p (Nguyen et al., ICLR 2025) are all standard, all
supported natively by the generation stack, and a reviewer will ask.
