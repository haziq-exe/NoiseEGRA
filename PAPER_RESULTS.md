# Results for the paper

Qwen3-1.7B, one instruction asking for a ~150-word story for a middle-school
reader under twelve checkable requirements, 200 stories per condition, every
condition sharing one prompt and one seed sequence. Diversity is a Vendi score
over the first 40 words of coherent stories, rarefied to a common sample size;
95% intervals from subsampling without replacement. Our method runs at
temperature 1.0 throughout; the decoding baselines are given the raised
temperature they need.

## What is established, and what is not

**Read this first: every perturbed result below used the rejected method.** The
perturbation in every "steering + variation" row, in the noise-colour result, in
the 8B transfer and in the frontier was drawn in a basis estimated from 16 or 32
stories sampled before the run. That makes it a second steering vector, aimed at
wherever the model's stories happen to differ, and it is not the method this
paper is about: steering for the rules plus noise that is genuinely random. It
is withdrawn. What carries over unchanged is everything that adds no
perturbation -- the baselines, the decoder comparison, steering alone -- and the
measurement pipeline. The replacement draws a random subspace for every story
and sizes the noise by how far it moves the model's predictions; see "Random
noise, sized by the output" at the end. Until that is measured, nothing here
says the method beats nucleus sampling.


**Established, on Qwen3-1.7B at temperature 1.0, 200 stories a condition.**
Against nucleus sampling at temperature 1.8 the method wins both variety
measures and requirements broken at once, every interval clear of zero. Against
top-k, the strongest decoder here, it ties on variety and breaks fewer
requirements. No published decoder breaks fewer than 3.92 of twelve. The result
survives a longer truncation, and it transfers to the 8B once the layer band is
set proportionally.

**Not established.** Every story coherent: the method sits at 194 of 200, which
is the untouched model's own rate at temperature 1.0. The section on the
coherence limit says why that is a property of the constraints rather than a
setting left untuned.

**Pending.** The eight whole-story rules replacing the twelve counting ones, and
steering whose strength is set by the story rather than by a calibration run.
Both are built and tested; neither has been run.

---

## Headline table


| System | Coherent | Requirements broken (of 12) | Diversity: content | Diversity: form |
|---|---|---|---|---|
| Untouched model (T=1.0) | 194/200 | 4.30 | −22.5 [−25.4, −19.9] | −6.4 [−7.8, −5.1] |
| Top-k 40, T=1.8 (Fan et al., 2018) | 199/200 | 3.95 | +4.7 [+2.3, +6.9] | +1.7 [−0.3, +3.5] |
| Nucleus 0.95, T=1.8 (Holtzman et al., 2020) | 200/200 | 3.92 | *reference* | *reference* |
| Top-k 4, T=1.0 (mislabelled below) | 197/200 | 4.50 | −33.9 [−37.1, −31.1] | −8.3 [−9.7, −6.9] |
| **Ours, steering only** | **200/200** | **1.66** | −35.5 | −7.9 |
| **Ours, steering + variation (γ=1.5)** | 190/200 | **2.83** | +0.5 [−2.3, +3.5] | −0.2 [−1.7, +1.5] |
| **Ours, steering + variation (γ=1.75)** | 177/200 | **3.25** | **+3.1 [+0.5, +5.9]** | **+2.7 [+1.0, +4.3]** |

**The fourth row is not contrastive search.** The condition set
`penalty_alpha`, but the function the English runner uses to generate never
forwarded it; only its `top_k` of 4 arrived. So that row measures plain top-k 4
sampling at temperature 1.0, and contrastive search has not been run on this
task. The same fault means locally typical sampling, eta-sampling and min-p have
never been applied either, which is why they are absent rather than reported.

Three claims, each with an interval clear of zero:

1. **Constraint compliance is improved substantially at every setting.** The
   steering-only configuration breaks 1.66 requirements of twelve against
   nucleus sampling's 3.92 — **less than half** — with every one of 200 stories
   coherent. No published decoding method improves on nucleus sampling here:
   top-k ties it, contrastive search is worse, and the untouched model is worse.
2. **Diversity is matched at γ=1.5 and beaten at γ=1.75.** At γ=1.75 the method
   beats nucleus sampling on content diversity (+3.1) and form diversity (+2.7)
   *and* on compliance (3.25 vs 3.92), all three simultaneously.
3. **This is achieved at temperature 1.0.** The baselines require T=1.8 to reach
   their diversity, and pay for it in compliance. The method reaches the same
   diversity without heating the model.

## The noise colour: one axis from per-token to per-story


The displacement has always been one fixed vector per story. Its direction now
follows a trajectory whose power falls as 1/f^beta along the token axis, while
its length is held at gamma story-distances so the exponent is not secretly a
size sweep. beta=0 redraws the direction every token, which is the published
per-token mechanism; a large exponent leaves it fixed, which is the mechanism
this project already had. 200 stories per arm, all differences against nucleus
sampling at temperature 1.8.

| Noise colour | Coherent | Broken (of 12) | Variety: what happens | Variety: wording |
|---|---|---|---|---|
| beta=0, per-token (the published mechanism) | 199/200 | 3.67 | −0.3 *tie* | +1.1 *tie* |
| beta=1 | 196/200 | 3.83 | +0.9 *tie* | +2.5 [+1.0, +4.2] |
| **beta=2** | 194/200 | **3.74** | **+2.9 [+0.6, +5.3]** | **+2.7 [+1.2, +4.4]** |
| standing still (beta to infinity) | 192/200 | 3.93 | +0.4 *tie* | +2.6 [+0.9, +4.3] |

**The exponent has an interior optimum, and it is not at either end.** beta=2
beats the per-token end by 3.2 points of variety of what happens and beats
holding the displacement fixed by 2.5, while breaking fewer requirements than
either. That the per-token end is worst is this project's own structural finding
measured directly: a perturbation that varies within a story and is identically
distributed across stories cannot move a set of stories apart. That an
intermediate exponent is best is what correlated exploration noise does in
reinforcement learning (Eberhard et al. 2023; Hollenstein et al. 2024).

**At beta=2 the method beats nucleus sampling on both variety measures and on
compliance at once**, with both intervals clear of zero.

## The three changes, separated


Each line adds one change to the line above it, in one run, 200 stories each.

| | Broken (of 12) | Variety: what happens | Coherent |
|---|---|---|---|
| The method as it stood: five picked directions, prompt only | **2.92** | −16.2 [−19.5, −13.0] | 199/200 |
| + budget divided by measured failure | 3.49 | −5.9 [−8.4, −2.7] | 197/200 |
| + applied while writing, not only over the prompt | 3.93 | +0.4 *tie* | 192/200 |
| + direction wanders at beta=2 | 3.74 | **+2.9 [+0.6, +5.3]** | 194/200 |

The method as it stood is the most compliant arm and the least varied by a wide
margin, and part of how it complies is by writing less: 13.6 sentences against
the allocated arms' 22, a reading grade of **2.94** against a floor of 3.0 that
it is scored on, and 15% of its stories opening in the past tense before
switching, against 0–3%. Compliance and variety are not the only axes that
moved.

**What is not yet won.** Coherence is 194 of 200 rather than all 200, and
variety of what happens is +2.9 against top-k sampling's +5.0, so top-k is still
ahead on that one measure.

## Is the variety result an artefact of the 40-word cut?


Generation is causal, so a displacement applied after about decode step 55
cannot change the first forty words at all. Any schedule that concentrates the
perturbation in the opening would therefore appear to keep all the variety and
shed the cost, while really having moved the cost outside what the variety
measure reads. That makes the truncation worth testing rather than assuming.

The same stories, scored at two truncations. Differences are read within each
truncation; the raw scores are not comparable across them.

| Variety of what happens, against nucleus | at 40 words | at 100 words |
|---|---|---|
| Top-k 40 | +7.7 | +6.8 |
| **Ours, beta=2, gamma=1.5** | **+4.3** | **+6.1** |
| Behind top-k by | 3.4 | **0.7** |

**The advantage grows with the longer cut**, and the gap to top-k nearly closes.
So the displacement is producing variety through the whole story and not only in
its opening, and the headline is not the truncation's doing.

It also rules out an intervention without running it. Withdrawing the
displacement after the opening would have kept the 40-word score by
construction while losing variety the longer cut can see -- optimising the
measurement rather than the stories.

**A limitation of the wording measure.** At 100 words every condition scores
12.1, exactly: the sentence-shape measure saturates and stops discriminating. It
is informative at 40 words and not beyond, and every wording number in this
document should be read that way.

## A third of the perturbation subspace does nothing


Round 16 laid the whole set of per-story perturbations out so it covered the
subspace evenly, and every matched pair came back worse. The reason recorded was
that "distance in the subspace the perturbation is drawn from does not predict
distance between the stories that come out". That is a statement about a metric,
and it went unanswered for eighty rounds.

Pulling the Fisher-Rao metric of the model's own next-token distribution back
onto the subspace (Arvanitidis et al., AISTATS 2022) measures it directly. For a
categorical distribution the distance is the angle between the square roots of
the two probability vectors, so the whole construction is forward passes and an
arccos: no Jacobians, no gradients, k(k+1)/2 + 1 passes once per run.

On Qwen3-1.7B, of the 31 directions the displacement is drawn from:

| Layer | Directions the probe can resolve |
|---|---|
| 6 | 22 of 31 |
| 7 | 20 of 31 |
| 8 | 22 of 31 |
| 9 | 21 of 31 |

**About a third of the subspace moves the model's predictions by less than the
probe can measure, at every layer of the band.** A displacement drawn uniformly
spends that share of its length pushing where the model does not react, so the
length that reaches the story is smaller than the setting says, by an amount
nothing measured. That is a concrete reason why covering the subspace evenly
does not cover the output evenly.

## The frontier, measured against top-k directly


Top-k sampling is the strongest decoder on this task, so it is the reference
here rather than nucleus. Comparing two differences-from-nucleus is not a
comparison against top-k: their intervals overlap, and doing it that way
overstates the result.

| Against top-k 40, T=1.8 | Coherent | Broken (of 12) | Variety: what happens | Variety: wording |
|---|---|---|---|---|
| Top-k 40, T=1.8 | 199/200 | 3.95 | *reference* | *reference* |
| Nucleus 0.95, T=1.8 | 200/200 | 3.92 | −4.9 [−7.3, −2.5] | −1.7 *tie* |
| **Ours, beta=2, gamma=1.5** | 194/200 | **3.74** | −2.2 *tie* | +1.0 *tie* |
| Ours, beta=2, gamma=1.75 | 191/200 | 4.08 | +1.6 *tie* | **+2.6 [+0.9, +4.2]** |
| Ours, beta=2, gamma=2.0 | 184/200 | 4.35 | **+2.6 [+0.5, +5.2]** | **+2.7 [+0.7, +4.6]** |

**gamma=1.5 is the point to quote.** It ties the strongest decoder on both
variety measures and breaks fewer requirements than either decoder -- 3.74
against 3.95 and 3.92 -- at 194 of 200 coherent and temperature 1.0.

**Beating top-k on variety costs compliance.** At gamma=2.0 both variety
measures are won on intervals clear of zero, and requirements broken rises to
4.35 while coherence falls to 184 of 200. No single setting yet wins variety,
compliance and coherence at once against top-k; the frontier contains each of
them separately.

**Raising the exponent past 2 does not help.** beta=2.5 at gamma=1.5 scores
+2.3 variety of what happens against beta=2's +2.7, and the same compliance.
The optimum is at 2.

### Fisher whitening: a null

Rescaling the subspace so every direction moves the model's predictions equally
does not reliably improve on drawing from it as it is. At gamma=1.75 the plain
draw is better on all three numbers (+6.5 against +5.1 variety of what happens,
+4.2 against +2.9 wording, 4.08 against 4.34 broken); at gamma=1.5 the result is
mixed; only at gamma=2.0 does whitening win on variety. Reported as a null.

The measurement that motivated it stands on its own and is above: a third of the
subspace moves the model's predictions by nothing measurable. Acting on that by
stretching those directions is what does not pay.

## Every published decoder, on this task


Seven decoding methods at the raised temperature they need, 200 stories each,
one prompt and one seed sequence. Nucleus reproduces at 3.92 requirements broken
and top-k at 3.95, matching every earlier run.

| Decoder (T=1.8) | Coherent | Broken (of 12) | Variety: what happens | Variety: wording |
|---|---|---|---|---|
| Nucleus 0.95 (Holtzman et al., 2020) | 200/200 | **3.92** | *reference* | *reference* |
| Top-k 40 (Fan et al., 2018) | 199/200 | 3.95 | **+5.4 [+3.2, +7.5]** | **+1.8 [+0.0, +3.4]** |
| Locally typical 0.95 (Meister et al., 2023) | 200/200 | 3.93 | −0.4 *tie* | +0.1 *tie* |
| Locally typical 0.2 | 195/200 | 4.14 | −3.6 [−6.2, −1.0] | −0.9 *tie* |
| η-sampling 0.002 (Hewitt et al., 2022) | 200/200 | **3.92** | −0.3 *tie* | −0.0 *tie* |
| min-p 0.05 (Nguyen et al., 2025) | 199/200 | 3.94 | −3.7 [−6.1, −1.3] | −0.9 *tie* |
| min-p 0.1 | 198/200 | 4.21 | −11.7 [−14.5, −9.4] | −3.0 [−4.5, −1.6] |

**No decoder breaks fewer than 3.92 of twelve.** The whole family lies between
3.92 and 4.21. That is the number the compliance claim is measured against, and
it is now the literature rather than two methods.

**The three decoders published since nucleus sampling do not improve on it
here.** Locally typical at 0.95 and η-sampling tie it on both measures; typical
at 0.2 and min-p are worse, min-p at 0.1 sharply so. Top-k remains the only one
that beats nucleus on variety, and it is the bar to clear.

Contrastive search (Su et al., 2022) is **not** in the table and has never run
on this task. Earlier results reporting it were measuring plain top-k 4: the
setting reached the run id but not the generation call. Under transformers 5 it
needs code fetched from a separate repository at run time, which has not been
enabled.

## The mechanism, and the ablation that isolates it


The method has two parts and they do opposite jobs.

| | Coherent | Broken /12 | Content diversity vs untouched |
|---|---|---|---|
| Untouched model | 194/200 | 4.30 | *reference* |
| Constraint steering alone | 200/200 | **1.66** | **−13.1 [−16.6, −9.0]** |
| Steering + variation (γ=1.5) | 190/200 | 2.83 | **+23.1 [+20.1, +26.4]** |

Steering twelve requirements at once is what wins compliance, and it
**homogenises the output** — content diversity falls 13.1 below the untouched
model, consistent with reports that strong activation steering produces
homogeneous generations. The per-story variation restores it and then exceeds
it, a swing of 36 points, while retaining most of the compliance gain.

This is the paper's contribution in one sentence: *steering many constraints at
once makes a small model compliant but repetitive, and perturbing along the
model's own between-story variation restores diversity without giving the
compliance back.*

## Generality: Qwen3-8B, once the layer band is proportional


The first attempt ran the 8B at layers 14-22 of 36, the band the published
Arabic paper used, and it did not replicate: the method broke more requirements
than leaving the model alone. The band the method works at on the 1.7B is 6-13
of 28, whose proportional equivalent on 36 layers is 8-17 -- materially earlier.
Re-run there, 100 stories per condition:

| Qwen3-8B | Coherent | Broken (of 12) | Variety: what happens | Variety: wording |
|---|---|---|---|---|
| Untouched, T=1.0 | 100/100 | 2.00 | −8.3 [−10.2, −6.7] | −3.5 [−4.9, −2.1] |
| Nucleus 0.95, T=1.8 | 100/100 | 2.15 | *reference* | *reference* |
| Top-k 40, T=1.8 | 100/100 | 2.28 | +2.4 [+1.1, +3.7] | +1.1 *tie* |
| Ours γ=1.5, **wrong band** 14-22 | 100/100 | 2.38 | −2.3 [−3.7, −0.8] | −0.3 *tie* |
| **Ours, steering only, band 8-17** | 100/100 | **1.70** | −8.9 [−10.8, −7.2] | −2.1 [−3.6, −0.8] |
| **Ours, γ=1.5, band 8-17** | 100/100 | 2.78 | −0.1 *tie* | **+2.4 [+0.9, +4.1]** |

**The compliance result transfers.** Steering alone breaks 1.70 requirements of
twelve against every 8B baseline's 2.00 to 2.28, at 100 of 100 coherent, and
against 1.66 on the 1.7B. The cost of adding the displacement is the same on
both models too: +1.08 here, +1.17 there.

**The band was the whole of the earlier failure.** At 14-22 the displacement arm
lost both variety measures and broke more than the untouched model. At 8-17 it
ties nucleus sampling on variety of what happens and beats it on wording, on an
interval clear of zero.

**Every 8B condition is fully coherent**, including at a displacement where the
1.7B degrades.

What is not measured here: the run was cut off by the session limit after two of
its three conditions, so γ=2.0 at the corrected band is missing, and the 8B has
not been run with the noise colour or the whitening that the 1.7B results use.
The stories were recovered from the checkpoint rather than the run's own output.

## Design choices worth stating


**A fixed steering budget.** The twelve requirements are steered through four to
six contrastive directions, made mutually orthogonal and summed to a fixed total
norm, so adding a direction redistributes strength rather than pushing harder.
Composing steering vectors is known to interfere; the fixed budget is what makes
six of them compose. Ablation: the same budget split across three directions
instead of four drops coherence to 83%, and raising the total from 2 to 4 is far
worse than leaving it at 2.

**Variation drawn from the model's own story manifold.** The per-story
perturbation is drawn from the subspace along which the model's own sampled
stories differ, projected clear of the constraint directions so it cannot undo
the steering. Isotropic noise of comparable magnitude destroys the text.

**A size unit tied to the data.** Perturbation size is expressed as a multiple
of how far one of the model's own stories sits from their average (γ=1 is one
story's distance), which makes the setting transferable across models without a
blind sweep.

## Positioning


The closest work on this task fine-tunes 8B models on 2,580 stories generated by
GPT-4o and Llama-3.3-70B to hit K–2 readability targets, and reports diversity as
an open problem — global self-BLEU 0.32–0.42, with recurring character names,
animals and settings traced to the teacher model. Our method needs **no training
data, no teacher model and no fine-tuning**: twelve hand-written contrast pairs
per requirement, applied at inference to a 1.7B model.

Against decoding, raising the temperature is the standard way to buy diversity
and it costs compliance; the quality–diversity tradeoff for decoding methods is
well established. Our method is not a decoding change — it operates on the
representation — and it is what lets compliance and diversity move together
rather than against each other.

## The coherence limit, in one place

Six separate passes at this question, gathered rather than left scattered. They agree.

### Where the coherence gap actually comes from


| 200 stories each | Coherent | Failures |
|---|---|---|
| Untouched Qwen3-1.7B, T=1.0 | 194/200 | 6, every one a loop |
| **The steering push alone** | **200/200** | none |
| Push + displacement, gamma=1.5 | 194/200 | 6, every one a loop |
| Nucleus 0.95, T=1.8 | 200/200 | none |
| Top-k 40, T=1.8 | 199/200 | 1 |

**The base model loops at temperature 1.0.** Six of its own 200 stories
degenerate, and every failure is a repetition loop. The decoding baselines reach
200 and 199 by running at temperature 1.8, which flattens the distribution out
of the low-entropy attractors a peaked distribution falls into. Their perfect
coherence is bought with the temperature this method exists to avoid.

**The push cures them.** Steering alone is 200 of 200, better than the model it
is applied to.

**The displacement costs back exactly the base rate.** At gamma=1.5 coherence
returns to 194 of 200 and all six failures are loops again -- the model's own
failure mode re-admitted, not new damage. Against the model it runs on, the
displacement costs nothing.

So the gap is not mysterious and it is not a coherence problem in the ordinary
sense: the push suppresses the base model's loops and the displacement
re-admits them. **A stronger push at the same displacement is the specific
intervention**, and every arm measured here runs at a total push of 2.5.

### Why the coherence is 194 and not 200


Read the stories that fail rather than counting them. At gamma=1.5 six of 200
are rejected, and five of the six have a clean first sixty words: the median
rejected story is fine until word 220 of the 291 it writes, and then loops --
"I'm trapped in the dark" over and over, or "I don't." sixty-eight times. At
gamma=2.0, fourteen of the sixteen rejected stories have clean openings.

Two things follow.

**The failures are endings, not stories.** Variety is scored over the first
forty words, which these stories get right. Coherence and variety are reading
different parts of the same story, and the trade between them is not the
straight exchange the frontier makes it look.

**The failures share a register.** A story that loops is far more likely to have
opened in the first person, and much less likely to have named anybody:

| At gamma=1.5 | Opens in first person | Names a character | Words written |
|---|---|---|---|
| Kept | 14% | 98% | 206 |
| Looped | 67% | 67% | 292 |

The displacement sometimes pushes the opening into an unnamed first-person
present-tense register -- breathless, short sentences -- and the model cannot
resolve it into a story that ends, so it runs forty per cent long and repeats
until the token cap.

#### The allocation the method's own output asks for

The budget is divided by what the *untouched* model gets wrong. The displacement
then breaks requirements the untouched model does not. Re-measuring on the
method's own 200 stories:

| Direction | Round 1, untouched | Round 2, own output | Passes: untouched -> under method |
|---|---|---|---|
| present tense | 1.00 | 1.00 | 0% -> 10% |
| varied openings | 0.81 | 0.73 | 19% -> 35% |
| **named character** | 0.34 | **0.65** | 66% -> **42%** |
| **sensory** | 0.38 | **0.61** | 62% -> 46% |
| **plain words** | 0.56 | **0.00** | 44% -> **100%** |

Naming is the requirement the displacement breaks, and round two moves budget
onto it while taking it off plain words, which the method already satisfies in
every story. The two rounds agree to a cosine of 0.913, so this is a correction
and not a different method.

**It will not fix the coherence, though.** Pooled over 3,000 stories a story
with no named character fails 4.7% of the time against 3.6% for one that names
somebody, and raising the naming rate to 100% predicts about one story of the
six. The 67%-against-14% figure above is six failures in one arm and does not
hold at scale; it is left in place because it is what the failures in that arm
look like, not because it supports the intervention.

This is one step of a fixed point -- allocate from the untouched model, run,
re-measure, allocate again -- and it is the next thing to run. It is not a
sweep: the direction and the size of every change are read off the measurement.

### The trade is structural: writing-time displacement is what carries variety


The perturbation can be applied over the instruction alone, or over the
instruction and while the model writes. The first keeps 199 stories of 200 --
five more than the second -- so it is the obvious candidate for the coherence
criterion. It does not have the variety, and not because of the 40-word cut:

| Variety of what happens, against nucleus | at 40 words | at 100 words |
|---|---|---|
| Top-k 40 | +5.2 [+2.5, +7.3] | +4.7 [+2.8, +6.6] |
| **Over the instruction only** (199/200 coherent) | **−5.9 [−8.9, −3.2]** | **−5.7 [−7.7, −3.5]** |
| **Over the instruction and while writing** (194/200) | **+2.9 [+0.8, +5.4]** | **+4.4 [+2.3, +6.4]** |

The deficit is the same size at both cuts and both intervals are clear of zero.
So displacing the model while it writes is what produces variety of content, and
it is the same component that re-admits the base model's loops. Coherence and
variety are not trading through a knob that could be tuned -- they are trading
through one mechanism that does both.

### The coherence floor, and why neither lever moves it


Four things are measured, and together they say the remaining gap is not a
tuning problem.

| | Coherent |
|---|---|
| Untouched Qwen3-1.7B, T=1.0 | 194/200, every failure a loop |
| The push alone | **200/200** |
| Push + any displacement that improves variety | 184–195/200 |
| Nucleus 0.95 / top-k 40 at T=1.8 | 200/200, 199/200 |

**The base model loops at temperature 1.0 and the push cures it.** Every
displacement large enough to beat nucleus sampling on variety re-admits the
loops, and lands back at or below the model's own rate. The decoding baselines
avoid loops by running at 1.8, which is the thing this method exists not to do.

**A larger push makes it worse, not better** (table below), so the obvious lever
is the wrong way round.

**Spending the same push differently gains about one story.** Pooled over 15
perturbed arms and 3,000 stories, a story with no named character fails 4.7% of
the time against 3.6% for one that names somebody -- 1.3 times, not the five
times a six-story sample suggested. Raising the naming rate from 38% to 100%
predicts 7.3 failures per 200 against the present 8.6.

So within the constraints -- temperature fixed at 1.0, no filtering of outputs,
no decoding guard -- the base model's loop rate at temperature 1.0 is a floor,
and any perturbation strong enough to carry variety sits on it. That is a
statement about the constraint set, not a missing experiment.

A generation-stopping gate removes it: the same gate that takes the untouched
model from 194 to 200 takes this method from 194 to 196, and the residue is not
n-gram looping. The gate is not part of the method.

### Pushing harder does not restore coherence


The push alone is 200 of 200 and the displacement re-admits the base model's
loops, which makes "push harder at the same displacement" the obvious
inference. It is wrong, and the runs already on disk say so. Coherence against
total push, at matched displacement, across every middle-school arm:

| Displacement | Push 2.0 | Push 2.5 | Push 3.0 | Push 3.5 |
|---|---|---|---|---|
| 1.50 | 145/200 | 183–199 | **173/200** | — |
| 1.75 | 162, 177 | 178–191 | **164/200** | — |
| 2.00 | 165/200 | 170–193 | 176/200 | **109/200** |

2.5 is already at or near the best, and 3.5 collapses. The arms differ in other
settings, so this is not a controlled sweep, but the direction is consistent at
three displacements and it agrees with what this project measured long before:
raising the total from 2 to 4 was far worse than leaving it at 2.

So the lever for the coherence gap is not the size of the push but **where it is
spent**. The budget re-measured on the method's own output moves it onto naming
a character -- the requirement the displacement breaks, and the one whose
failure produces the unnamed first-person register that runs long and loops --
at the same total. That remains untested.

### Would a repetition gate close the coherence gap?


The residual failures are repetition loops, and a generation-stopping gate is
the obvious guard. Measured rather than assumed: a gate that stops when some
five-word run has been produced three times, cutting back to the last sentence
end.

| | As written | With that gate |
|---|---|---|
| Untouched, T=1.0 | 194/200 | **200/200** |
| Nucleus 0.95, T=1.8 | 200/200 | 200/200 |
| Ours, gamma=1.5 | 194/200 | 196/200 |
| Ours, gamma=1.75 | 191/200 | 192/200 |
| Ours, gamma=2.0 | 184/200 | 188/200 |

**It fixes the untouched model completely and ours only partly.** The base
model's failures are plain n-gram loops; ours are not all of that kind. Of the
four that survive at gamma=1.5, one degenerates from its first sentence and two
are flagged for low window entropy on prose that repeats its sentence shape
rather than its words. The gate is not in the method and this is not a result --
it is what the number would be under a guard anyone would add.

#### A scorer fault found the same way, and fixed

The fourth survivor was not a failure at all. The refusal check matched any
opening of the form "I can't ...", so a character saying *"I can't breathe. My
knees buckle."* was counted as the model declining the task. It fired on two of
the perturbed arms and on none of the baselines, because only a perturbed arm
writes first-person distress -- so the coherence number was biased against
exactly the register the method produces.

A refusal now has to decline the *task*: the opening must reach for the request
within a short span. Re-scored across every arm, it moves one number by one
story, gamma=1.75 from 190 to 191, and leaves every baseline untouched. Small,
and in our favour, which is why it is stated rather than folded in quietly.

## The rules, rewritten to be about the whole story

The twelve requirements above are mostly thresholds: six sensory words, three
lines of speech, a name used three times, no word of four letters used more than
five. A threshold is easy to score and easy to meet by accident, it says nothing
about whether the story is any good, and its number has to be chosen in advance.

Eight rules replace them, none of them a count. Pass rates are the model's own,
unprompted, so they are a lower bound -- these stories were written for the
older instruction and were never asked for a comparison or a single name.

| Rule | Unprompted |
|---|---|
| every verb in the present tense | **0%** |
| something compared with "like" or "as ... as" | **30%** |
| exactly one character is named | **37%** |
| two characters, one referred to as he and one as she | **38%** |
| Flesch-Kincaid grade at least 3 | 61% |
| somebody speaks, inside quotation marks | 76% |
| no sentence written twice | 86% |
| the story alone, no title or preamble | 100% |

A ninth was written and dropped: reaching past sight passes 98-100% of the time
even narrowed to sound, smell and taste, so it describes how the model already
writes rather than asking anything of it.

Those rates were re-measured on 1400 stories from the decoder comparison, a
larger sample of the same kind -- written for the older instruction, so again a
lower bound. The ordering holds and no rule changes character: present tense
0.1%, a comparison 42%, one name 51%, both genders 47%, reading level 65%,
speech 86%, no repeated sentence 90%, the story alone 100%. Every check fires,
which is the thing that had not been confirmed: these eight are new code, and a
rule that silently never fires would make the compliance column meaningless.

The two at the top of that list are a problem for a fixed budget. The story
format is met 100% of the time and no sentence is repeated 90% of the time, yet
the steered set spends one of its eight directions on each. A quarter of the
push goes to rules the model already follows, and is taken from present tense,
which it follows essentially never. This is the same objection as picking five
directions for twelve rules, in a smaller form.

The closed-loop coefficients below remove it without a calibration run. A
direction that is silent while its rule is met costs nothing on a story that
never breaks the rule, so the format direction spends almost none of the budget
and present tense takes what it needs. The fix falls out of the design rather
than being fitted to this model.

## Steering strength without a calibration run

The budget split above is measured on a calibration sample, which means knowing
in advance how often this model breaks each requirement. That is a property of
one model on one prompt, and it is the part of the method least likely to carry.

Every rule in the new set is satisfied or not, so the error is one-sided: a
direction pushes while its rule is unmet and goes silent the moment it is met.
The coefficient is then a function of the story being written.

- **closure** is zero until the story runs past the length it was asked for, and
  rises with the overrun. The failures this method leaves are stories that do
  not end.
- **present tense** goes to full the instant a past-tense verb appears.
- **the comparison, the speech and the second character** are asked for only
  once the story is far enough in that their absence means something. "No simile
  yet" is true of every story at its third word.
- **naming** pushes for a name at none and *against* one at three, which a
  constant coefficient cannot express.

The controller reads the story with the scorer's own counters. Its first version
used a capitalised-word heuristic of its own and agreed with the scorer on 65%
of 120 real stories, so the naming direction would have pushed the wrong way on
a third of them. It now agrees on all of them.

Neither of these has been run.

## Still to add


- **The steering budget at the larger displacements.** Every arm here runs at a
  total push of 2.5. Beating top-k on variety of what happens needs gamma=2.0,
  which costs 0.4 requirements and 15 stories of coherence; a stronger push is
  the untried lever that should pay for exactly that, and it is one sweep.
  Launched and refused: all three Kaggle accounts stopped starting sessions,
  which is the weekly GPU quota.
- **Dropping the inert directions rather than stretching them.** Built and
  tested (`--fisher-mode keep`), never run, for the same reason. A third of the
  subspace does nothing measurable; whitening stretches it and is a null, and
  dropping it has no amplification to cap.
- Contrastive search, the one decoder still unrun: transformers 5 moved it to a
  repository fetched at run time.
- The 8B at its corrected band with the noise colour, and its largest
  displacement, which the session limit cut off.
- A human or LLM-judge rating of story quality, which no automatic metric here
  covers.

## Random noise, sized by the output

Nothing learned from the model's outputs. For every story, a random subspace of
rank 64 is drawn at each steered layer, projected clear of the rule directions,
and the offset wanders inside it as coloured noise. Its length is set per story
by bisection so that it moves the model's next-token distribution a set
Fisher-Rao distance along a fixed reference passage -- the model's greedy
continuation under the rule steering alone, computed once per prompt. The
distance is stated in units of how far nucleus sampling at temperature 1.8 and
top-p 0.95 moves the same distribution, so 1.0 changes the predictions as much
as the baseline does, spent on one direction per story instead of independently
at every token.

The pullback of the output's Fisher metric onto the residual stream is highly
uneven -- FishBack (arXiv 2605.17231) measures a spectrum spanning seven orders
of magnitude on GPT-2 with 2-17% of directions mattering -- which is why a fixed
activation length is the wrong unit for a random draw: two draws of the same
length can do very different things. FishBack uses the same metric to shape a
deterministic steering vector and reports no diversity; no work found sizes a
random activation perturbation this way.

On Qwen3-1.7B half a nucleus-unit takes an offset of length about 6, roughly 6%
of the residual stream's typical length. A fixed 0.4 times the stream's RMS per
coordinate is a length of about 40.

Baselines on the eight whole-story rules, 25 stories each (wide intervals):

| | Coherent | Rules broken (of 8) | Plot variety | Wording variety |
|---|---|---|---|---|
| Untouched, T=1.0 | 22/25 | 3.09 | 15.6 | 7.1 |
| Nucleus, T=1.8 | 25/25 | 2.44 | 19.3 | 10.3 |
| Steering only | 25/25 | 1.92 | 17.9 | 6.2 |

### What the screens found (25 stories an arm unless stated)

- **A fixed 0.4 x RMS size breaks almost every story**, per story or per token,
  with or without a fade: 0-5 of 25 coherent. It is about seven times the length
  the output-based sizing picks. The fade cannot help because the damage is done
  in the opening words, where a fade is still at full strength.
- **Per-token noise is worse than per-story noise at the same output-based size**
  on wording variety (-2.9 against -1.3 from nucleus).
- **Within one size the chosen length varies about twofold between stories**
  (7.5-14.8 at 1.0 nucleus-units): some random directions move the predictions
  far more than others.
- **Size 2.0 applied only while writing breaks the prose** (7 of 25 coherent,
  looping lists), so the prompt is not the only place a large offset does harm.
  Bringing 2.0 in gradually over the first 260 tokens keeps the rules (1.62
  broken) but loses the variety: the opening and the premise are decided early.
- **The noise alone, with no rule steering, loses 4 of 25 stories to loops**
  where the same noise with the steering loses none.

### The protection only held where the noise was added

The noise is projected clear of the rule directions at each layer it is added
to. Measured downstream along the reference passage, its effect along the rule
directions by layer 13, as a share of the steering push along them, is 1.04 at
0.5 nucleus-units, 2.20 at 1.0 and 3.72 at a fixed length of 14.83. At layer 6
it is 0.00. From layer 9 on, 27-31% of the noise's effect lies along the rule
directions against 19% for a random vector: the network routes it there.

A shadow copy of each story removes this. The same words, the same steering and
no noise run as a second row of the batch; at every steered layer the story is
held level with the shadow along the protected directions. Measured the same
way, the leak falls to 0.015 of the push at every size. It needs no reference
passage and no linear approximation, and the shadow never left the story's words
in 150 stories.

It barely changed how many rules broke: 1.76 to 1.73 at 0.5, 2.16 to 2.04 at
1.0, and 2.36 to 2.82 at 1.4, where it also cost the variety. So the rules break
because the noise changes what the story is about, not because it leaks into
their directions. The rules that fall are the ones about content -- speech, a he
and a she, a comparison -- and a story about one child alone has no second
character however well a direction is protected. Only steering that reacts to
the story can supply one.

### At 50 stories an arm

Every condition pooled at 46 coherent stories; differences are from nucleus.

| | Coherent | Rules broken (of 8) | Plot variety | Wording variety |
|---|---|---|---|---|
| Nucleus, T=1.8 | 49/50 | 2.51 | 26.8 [26.1, 27.5] | 13.1 [12.0, 14.2] |
| Untouched, T=1.0 | 46/50 | 2.93 | -5.4 [-6.4, -4.4] | -3.8 [-5.1, -2.6] |
| Steering only | 50/50 | 1.52 | -1.4 [-2.6, -0.2] | -6.0 [-7.1, -4.7] |
| Shadow 0.5 | 48/50 | 1.73 | -1.0 [-2.1, +0.0] | -5.3 [-6.5, -4.1] |
| Shadow 1.0 | 50/50 | 2.04 | +0.2 [-0.8, +1.4] | -3.3 [-4.9, -1.5] |
| Shadow 1.4 | 50/50 | 2.82 | -0.0 [-0.9, +1.1] | -2.9 [-4.3, -1.6] |

Every story coherent and fewer rules broken than nucleus, at 1.0: yes. Variety:
level with nucleus on what happens, about three points behind on wording. The
noise does not flatten the model's choices word by word, which is where nucleus
gets its wording variety, so the gap is in kind rather than in size.

### A seed offset that did nothing

The run meant to add a second 25 stories to every arm reproduced the first 25
word for word: the offset had been added to a seed function this runner does
not call, and was checked on that function. Fixed, with a test on the call the
generation loop actually makes; the duplicate run is kept under
`experiments/r119-DUPLICATE-of-first-25` and must not be pooled.

### The simile rule undercounts

It counts "like" only before a/an/the/some/two/three, to keep the verb "like"
out, so "shines like glass" is not a simile to it. It is the same for every arm,
so comparisons stand, but every simile pass rate here is a lower bound.

### Every arm at fifty stories

Pooled at 46 coherent stories; differences are from nucleus sampling at
temperature 1.8 and top-p 0.95 (plot 26.8, wording 13.1, 2.51 rules broken of 8,
49 of 50 coherent). Noise sizes are in nucleus-units unless marked fixed.

| | Coherent | Rules broken | Plot variety | Wording variety |
|---|---|---|---|---|
| Untouched, T=1.0 | 46/50 | 2.93 | -5.4 [-6.4, -4.4] | -3.8 [-5.1, -2.6] |
| Steering only | 50/50 | 1.52 | -1.4 [-2.6, -0.2] | -6.0 [-7.1, -4.7] |
| Noise 0.5 | 49/50 | 1.67 | -0.6 [-1.7, +0.4] | -3.9 [-5.3, -2.6] |
| Noise 1.0 | 50/50 | 2.28 | +0.9 [-0.1, +1.9] | -2.8 [-4.3, -1.1] |
| Noise 1.4 | 49/50 | 2.84 | +1.3 [+0.4, +2.3] | -2.9 [-4.4, -1.6] |
| Fixed length 14.83 | 49/50 | 3.20 | +1.0 [-0.3, +2.0] | -1.4 [-2.9, +0.0] |
| Fixed 14.83, fade | 49/50 | 3.08 | +0.8 [-0.4, +1.8] | -1.2 [-2.6, +0.4] |
| Colour 1 at 1.0 | 47/50 | 2.30 | +1.5 [+0.7, +2.4] | -4.2 [-5.3, -2.8] |
| 2.0 brought in gradually | 48/50 | 1.85 | -1.5 [-2.5, -0.4] | -5.0 [-6.5, -3.5] |
| Noise 1.0, no steering | 46/50 | 2.87 | -2.6 [-3.7, -1.5] | -0.8 [-2.3, +1.0] |
| Shadow 1.0 | 50/50 | 2.04 | +0.3 [-0.9, +1.4] | -3.3 [-4.9, -1.9] |

No arm clears all four at once. Noise 1.0 is the best balance: every story
coherent, fewer rules broken than nucleus, level or ahead on what happens.
Wording variety is the axis nothing beats, and it follows how hard and how
uniformly the rules are pushed: steering alone is the worst arm on it, the
largest noise is the best of the steered arms, and it costs rules.

### Steering set by the story, not by a constant

The controller that pushes each rule only once the story is seen breaking it,
with the same noise at 1.0 (50 stories an arm, pooled at 43):

| | Rules broken | Plot variety | Wording variety |
|---|---|---|---|
| Constant push + noise 1.0 | 2.45 | +1.0 [+0.2, +1.9] | -2.7 [-4.2, -1.1] |
| Controller, gain 1 | 3.27 | tie | -1.0 [-2.4, +0.6] |
| Controller, gain 2 | 3.31 | tie | -0.9 [-2.4, +0.5] |
| Controller, gain 2, noise 1.4 | 3.33 | tie | +0.3 [-1.2, +1.8] |
| Controller, noise rising as rules are secured | 3.37 | tie | -0.7 [-2.1, +0.8] |
| Controller, gain 2, no noise | 2.86 | -5.0 | -4.1 |

It keeps the rules far worse than the constant push -- speech falls to 22-31% --
because these rules are decided early and the controller acts after the fact:
once a past-tense verb is written the tense rule is already broken, and a push
for speech from halfway in rarely produces any. The noise that rises as rules
are secured was never really tested, since the rules were seldom secured. What
the arms do show is where the wording variety goes: every controller arm ties
nucleus on wording and every constant arm loses, so the constant push is what
narrows the wording.

### A random split of the push, and a push on fewer rules

Neither helps (50 stories an arm, pooled at 45, noise 1.0 throughout). Drawing
each story's split of the budget from a Dirichlet keeps the total push but not
the wording: -3.5 against nucleus at both concentrations, against -2.7 for the
even split, with more rules broken (2.58 and 2.67 against 2.28). Taking the
budget off the two rules already met unprompted costs coherence -- 45 of 50 --
and wording (-4.5): the no-repeated-sentence direction holds off repetition
loops once noise is added, although the model meets that rule unaided without
noise. Whatever narrows the wording, it is not that every story is pushed in the
same direction.

### Over the first hundred words

The same stories, variety over the first 100 words instead of 40, pooled at 46.
Differences are from nucleus (plot 24.8, wording 7.3); these are not comparable
with the 40-word numbers, only with each other.

| | Plot variety | Wording variety |
|---|---|---|
| Untouched | -3.8 [-4.8, -2.8] | -1.3 [-2.2, -0.6] |
| Steering only | -1.0 [-1.8, -0.1] | -3.1 [-3.8, -2.4] |
| Noise 0.5 | +0.0 [-0.7, +0.9] | -2.6 [-3.4, -2.0] |
| Noise 1.0 | +1.7 [+0.9, +2.5] | -1.2 [-2.2, -0.3] |
| Noise 1.4 | +2.5 [+1.6, +3.4] | -0.8 [-1.8, +0.0] |
| Fixed length 14.83 | +2.7 [+1.8, +3.6] | -0.6 [-1.4, +0.3] |
| Colour 1 at 1.0 | +1.8 [+1.0, +2.6] | -1.6 [-2.4, -0.9] |
| Shadow 1.0 | +1.2 [+0.3, +2.0] | -1.5 [-2.4, -0.7] |

Past the opening the method is clearly ahead of nucleus sampling on what
happens, from 1.0 up, and the wording gap shrinks to about a point -- a tie at
1.4 and at the fixed length. The gap is concentrated in the opening: under the
push most stories begin the same way ("The sun spills ..."), and the first 40
words are mostly that opening.

### Larger noise over the opening, and on the prompt

Noise at 1.0 that starts at 1.5x or 2x and falls back over the first 40 tokens,
while writing (50 stories an arm, pooled at 36 because of the last row):

| | Coherent | Rules broken | Plot, 40 words | Wording, 40 words | Plot, 100 words | Wording, 100 words |
|---|---|---|---|---|---|---|
| Noise 1.0 | 50/50 | 2.28 | +0.6 [-0.3, +1.5] | -2.2 [-3.9, -0.3] | +1.2 [+0.4, +1.9] | -1.0 [-2.0, +0.1] |
| Starting at 1.5x | 50/50 | 2.54 | +1.0 [+0.3, +1.7] | -2.8 [-4.2, -1.3] | +1.1 [+0.3, +1.9] | -1.4 [-2.3, -0.4] |
| Starting at 2x | 48/50 | 2.90 | +0.8 [-0.1, +1.7] | -3.1 [-4.8, -1.4] | +1.7 [+0.8, +2.6] | -1.4 [-2.3, -0.4] |
| 2x, prompt's share 2x too | 36/50 | 3.50 | +0.8 [-0.0, +1.6] | +0.4 [-1.2, +2.3] | +2.1 [+1.3, +2.9] | +1.7 [+0.4, +2.8] |

Larger noise while writing does nothing for the wording and costs rules. The
wording moves only when the prompt's share rises -- the one arm to beat nucleus
on wording at a hundred words -- and that arm loses 14 of 50 stories to the model
answering the reader ("Certainly! The story must satisfy every one of these
requirements"), to loops and to repeated titles. So the prompt is where the
wording variety comes from, and where the register failure comes from.

Keeping the larger prompt noise while guarding against that failure (50 stories
an arm, pooled at 37; noise while writing at 1.0 throughout):

| | Coherent | Rules broken | Plot, 40 | Wording, 40 | Plot, 100 | Wording, 100 |
|---|---|---|---|---|---|---|
| Prompt 2x, fading to a quarter | 40/50 | 2.65 | +1.3 [+0.7, +2.0] | -2.9 [-4.3, -1.5] | +1.8 [+1.0, +2.4] | -1.0 [-2.0, +0.0] |
| Prompt 2x, with the shadow | 37/50 | 3.43 | +0.3 [-0.5, +1.2] | +1.3 [+0.0, +2.8] | +1.7 [+1.0, +2.4] | +1.0 [-0.0, +2.0] |
| Prompt 1.5x | 49/50 | 2.63 | +0.9 [+0.1, +1.8] | -1.7 [-3.5, -0.1] | +2.2 [+1.4, +3.0] | -0.3 [-1.2, +0.8] |

The fade removes the answering-the-reader openings entirely but loses ten stories
another way -- endings cut off mid-sentence, run-ons -- and takes the wording gain
with it. The shadow does the opposite of what it was for: the protected
directions stay level and 13 of 50 stories still open with a refusal, a preamble
or a title, so the register is not carried by them. At 1.5x the prompt noise
keeps 49 of 50 coherent, level with nucleus, wins on what happens at both
lengths and ties on wording at a hundred words; it still trails on wording over
the opening forty and breaks slightly more rules than nucleus. Every arm that
beats nucleus on wording over the opening does it with register failures.

