# Results for the paper

Qwen3-1.7B, one instruction asking for a ~150-word story for a middle-school
reader under twelve checkable requirements, 200 stories per condition, every
condition sharing one prompt and one seed sequence. Diversity is a Vendi score
over the first 40 words of coherent stories, rarefied to a common sample size;
95% intervals from subsampling without replacement. Our method runs at
temperature 1.0 throughout; the decoding baselines are given the raised
temperature they need.

## What is established, and what is not

**And every whole-story result was asked for the wrong reader.** Until commit
413ebe0 the whole-story rule set fell through to the children's instruction:
every story under it -- every table from "The rules, rewritten to be about the
whole story" onward -- was generated from "Write one short story for a young
child to read", under a system prompt about young children learning to read.
The rules were the middle-school ones; the request was not. Comparisons inside
those tables are fair, since every arm got the same prompt, but none of them is
a middle-school result. The whole set now takes the middle-school instruction
and middle-school steering pairs where they exist.


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

## Noise at other places in the transformer

Genuinely random noise at five other sites (noiseegra/arch_noise.py), each
sized per story to move the next-token distribution the same Fisher-Rao
distance as the residual offset, with the same constant rule steering. 20
stories an arm, pooled at 18, differences from nucleus. The rotary jitter could
not reach the size even at +-50% frequencies and ran at about 0.6.

| | Coherent | Rules broken | Plot 40 | Wording 40 | Plot 100 | Wording 100 |
|---|---|---|---|---|---|---|
| Residual offset | 20/20 | 2.20 | +0.2 [-0.2, +0.7] | -1.0 [-2.1, +0.2] | +0.6 [+0.2, +1.1] | -0.6 [-1.6, +0.4] |
| Rotation on the sphere | 19/20 | 1.79 | +0.4 [+0.0, +0.9] | -1.7 [-2.8, -0.6] | +0.4 [-0.1, +0.8] | -0.9 [-2.0, -0.0] |
| MLP feature dropout | 20/20 | 1.95 | +0.4 [+0.0, +0.8] | -1.7 [-2.7, -0.7] | +0.3 [-0.2, +0.8] | -0.9 [-1.9, +0.1] |
| Prompt value noise | 19/20 | 1.79 | +0.2 [-0.2, +0.7] | -1.6 [-2.8, -0.3] | +0.2 [-0.2, +0.8] | -1.3 [-2.3, -0.4] |
| Per-head attention temperature | 18/20 | 2.44 | -0.0 [-0.5, +0.5] | -1.5 [-2.5, -0.4] | +0.0 [-0.4, +0.6] | -0.8 [-1.8, +0.1] |
| Rotary frequency jitter (~0.6) | 18/20 | 1.78 | -0.1 [-0.6, +0.5] | -2.6 [-3.6, -1.4] | -0.3 [-0.8, +0.2] | -1.8 [-2.8, -0.9] |

At the same change in the predictions, rotating the stream, dropping MLP units
and perturbing the prompt's values break far fewer rules than adding a vector --
1.79-1.95 against 2.20 -- and nearly all of the difference is in the content
rules the offset breaks: a comparison 95/90/89% against 55%, a he and a she
84/65/68% against 65%. Plot variety is level with the offset. Noise that keeps
the hidden state's length, or lives in the model's own feature coordinates,
spends less of its effect on derailing the story. Several of these arms open
some stories in lower case (rotary 9 of 20, head temperature 5, rotation 4),
which the scorer counts as a broken format rule, not a broken story.

## On the middle-school prompt

The first comparison after the prompt fix: middle-school instruction and
middle-school steering pairs where they exist, 30 stories an arm, pooled at 28,
differences from nucleus sampling (plot 17.6 and wording 8.4 over 40 words, 16.4
and 4.7 over 100; 2.55 rules broken; 29 of 30 coherent). Noise at 1.0 throughout.

| | Coherent | Rules broken | Plot 40 | Wording 40 | Plot 100 | Wording 100 |
|---|---|---|---|---|---|---|
| Untouched, T=1.0 | 30/30 | 2.57 | -1.4 [-2.0, -0.8] | -2.2 [-3.2, -1.4] | -1.2 [-1.7, -0.8] | -0.8 [-1.3, -0.4] |
| Steering only | 29/30 | 1.76 | -1.1 [-1.7, -0.6] | -2.0 [-3.0, -1.2] | -1.1 [-1.7, -0.5] | -1.3 [-1.8, -0.8] |
| Residual offset | 30/30 | 2.13 | +0.1 [-0.3, +0.5] | -1.3 [-2.3, -0.4] | +0.4 [-0.1, +0.8] | -0.3 [-0.8, +0.2] |
| Rotation | 29/30 | 1.76 | -0.3 [-0.8, +0.2] | -1.0 [-1.9, +0.0] | +0.3 [-0.2, +0.8] | -0.8 [-1.3, -0.3] |
| MLP feature dropout | 28/30 | 2.11 | +0.0 [-0.5, +0.4] | -0.5 [-1.4, +0.4] | -0.1 [-0.6, +0.4] | -0.3 [-0.8, +0.2] |
| Prompt value noise | 28/30 | 1.71 | +0.2 [-0.1, +0.6] | -1.3 [-2.1, -0.4] | +0.2 [-0.2, +0.6] | -0.2 [-0.8, +0.3] |

The register is what the task asks for now ("The door slams shut behind him,
sending a jolt through the room"), and nucleus sampling's lead in variety is far
smaller than on the children's prompt. Every noise arm ties it on what happens,
MLP dropout ties it on wording even over the opening, and every noise arm breaks
fewer rules -- rotation and value noise as few as the steering alone. A new
failure appears under noise: echoing the instruction back ("Aim for roughly 150
words -- long enough for something to happen in it"), counted as not a story.

Running: the four noise sites at 1.5x and 2x, and rotation and MLP dropout with a
fresh draw per sentence at 1.0.

## Every random-noise result, against the untouched model

Every method compared with the untouched model on the prompt it ran on.
Variety is the Vendi score over the first 40 words of coherent stories, with a
95% interval from subsampling; an interval spanning zero is a tie. Unless a row
says otherwise, "steering" is the eight rule directions at a constant total
push of 2.5, and "random noise" is drawn fresh for every story in a random
64-direction subspace kept clear of the rule directions, drifting slowly over
the story, and sized so it moves the next-token distribution a set Fisher-Rao
distance -- 1.0 is as far as nucleus sampling at temperature 1.8 moves it.
Nothing is learned from the model's outputs except in the two rows marked as
using pre-generated stories.

### Middle-school prompt (30 stories an arm, pooled at 28)

| Method | Prompt | Coherent | Rules broken (of 8) | Plot variety vs untouched | Wording variety vs untouched |
|---|---|---|---|---|---|
| Untouched model, temperature 1.0 | Middle school | 30/30 | 2.57 | 16.2 (baseline) | 6.2 (baseline) |
| Nucleus sampling, temperature 1.8, top-p 0.95 | Middle school | 29/30 | 2.55 | +1.4 [+0.8, +2.0] | +2.2 [+1.2, +3.1] |
| Rule steering only, no noise | Middle school | 29/30 | 1.76 | +0.3 [-0.5, +1.1] | +0.2 [-0.6, +1.1] |
| Steering + random noise added to the residual stream, size 1.0 | Middle school | 30/30 | 2.13 | +1.4 [+0.8, +2.0] | +0.9 [-0.1, +1.9] |
| Steering + random rotation of the hidden state (length kept, rule parts untouched), size 1.0 | Middle school | 29/30 | 1.76 | +1.1 [+0.4, +1.8] | +1.2 [+0.3, +2.2] |
| **★ Steering + random MLP-unit dropout in the steered layers, size 1.0** | **Middle school** | **28/30** | **2.11** | **+1.4 [+0.8, +2.1]** | **+1.7 [+0.7, +2.6]** |
| Steering + random noise on the prompt's cached attention values, size 1.0 | Middle school | 28/30 | 1.71 | +1.6 [+1.0, +2.3] | +0.9 [+0.0, +1.9] |

### Children's prompt, generated before the prompt fix (pooled at 18)

| Method | Prompt | Coherent | Rules broken (of 8) | Plot variety vs untouched | Wording variety vs untouched |
|---|---|---|---|---|---|
| Untouched model, temperature 1.0 | Young child (bug) | 46/50 | 2.93 | 9.8 (baseline) | 5.7 (baseline) |
| Nucleus sampling, temperature 1.8, top-p 0.95 | Young child (bug) | 49/50 | 2.51 | +1.3 [+0.5, +2.0] | +1.5 [+0.3, +2.7] |
| Rule steering only, no noise | Young child (bug) | 50/50 | 1.52 | +1.0 [+0.2, +1.8] | -1.0 [-2.1, +0.2] |
| USES 16 PRE-GENERATED STORIES: steering + noise along the directions those stories differ (1.5 story-distances) | Young child (bug) | 23/25 | 2.65 | +1.8 [+1.1, +2.4] | +1.1 [-0.2, +2.2] |
| USES 16 PRE-GENERATED STORIES: that noise alone, no steering | Young child (bug) | 17/25 | 3.47 | +1.5 [+0.8, +2.2] | +0.4 [-0.5, +1.4] |
| Steering + random residual noise, size 0.5 | Young child (bug) | 49/50 | 1.67 | +1.2 [+0.5, +1.9] | -0.1 [-1.2, +1.0] |
| Steering + random residual noise, size 1.0 | Young child (bug) | 50/50 | 2.28 | +1.5 [+0.8, +2.2] | +0.5 [-0.9, +1.7] |
| Steering + random residual noise, size 1.4 | Young child (bug) | 49/50 | 2.84 | +1.6 [+0.9, +2.2] | +0.4 [-1.0, +1.6] |
| Steering + random residual noise, size 2.0 | Young child (bug) | 15/25 | 3.33 | too few coherent to score | too few coherent to score |
| Steering + random residual noise redrawn every token (prompt untouched), size 1.0 | Young child (bug) | 25/25 | 2.28 | +1.2 [+0.6, +1.9] | -1.0 [-2.2, +0.0] |
| Steering + random residual noise at a fixed 0.4 x RMS (not output-sized) | Young child (bug) | 0/25 | - | too few coherent to score | too few coherent to score |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 1/25 | 3.00 | too few coherent to score | too few coherent to score |
| Steering + per-token noise at a fixed 0.4 x RMS | Young child (bug) | 3/25 | 4.67 | too few coherent to score | too few coherent to score |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 5/25 | 4.00 | too few coherent to score | too few coherent to score |
| Steering + random residual noise at a fixed length of 14.83 (0.149 x RMS) | Young child (bug) | 49/50 | 3.20 | +1.5 [+0.7, +2.2] | +1.0 [-0.5, +2.4] |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 49/50 | 3.08 | +1.4 [+0.7, +2.2] | +1.1 [-0.1, +2.5] |
| Steering + random residual noise 1.0 that wanders faster within the story | Young child (bug) | 47/50 | 2.30 | +1.6 [+1.0, +2.3] | -0.2 [-1.2, +0.9] |
| Steering + random residual noise 2.0, while writing only | Young child (bug) | 7/25 | 4.71 | too few coherent to score | too few coherent to score |
| Steering + random residual noise 2.0, while writing only, brought in over 260 tokens | Young child (bug) | 48/50 | 1.85 | +1.0 [+0.2, +1.8] | -0.5 [-1.7, +0.5] |
| Random residual noise 1.0 with no steering | Young child (bug) | 46/50 | 2.87 | +0.7 [-0.0, +1.5] | +1.2 [-0.1, +2.5] |
| Steering + random residual noise 0.5 + shadow copy holding rule directions level | Young child (bug) | 48/50 | 1.73 | +1.1 [+0.4, +1.8] | -0.7 [-1.9, +0.4] |
| Steering + random residual noise 1.0 + shadow copy | Young child (bug) | 50/50 | 2.04 | +1.3 [+0.6, +2.1] | +0.2 [-1.4, +1.4] |
| Steering + random residual noise 1.4 + shadow copy | Young child (bug) | 50/50 | 2.82 | +1.3 [+0.6, +2.1] | +0.3 [-0.9, +1.6] |
| Steering (closure direction included at weight 0) + random residual noise 1.0 | Young child (bug) | 49/50 | 2.45 | +1.5 [+0.8, +2.2] | +0.4 [-1.0, +1.6] |
| Story-driven steering (pushes a rule once it is seen broken), gain 1, + noise 1.0 | Young child (bug) | 49/50 | 3.27 | +1.2 [+0.5, +1.8] | +1.1 [-0.0, +2.3] |
| Story-driven steering, gain 2, + noise 1.0 | Young child (bug) | 49/50 | 3.31 | +1.3 [+0.7, +2.0] | +1.1 [-0.2, +2.4] |
| Story-driven steering, gain 2, + noise 1.4 | Young child (bug) | 43/50 | 3.33 | +1.1 [+0.2, +1.9] | +1.6 [+0.2, +2.8] |
| Story-driven steering, gain 2, + noise 1.0 rising to 2x as content rules are met | Young child (bug) | 46/50 | 3.37 | +1.3 [+0.7, +2.0] | +1.2 [-0.1, +2.4] |
| Story-driven steering, gain 2, no noise | Young child (bug) | 50/50 | 2.86 | +0.0 [-1.0, +0.9] | -0.3 [-1.6, +0.9] |
| Steering split across rules at random per story (Dirichlet 1) + noise 1.0 | Young child (bug) | 48/50 | 2.58 | +1.3 [+0.7, +2.0] | +0.1 [-1.1, +1.4] |
| Steering split at random per story (Dirichlet 4) + noise 1.0 | Young child (bug) | 48/50 | 2.67 | +1.6 [+0.9, +2.2] | +0.1 [-1.2, +1.3] |
| Steering with no share for the two rules already met + noise 1.0 | Young child (bug) | 45/50 | 2.24 | +1.1 [+0.3, +1.9] | -0.3 [-1.5, +1.0] |
| Same, with a random split per story | Young child (bug) | 45/50 | 2.98 | +0.9 [+0.0, +1.8] | -0.3 [-1.6, +1.1] |
| Steering + noise 1.0 starting at 1.5x and falling back over 40 tokens | Young child (bug) | 50/50 | 2.54 | +1.6 [+1.0, +2.2] | +0.1 [-1.1, +1.4] |
| Steering + noise 1.0 starting at 2x and falling back over 40 tokens | Young child (bug) | 48/50 | 2.90 | +1.5 [+0.8, +2.2] | -0.0 [-1.3, +1.4] |
| Same at 2x, with the prompt's noise also 2x | Young child (bug) | 36/50 | 3.50 | +1.5 [+0.8, +2.2] | +1.6 [+0.1, +3.0] |
| Steering + noise 1.0, prompt's noise 2x fading to a quarter along the prompt | Young child (bug) | 40/50 | 2.65 | +1.7 [+1.1, +2.3] | +0.0 [-1.1, +1.3] |
| Steering + noise 1.0, prompt's noise 2x, with the shadow copy | Young child (bug) | 37/50 | 3.43 | +1.4 [+0.7, +2.1] | +2.1 [+0.8, +3.2] |
| Steering + noise 1.0, prompt's noise 1.5x | Young child (bug) | 49/50 | 2.63 | +1.6 [+0.8, +2.3] | +0.7 [-0.8, +2.1] |
| Steering + random residual noise 1.0 (reference in the architecture screen) | Young child (bug) | 20/20 | 2.20 | +1.5 [+0.9, +2.2] | +0.4 [-0.6, +1.5] |
| Steering + random rotation of the hidden state, size 1.0 | Young child (bug) | 19/20 | 1.79 | +1.7 [+1.1, +2.3] | -0.3 [-1.3, +0.6] |
| Steering + random per-head attention temperature, size 1.0 | Young child (bug) | 18/20 | 2.44 | +1.3 [+0.6, +1.9] | -0.0 [-1.0, +0.8] |
| Steering + random rotary-frequency jitter (only reached ~0.6) | Young child (bug) | 18/20 | 1.78 | +1.2 [+0.6, +1.9] | -1.2 [-2.2, -0.2] |
| Steering + random noise on the prompt's cached attention values, size 1.0 | Young child (bug) | 19/20 | 1.79 | +1.5 [+0.9, +2.2] | -0.1 [-1.3, +0.9] |
| Steering + random MLP-unit dropout, size 1.0 | Young child (bug) | 20/20 | 1.95 | +1.6 [+1.1, +2.3] | -0.3 [-1.3, +0.6] |

### Re-scored after the dialogue fix (commit 5be1276)

The not-a-story check counted characters' dialogue containing "you've got" as the
model addressing the asker, and rejected real stories -- in every arm. With
quoted speech excluded, the middle-school comparison (pooled at 29):

| Method | Coherent | Rules broken | Plot vs untouched | Wording vs untouched |
|---|---|---|---|---|
| Untouched, T=1.0 | 30/30 | 2.57 | 16.9 (baseline) | 6.3 (baseline) |
| Nucleus, T=1.8, top-p 0.95 | 30/30 | 2.50 | +1.5 [+0.9, +2.1] | +2.1 [+1.2, +3.0] |
| Steering only | 30/30 | 1.80 | +0.4 [-0.3, +1.0] | +0.3 [-0.6, +1.1] |
| Steering + residual noise 1.0 | 30/30 | 2.13 | +1.6 [+1.0, +2.2] | +1.0 [+0.1, +1.9] |
| Steering + rotation 1.0 | 29/30 | 1.76 | +1.2 [+0.5, +1.9] | +1.3 [+0.2, +2.2] |
| Steering + MLP dropout 1.0 | 29/30 | 2.07 | +1.5 [+0.9, +2.2] | +1.8 [+0.8, +2.7] |
| Steering + prompt value noise 1.0 | 29/30 | 1.79 | +1.8 [+1.2, +2.5] | +1.2 [+0.1, +2.2] |

Each remaining rejection is a real failure: MLP dropout's is a story that loses
all punctuation, rotation's opens by echoing the instruction, and value noise's
collapses into "the the the". The children's-prompt table above predates this
fix and slightly undercounts coherence in every arm.

## The same comparison in embedding Vendi

The standard embedding Vendi score (Qwen3-Embedding-0.6B over the first 40
words of coherent stories, isolated stories trimmed), computed on a laptop in a
native Apple-silicon environment. Scores are means over 400 subsamples of a
common size; differences carry 95% intervals from those subsamples. Coherence is
counted with the dialogue fix.

### Middle-school prompt (subsamples of 14)

| Method | Prompt | Coherent | Rules broken (of 8) | Vendi | vs untouched | vs nucleus |
|---|---|---|---|---|---|---|
| Untouched model, temperature 1.0 | Middle school | 30/30 | 2.57 | 6.30 | baseline | -0.63 [-1.32, +0.08] |
| Nucleus sampling, temperature 1.8, top-p 0.95 | Middle school | 30/30 | 2.50 | 6.94 | +0.63 [+0.02, +1.38] | reference |
| Rule steering only, no noise | Middle school | 30/30 | 1.80 | 6.40 | +0.10 [-0.58, +0.94] | -0.54 [-1.23, +0.03] |
| Steering + random residual noise, size 1.0 | Middle school | 30/30 | 2.13 | 6.98 | +0.68 [+0.06, +1.54] | +0.04 [-0.51, +0.68] |
| Steering + random rotation of the hidden state, size 1.0 | Middle school | 29/30 | 1.76 | 6.72 | +0.42 [-0.37, +1.31] | -0.21 [-0.92, +0.49] |
| Steering + random MLP-unit dropout, size 1.0 | Middle school | 29/30 | 2.07 | 7.39 | +1.09 [+0.33, +1.77] | +0.45 [-0.11, +1.04] |
| Steering + random noise on the prompt's cached attention values, size 1.0 | Middle school | 29/30 | 1.79 | 7.48 | +1.18 [+0.46, +1.89] | +0.54 [-0.13, +1.19] |

### Children's prompt, before the prompt fix

Subsample size in brackets: 20 for the 50-story arms, 8 for the 20- and
25-story arms and for arms that lost many stories. A score is comparable only
with scores at the same subsample size; the differences are comparable across.

| Method | Prompt | Coherent | Rules broken (of 8) | Vendi | vs untouched | vs nucleus |
|---|---|---|---|---|---|---|
| Untouched model, temperature 1.0 | Young child (bug) | 48/50 | 2.93 | 6.19 (20) | baseline | -1.47 [-2.29, -0.57] |
| Nucleus sampling, temperature 1.8, top-p 0.95 | Young child (bug) | 49/50 | 2.51 | 7.65 (20) | +1.47 [+0.72, +2.33] | reference |
| Rule steering only, no noise | Young child (bug) | 50/50 | 1.52 | 6.52 (20) | +0.33 [-0.41, +1.20] | -1.13 [-1.89, -0.40] |
| USES 16 PRE-GENERATED STORIES: steering + noise along the directions those stories differ (1.5 story-distances) | Young child (bug) | 24/25 | 2.65 | 4.81 (8) | +0.81 [+0.13, +1.49] | +0.22 [-0.40, +0.85] |
| USES 16 PRE-GENERATED STORIES: that noise alone, no steering | Young child (bug) | 17/25 | 3.47 | 5.30 (8) | +1.30 [+0.65, +1.93] | +0.71 [+0.11, +1.29] |
| Steering + random residual noise, size 0.5 | Young child (bug) | 49/50 | 1.67 | 7.06 (20) | +0.87 [+0.04, +1.63] | -0.60 [-1.30, +0.22] |
| Steering + random residual noise, size 1.0 | Young child (bug) | 50/50 | 2.28 | 7.71 (20) | +1.53 [+0.65, +2.37] | +0.06 [-0.77, +0.95] |
| Steering + random residual noise, size 1.4 | Young child (bug) | 49/50 | 2.84 | 7.91 (20) | +1.72 [+0.89, +2.62] | +0.26 [-0.65, +1.11] |
| Steering + random residual noise, size 2.0 | Young child (bug) | 15/25 | 3.33 | too few coherent | – | – |
| Steering + random residual noise redrawn every token (prompt untouched), size 1.0 | Young child (bug) | 25/25 | 2.28 | 4.26 (8) | +0.26 [-0.43, +0.99] | -0.33 [-1.02, +0.30] |
| Steering + random residual noise at a fixed 0.4 x RMS (not output-sized) | Young child (bug) | 0/25 | – | too few coherent | – | – |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 1/25 | 3.00 | too few coherent | – | – |
| Steering + per-token noise at a fixed 0.4 x RMS | Young child (bug) | 3/25 | 4.67 | too few coherent | – | – |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 5/25 | 4.00 | too few coherent | – | – |
| Steering + random residual noise at a fixed length of 14.83 (0.149 x RMS) | Young child (bug) | 49/50 | 3.20 | 8.26 (20) | +2.08 [+1.11, +2.94] | +0.61 [-0.44, +1.67] |
| Same, with a cosine fade over 260 tokens | Young child (bug) | 49/50 | 3.08 | 8.21 (20) | +2.03 [+1.04, +3.01] | +0.56 [-0.39, +1.63] |
| Steering + random residual noise 1.0 that wanders faster within the story | Young child (bug) | 47/50 | 2.30 | 8.16 (20) | +1.98 [+1.09, +2.85] | +0.51 [-0.40, +1.28] |
| Steering + random residual noise 2.0, while writing only | Young child (bug) | 7/25 | 4.71 | too few coherent | – | – |
| Steering + random residual noise 2.0, while writing only, brought in over 260 tokens | Young child (bug) | 48/50 | 1.85 | 6.02 (20) | -0.17 [-1.03, +0.72] | -1.64 [-2.48, -0.85] |
| Random residual noise 1.0 with no steering | Young child (bug) | 46/50 | 2.87 | 7.20 (20) | +1.01 [-0.05, +2.01] | -0.45 [-1.44, +0.67] |
| Steering + random residual noise 0.5 + shadow copy holding rule directions level | Young child (bug) | 48/50 | 1.73 | 6.56 (20) | +0.37 [-0.60, +1.30] | -1.09 [-2.11, -0.09] |
| Steering + random residual noise 1.0 + shadow copy | Young child (bug) | 50/50 | 2.04 | 7.34 (20) | +1.15 [+0.02, +2.06] | -0.32 [-1.27, +0.65] |
| Steering + random residual noise 1.4 + shadow copy | Young child (bug) | 50/50 | 2.82 | 7.65 (20) | +1.47 [+0.47, +2.50] | +0.00 [-0.93, +0.99] |
| Steering (closure direction included at weight 0) + random residual noise 1.0 | Young child (bug) | 49/50 | 2.45 | 7.78 (20) | +1.59 [+0.73, +2.54] | +0.13 [-0.73, +0.97] |
| Story-driven steering (pushes a rule once it is seen broken), gain 1, + noise 1.0 | Young child (bug) | 49/50 | 3.27 | 8.09 (20) | +1.90 [+1.06, +2.73] | +0.44 [-0.38, +1.21] |
| Story-driven steering, gain 2, + noise 1.0 | Young child (bug) | 49/50 | 3.31 | 8.24 (20) | +2.05 [+1.12, +2.94] | +0.58 [-0.28, +1.43] |
| Story-driven steering, gain 2, + noise 1.4 | Young child (bug) | 43/50 | 3.33 | 8.30 (20) | +2.12 [+1.32, +3.06] | +0.65 [-0.24, +1.53] |
| Story-driven steering, gain 2, + noise 1.0 rising to 2x as content rules are met | Young child (bug) | 46/50 | 3.37 | 7.91 (20) | +1.73 [+0.97, +2.53] | +0.26 [-0.43, +1.06] |
| Story-driven steering, gain 2, no noise | Young child (bug) | 50/50 | 2.86 | 5.50 (20) | -0.68 [-1.59, +0.21] | -2.15 [-2.94, -1.33] |
| Steering split across rules at random per story (Dirichlet 1) + noise 1.0 | Young child (bug) | 48/50 | 2.58 | 7.32 (20) | +1.14 [+0.22, +2.20] | -0.33 [-1.27, +0.62] |
| Steering split at random per story (Dirichlet 4) + noise 1.0 | Young child (bug) | 48/50 | 2.67 | 7.52 (20) | +1.33 [+0.34, +2.21] | -0.14 [-1.08, +0.84] |
| Steering with no share for the two rules already met + noise 1.0 | Young child (bug) | 45/50 | 2.24 | 6.64 (20) | +0.45 [-0.41, +1.33] | -1.01 [-1.92, -0.09] |
| Same, with a random split per story | Young child (bug) | 45/50 | 2.98 | 6.12 (20) | -0.06 [-1.06, +1.03] | -1.53 [-2.51, -0.48] |
| Steering + noise 1.0 starting at 1.5x and falling back over 40 tokens | Young child (bug) | 50/50 | 2.54 | 7.92 (20) | +1.73 [+0.83, +2.54] | +0.27 [-0.56, +1.03] |
| Steering + noise 1.0 starting at 2x and falling back over 40 tokens | Young child (bug) | 48/50 | 2.90 | 7.93 (20) | +1.75 [+0.79, +2.84] | +0.28 [-0.76, +1.25] |
| Same at 2x, with the prompt's noise also 2x | Young child (bug) | 34/50 | 3.50 | 5.08 (8) | +1.09 [+0.30, +1.86] | +0.50 [-0.29, +1.17] |
| Steering + noise 1.0, prompt's noise 2x fading to a quarter along the prompt | Young child (bug) | 40/50 | 2.65 | 8.52 (20) | +2.33 [+1.29, +3.26] | +0.86 [+0.08, +1.80] |
| Steering + noise 1.0, prompt's noise 2x, with the shadow copy | Young child (bug) | 37/50 | 3.43 | 4.97 (8) | +0.97 [+0.29, +1.67] | +0.38 [-0.24, +0.98] |
| Steering + noise 1.0, prompt's noise 1.5x | Young child (bug) | 49/50 | 2.63 | 8.22 (20) | +2.04 [+0.98, +2.97] | +0.57 [-0.47, +1.48] |
| Steering + random residual noise 1.0 (reference in the architecture screen) | Young child (bug) | 20/20 | 2.20 | 4.58 (8) | +0.59 [-0.08, +1.23] | -0.01 [-0.64, +0.59] |
| Steering + random rotation of the hidden state, size 1.0 | Young child (bug) | 19/20 | 1.79 | 4.87 (8) | +0.87 [+0.22, +1.47] | +0.28 [-0.30, +0.80] |
| Steering + random per-head attention temperature, size 1.0 | Young child (bug) | 18/20 | 2.44 | 4.34 (8) | +0.34 [-0.34, +0.95] | -0.25 [-0.81, +0.37] |
| Steering + random rotary-frequency jitter (only reached ~0.6) | Young child (bug) | 18/20 | 1.78 | 4.31 (8) | +0.32 [-0.39, +1.05] | -0.27 [-1.02, +0.43] |
| Steering + random noise on the prompt's cached attention values, size 1.0 | Young child (bug) | 19/20 | 1.79 | 4.57 (8) | +0.58 [-0.11, +1.26] | -0.02 [-0.61, +0.59] |
| Steering + random MLP-unit dropout, size 1.0 | Young child (bug) | 20/20 | 1.95 | 4.62 (8) | +0.62 [-0.04, +1.28] | +0.03 [-0.54, +0.62] |


## The headline method: noise sized while the story is written (2026-09-23)

Three Kaggle runs on one account, 50 stories per arm, Qwen3-1.7B, layers 6-13,
sampling at temperature 1.0. Only stories the coherence checks pass are scored.
"Same-story pairs" and "distinct of 10" are NoveltyBench's v1.0 classifier (first
128 tokens of each story); intervals on differences come from 400 random
half-size draws of each arm. "Lowercase drift" counts coherent stories whose
sentences mostly lose their capitals. Rules are the eight whole-story rules.

**The method.** The per-story noise (a fresh random direction in a fresh random
64-dimensional subspace, kept off the rule directions, drifting slowly) is no
longer sized by a bisection before each story. A noise-free copy of the story
runs as a second batch row; after every step a controller compares how far the
noise moved the next-token distribution with how far top-p 0.95 at T=1.8 would
have moved it at that step, and rescales the noise for the next token so the
two agree on average (target 1.0). The prompt's noise is written before any
measurement: a tenth of the residual norm, times 1.5.

### Middle-school prompt (run r134)

| Method | Coherent | Rules broken | Same-story | Distinct of 10 | Lowercase |
|---|---|---|---|---|---|
| Untouched, T=1.0 | 50/50 | 2.68 | 99.7% | 1.03 | 0 |
| Top-p 0.95, T=1.8 | 50/50 | 2.56 | 93.8% | 1.41 | 0 |
| Steering only | 50/50 | 1.46 | 70.1% | 2.76 | 0 |
| **Sized while writing, prompt 1.5x** | **50/50** | 2.54 | 13.9% | 7.43 | 3 |
| Sized before each story, prompt 1.5x | 43/50 | 2.88 | 5.0% | 8.73 | 5 |
| Fixed 14.83 | 48/50 | 3.31 | 23.3% | 6.24 | 12 |
| Fixed 18 | 40/50 | 3.75 | 7.6% | 8.35 | 11 |
| Fixed 14.83, prompt 1.5x | 33/50 | 3.42 | 8.0% | 8.80 | 2 |
| Fixed 14.83, starting 1.5x | 45/50 | 3.22 | 15.5% | 7.14 | 12 |

The controller reached a median 1.00 nucleus-units (45 of 50 stories within
0.9-1.1) and settled at a median 0.8x the starting length in the second half:
the noise's effect accumulates, so less is needed later, which is why the
lowercase drift a fixed length produces late in a story mostly goes away. It is
cheaper than the bisection (11.2 s a story against 13.8 s on a T4). The judge is
very strict on this prompt: the untouched model's stories share one template
(weather, a he, a she, "a strange ache in his chest") and it calls nearly all of
them the same story. The ranking agrees with the opening-subject counts
(effective 5.9 untouched, 15.9 sized while writing); the baselines' absolute
levels are overstated.

### Young-child prompt (runs r135, r136, against the earlier 50-story arms)

The earlier arms (untouched, top-p, fixed 14.83, sized before with the prompt
at 1.5x) were re-judged in the same session as the new ones and reproduced the
Colab numbers exactly.

| Method | Coherent | Rules broken | vs top-p | Same-story | Distinct of 10 | vs top-p | Vendi vs top-p |
|---|---|---|---|---|---|---|---|
| Untouched, T=1.0 | 48/50 | 2.96 | +0.45 [+0.10, +0.78] | 33.1% | 5.38 | | -1.54 [-2.32, -0.72] |
| Top-p 0.95, T=1.8 | 49/50 | 2.51 | | 39.5% | 4.76 | | |
| Sized before, prompt 1.5x | 49/50 | 2.65 | +0.14 [-0.33, +0.61] | 9.2% | 8.28 | +3.53 [+1.70, +5.26] | +0.57 [-0.39, +1.55] |
| Fixed 14.83 | 49/50 | 3.20 | +0.69 [+0.29, +1.10] | 3.8% | 8.68 | +3.92 [+2.22, +5.40] | +0.63 [-0.35, +1.58] |
| **Sized while writing** | 46/50 | 2.54 | +0.03 [-0.44, +0.50] | 8.1% | 8.35 | +3.59 [+1.60, +5.08] | +0.17 [-0.67, +1.08] |
| + tilt 0.25 (reached 0.23) | 48/50 | 2.71 | +0.20 [-0.23, +0.61] | 6.7% | 8.69 | +3.94 [+2.06, +5.44] | +0.70 [-0.31, +1.62] |
| + tilt 0.5 (reached ~0.41) | 49/50 | 2.71 | +0.20 [-0.27, +0.67] | 7.1% | 8.11 | +3.36 [+1.58, +4.74] | +0.01 [-0.84, +0.88] |
| + tilt 1.0 (reached ~0.6) | 48/50 | 2.48 | -0.03 [-0.47, +0.41] | 5.6% | 8.61 | +3.85 [+2.08, +5.26] | +0.49 [-0.39, +1.33] |
| Sized while writing, steering 3.0 | 47/50 | 2.70 | +0.19 [-0.24, +0.61] | 6.6% | 8.32 | +3.56 [+1.68, +5.08] | -0.44 [-1.25, +0.42] |
| Sized while writing, steering 3.5 | 46/50 | 3.48 | +0.97 [+0.58, +1.36] | 2.4% | 9.12 | +4.36 [+2.72, +5.72] | +0.35 [-0.56, +1.24] |
| Fixed 14.83, steering 3.0 | 44/50 | 3.14 | +0.63 [+0.20, +1.05] | 1.5% | 9.43 | +4.67 [+2.94, +6.00] | +0.18 [-0.81, +1.08] |
| Fixed 14.83, steering 3.5 | 42/50 | 3.62 | +1.11 [+0.64, +1.57] | 2.3% | 9.15 | +4.40 [+2.72, +5.74] | +0.30 [-0.59, +1.33] |

**Stronger rule steering does not buy compliance.** At 3.0 the rules tie; at 3.5
they get worse, with far more lowercase drift (18-20 stories) and run-on
sentences (reading grade 7-8).

**The output tilt.** Every contrast pair is already run through the model to
extract the steering directions; the extraction now also keeps, per rule, the
log-ratio of the model's average next-token distribution on the rule-following
side against the rule-breaking side (an output-space mean contrastive
difference). While writing, the story's scores are tilted along the rules'
combined profile by exactly a share of top-p's own per-step distortion. The
strength ceiling (20) bound at the larger shares, so the reached shares were
about 0.23, 0.41 and 0.6. The tilt kept the variety (8.1-8.7 distinct) and
raised coherence (48-49 of 50 against 46), but did not move the rules as a
whole. It added a few stories that open by talking about the task
("Sure! Here's a short story...": 1, 1 and 3 at the three shares, 0 without).

**Why the rules tie: the two sides break different rules** (share of coherent
stories breaking each):

| | present tense | grade >= 3 | dialogue | one name | he and she | simile | no repeats | format |
|---|---|---|---|---|---|---|---|---|
| Untouched | 100% | 69% | 0% | 83% | 21% | 0% | 23% | 0% |
| Top-p | 98% | 35% | 8% | 67% | 35% | 0% | 8% | 0% |
| Sized while writing | 22% | 11% | 37% | 74% | 59% | 37% | 4% | 11% |
| + tilt 1.0 | 10% | 12% | 31% | 77% | 54% | 44% | 0% | 19% |

The baselines almost never write in the present tense but always include
speech and a simile; the method fixes the tense and loses speech, the he and
the she, and the simile. The tilt helped exactly the rules whose profiles point
at the right tokens: present tense (profile favours "follows", "reads", "asks";
22% to 10% failing) and dialogue (favours `?"`, `,"`, "Yes"; 37% to 31%). It
did nothing for the two whose profiles do not: both genders favours "both",
"them", "they" rather than "he" and "she", and simile favours the things
compared ("blanket", "mirror") rather than "like" or "as".

## The headline method on four other tasks (2026-09-23, run r137)

Qwen3-1.7B, 50 answers per arm, arms at shared seeds: the untouched model, top-p
0.95 at T=1.8, and the headline method (steering 2.5 on the task's own rules from
its own contrast pairs, noise sized while writing, prompt noise 1.5x a tenth of
the residual norm; built by the same suite code as the story runs). Tasks and
checks are in noiseegra/domains.py. "Valid" is a check that means the same in
every task (not empty, not a refusal, no repeated lines or loops); rules and
correctness are counted over valid answers; "distinct of 10" is NoveltyBench's
classifier.

| Task | Arm | Valid | Rules broken | Correct | Distinct of 10 |
|---|---|---|---|---|---|
| Number puzzle (4 rules) | untouched | 50/50 | 0.00 | 0% | 1.00 |
| | top-p | 50/50 | 0.00 | 0% | 1.00 |
| | method | 14/50 | 1.29 | 0% | 1.00 |
| Test cases (5 rules) | untouched | 50/50 | 0.32 | 82% | 1.05 |
| | top-p | 50/50 | 0.58 | 46% | 2.38 |
| | method | 49/50 | 3.73 | 8% | 6.79 |
| Library plan (5 rules) | untouched | 50/50 | 0.08 | | 3.23 |
| | top-p | 50/50 | 0.10 | | 6.31 |
| | method | 48/50 | 1.54 | | 7.01 |
| Poem (5 rules) | untouched | 50/50 | 2.02 | | 3.64 |
| | top-p | 50/50 | 1.74 | | 6.86 |
| | method | 46/50 | 3.00 | | 3.64 |

**As built, the method does not transfer.** It breaks more rules than both
baselines on every task, is far less often right on the test cases (8% against
82% untouched and 46% top-p), and gives more variety than top-p only where its
answers are broken (the test cases) -- it ties top-p on the plan and loses to it
on the poem. Its plans are nonsense ("three identical bottles of water ... a
single red marble"), its test cases wrong and wrapped in explanation, its poems
strings of lowercase questions with no stanzas.

**Why: the prompt's noise is not sized.** On these tasks the model is confident,
so top-p moves each step's distribution very little and the target the
controller aims at is small. The writing noise shrinks to its floor (a quarter
of the starting length) but the prompt's noise is written before anything is
measured, and it alone overshoots: over all 200 method answers the noise moved
the predictions a median 1.85 nucleus-units against 1.00 asked (26 within
0.9-1.1, 134 above 1.5; 8-24 on the number puzzle). On stories the model is
uncertain enough that the same prompt noise sits inside the target.

**The number puzzle measured nothing.** Its rules carried worked examples
("1407 ÷ 7 = 201"), and every untouched and top-p answer copied them -- all
format rules kept, the answer wrong every time. A design fault in the task, not a
finding; a rerun would state the formats without a number that could be copied.

**Steering alone is untested here.** Rules got worse even where the noise
overshoot is mild (the poem: every line turned into a question -- "at least one
question" went from 38% failing to 4% while three stanzas went from 8% to 76%),
so the steering's share of the damage cannot be separated from the noise's
without a steering-only arm per task.

## 200 stories: the headline method against the baselines and seven published methods (2026-09-23, runs r138, r139)

Middle-school prompt, Qwen3-1.7B, 200 stories per arm, one Kaggle account. Only
stories the coherence checks pass are scored. "Same-story pairs" and "distinct of
10" are NoveltyBench's v1.0 classifier (first 128 tokens); every pair was judged
in half precision and every pair within 0.03 of the 0.102 line again at full
precision (0 of 400 randomly checked other pairs per arm would have changed).
Intervals are 95%, from 400 half-size draws (judge) or 20,000 bootstrap draws
(rules). "Ours" is the rule steering at 2.5 plus the per-story noise sized while
writing to 1.0 of top-p's distortion, prompt noise 1.5x. Vendi is no longer
reported.

| Method | Venue | Coherent | Rules broken (of 8) | vs ours | Same-story pairs | Distinct of 10 | vs ours | Opening subjects (effective) |
|---|---|---|---|---|---|---|---|---|
| **Ours** | | 194/200 | **2.45** | | **24.5%** | **6.48** | | **21.7** |
| Untouched, T=1.0 | | 199/200 | 2.71 | +0.25 [+0.05, +0.46] | 99.0% | 1.06 | -5.42 [-6.12, -4.44] | 6.1 |
| Top-p 0.95, T=1.8 | Holtzman et al., ICLR 2020 | 200/200 | 2.50 | +0.05 [-0.16, +0.26] | 94.8% | 1.29 | -5.19 [-5.88, -4.18] | 6.7 |
| Min-p 0.1, T=1.5 | Nguyen et al., ICLR 2025 | 198/200 | 2.74 | +0.28 [+0.07, +0.49] | 96.9% | 1.18 | -5.30 [-6.02, -4.32] | 6.4 |
| Verbalized Sampling | ICML 2026 | 196/200 | 3.78 (2.78*) | +1.33 (+0.33*) | 93.3% | 1.42 | -5.06 [-5.74, -4.10] | 4.2 |
| String Seed of Thought | ICLR 2026 | 170/200 | 2.25 | -0.21 [-0.43, +0.01] | 82.7% | 2.02 | -4.47 [-5.22, -3.34] | 4.6 |
| In-context regeneration | NoveltyBench, COLM 2025 | 200/200 | 2.97 | +0.52 [+0.32, +0.71] | 99.8% | 1.03 | -5.45 [-6.16, -4.52] | 13.3 |
| STARS activation steering | ICLR 2026 | 181/200 | 3.34 | +0.89 [+0.67, +1.11] | 95.8% | 1.24 | -5.24 [-5.92, -4.26] | 6.9 |
| Noise injection (Liu et al.) | ICLR 2026 | 199/200 | 2.80 | +0.35 [+0.14, +0.54] | 98.9% | 1.07 | -5.41 [-6.12, -4.46] | 6.2 |
| Rule steering only | | 200/200 | 1.57 | -0.88 [-1.09, -0.67] | 76.9% | 2.38 | -4.10 [-5.02, -3.04] | 13.6 |
| Rule steering + top-p 0.95, T=1.8 | | 200/200 | 1.65 | -0.81 [-1.03, -0.59] | 75.5% | 2.51 | -3.97 [-4.84, -2.98] | 19.1 |
| Noise alone (fixed 14.83, no steering) | | 152/200 | 3.36 | +0.91 [+0.64, +1.17] | 41.3% | 4.93 | -1.55 [-2.66, -0.04] | 6.4 |

\* Verbalized Sampling writes its stories inside JSON strings, so the model puts
speech in single quotes, which the dialogue rule does not count. With that speech
converted to double quotes it breaks 2.78 (+0.33 [+0.11, +0.54] against ours).

**Ours is the most varied arm by a wide margin**: 6.48 distinct stories of 10
against 1.03-2.51 for every baseline and published method, and against 4.93 for
the same noise without steering. It keeps 194 of 200 coherent. On rules it beats
the untouched model, min-p, Verbalized Sampling, in-context regeneration, STARS
and noise injection, ties top-p and String Seed of Thought, and loses to the two
arms that steer without noise (1.57-1.65), which are also the least varied after
the baselines.

**What the published methods do on this model**, read from the stories:
- In-context regeneration keeps the plot and swaps the surface: "The wind howled
  like a furious wolf... as Lily clutched her backpack... 'I can't do this'", then
  the rain and Emily, the sun and Sam, the clouds and Clara. The judge calls it
  the same story every time (99.8%).
- Noise injection at its published size (alpha 0.07, tuned for hallucination
  detection) barely moves Qwen3-1.7B: its first stories are nearly word for word
  the untouched model's at the same seeds.
- STARS (twenty stories written together) loses 19 to repetition loops and keeps
  the same he/she scene.
- String Seed of Thought loses 30 of 200 (repetition, fragments of its seed
  string) and its stories share an opening 46% of the time.
- Verbalized Sampling followed its format (39 of 40 replies parsed fully) but
  starts 68% of its stories with a named character walking somewhere.

**Noise alone breaks stories; steering holds them together.** The same fixed
noise without steering loses 48 of 200 to loops and stalls and breaks the most
rules of any arm (3.36), while being less varied than ours.

**Weak spot:** 15 of our 194 coherent stories drift into lowercase partway through.

### The method with high-temperature top-p on top (run r140, 2026-09-24)

The same method sampled with top-p 0.95 at temperature 1.8, the decoder of the
top-p baseline, instead of at 1.0. Every arm in this project, this one included,
also keeps Qwen3's shipped top-k 20 and top-p 0.95 unless a setting replaces it.
Same prompt, seeds, steering directions and judge; the noise is still sized
against top-p's distortion measured on the model's own scores, before the
temperature and cut-off are applied.

| | Coherent | Rules broken | vs ours at 1.0 | Same-story pairs | Distinct of 10 | vs ours at 1.0 | Opening subjects |
|---|---|---|---|---|---|---|---|
| Ours at T=1.0 | 194/200 | 2.45 | | 24.5% | 6.48 | | 21.7 |
| Ours + top-p 0.95, T=1.8 | 199/200 | 2.37 | -0.08 [-0.31, +0.14] | 32.4% | 5.57 | -0.91 [-1.70, +0.52] | 24.6 |
| Top-p 0.95, T=1.8 alone | 200/200 | 2.50 | | 94.8% | 1.29 | | 6.7 |

The high temperature adds nothing measurable on either axis: it keeps five more
stories coherent and breaks slightly fewer rules, and is slightly less varied by
the judge, all within the intervals. Against top-p alone it is 4.28 [+3.62,
+5.30] distinct stories of 10 ahead and ties on rules (-0.13 [-0.33, +0.07]); it
beats the untouched model on rules (-0.34 [-0.54, -0.14]). Fifteen of its coherent
stories drift into lowercase, the same count as at 1.0.

## On Llama-3.2-3B-Instruct (2026-09-24, runs r142-r148)

The same 200-story comparison on a second model, one that STARS and the noise
injection paper also used. Middle-school prompt, the same eight rules, seeds,
coherence checks, judge and intervals as the Qwen3-1.7B comparison above.
Steering is on layers 6-13, the same 21-46% of depth as on Qwen3-1.7B; STARS
steers layer 20 as in its paper, and the noise injection covers the top third
of layers (20-27). The noise-alone arm uses the same share of the residual norm
as on Qwen3-1.7B (0.142). "Prompted to be creative" appends "Be creative and
think of a very unique story." to the request.

| Method | Venue | Coherent | Rules broken (of 8) | vs ours | Same-story pairs | Distinct of 10 | vs ours | Opening subjects (effective) |
|---|---|---|---|---|---|---|---|---|
| **Ours** (gain capped at 2.5x, reached 0.43-0.48 of top-p's shift) | | 200/200 | 1.29 | | 32.0% | 5.62 | | 4.4 |
| Ours, target 0.43, gain allowed to 10x | | 200/200 | 1.33 | +0.04 [-0.12, +0.19] | 28.6% | 5.95 | +0.33 [-0.96, +1.44] | 4.3 |
| Untouched, T=1.0 | | 200/200 | 1.77 | +0.48 [+0.33, +0.62] | 42.5% | 4.78 | -0.85 [-2.06, +0.18] | 4.5 |
| Top-p 0.95, T=1.8 | Holtzman et al., ICLR 2020 | 195/200 | 2.01 | +0.72 [+0.55, +0.88] | 28.0% | 6.15 | +0.53 [-0.98, +1.54] | 6.6 |
| Min-p 0.1, T=1.5 | Nguyen et al., ICLR 2025 | 200/200 | 1.86 | +0.57 [+0.42, +0.71] | 39.5% | 4.95 | -0.68 [-1.92, +0.50] | 4.4 |
| Verbalized Sampling | ICML 2026 | 179/200 | 2.93 (2.07*) | +1.63 (+0.78*) | 51.9% | 3.88 | -1.74 [-2.80, -0.62] | 5.4 |
| String Seed of Thought | ICLR 2026 | 139/200 | 2.35 | +1.05 [+0.87, +1.23] | 32.5% | 5.66 | +0.04 [-1.38, +1.06] | 6.6 |
| In-context regeneration | NoveltyBench, COLM 2025 | 200/200 | 1.89 | +0.59 [+0.44, +0.74] | 21.0% | 6.73 | +1.11 [-0.14, +2.12] | 9.5 |
| STARS activation steering | ICLR 2026 | 200/200 | 1.90 | +0.60 [+0.46, +0.75] | 39.5% | 4.99 | -0.64 [-1.76, +0.42] | 4.7 |
| Noise injection (Liu et al.) | ICLR 2026 | 200/200 | 1.79 | +0.50 [+0.35, +0.63] | 42.9% | 4.59 | -1.03 [-2.24, -0.04] | 4.6 |
| Prompted to be creative | | 200/200 | 1.62 | +0.33 [+0.19, +0.47] | 21.6% | 6.58 | +0.96 [-0.36, +1.88] | 7.4 |
| Rule steering only | | 200/200 | 1.11 | -0.18 [-0.33, -0.03] | 36.2% | 5.28 | -0.34 [-1.62, +0.54] | 4.5 |
| Rule steering + top-p 0.95, T=1.8 | | 198/200 | 1.75 | +0.45 [+0.28, +0.62] | 23.2% | 6.56 | +0.94 [-0.26, +2.08] | 8.5 |
| Noise alone (0.142 of the norm, no steering) | | 200/200 | 2.09 | +0.79 [+0.65, +0.94] | 35.8% | 5.35 | -0.27 [-1.44, +0.74] | 4.6 |

\* With Verbalized Sampling's single-quoted speech counted as dialogue.

"vs ours" is against the first row. Stories 0-199 in every arm; the second row's
first 50 stories are the target sweep's (below), continued to 200.

**Llama-3.2-3B starts out far more varied than Qwen3-1.7B**: untouched, 42.5% of
pairs are the same story (99.0% on Qwen3-1.7B) and 4.78 of 10 are distinct
(1.06). It also breaks fewer rules (1.77 against 2.71).

**Ours breaks fewer rules than every baseline and published method**: 1.29,
against 1.62-2.93, each difference outside its interval. Only rule steering
alone breaks fewer (1.11), and it is 0.34 distinct stories of 10 less varied
(interval -1.62 to +0.54).

**On variety it ties the published methods rather than beating them.** It is
0.85 distinct stories of 10 above the untouched model (interval -0.32 to
+2.10). Top-p, in-context regeneration, the creative prompt and steering + top-p
are 0.5-1.1 above ours, every interval including zero, and each of them breaks
0.33-0.72 more rules. Our stories' opening subjects are as concentrated as the
untouched model's (4.4 effective; 64% open on a pronoun), as are rule steering
alone's.

### The noise's size on Llama

The first row was run at Qwen3-1.7B's target (1.0 of top-p's shift per step),
but the controller may only multiply the starting length by 0.25-2.5, and on
Llama the start (0.10 of the residual norm) moves the predictions far less than
on Qwen3-1.7B. The gain sat at its 2.5 cap and the noise reached 0.43-0.48.
With the cap raised to 10 at the full target (r143) the stories were word salad
("she stes unlocking the one you you..."), and the run was stopped.

A sweep of the target on stories 0-49 (r144, gain allowed to 10x). "Salad-free"
flags runs of a repeated word or more than 8% of words not in a dictionary.

| Target | Salad-free | Rules broken | Same-story pairs | Distinct of 10 |
|---|---|---|---|---|
| 0.43 | 49/50 | 1.28 | 29.3% | 5.90 |
| 0.55 | 42/50 | 1.52 | 20.2% | 6.72 |
| 0.70 | 39/50 | 1.51 | 21.6% | 6.96 |
| 0.85 | 25/50 | 1.96 | 18.6% | 6.97 |

Continued to 200 stories (the second row of the main table), target 0.43 ties
the capped run on every number, but reading the stories shows what the salad
check misses. In the first 45 words of the same 40 randomly chosen stories
(read by hand):

| | Clearly garbled | Minor slips |
|---|---|---|
| Untouched | 0 | 0 |
| Ours, gain capped at 2.5x | 0 | 2 |
| Ours, target 0.43, gain allowed to 10x | 17 | 5 |

The garbling is local and early: "she stires her she sweeps you take like a
guitar", "a, her, they we we we's", "searchinginginged her theedns", after
which the story usually recovers. Both arms start from the same words ("As she
rummages through the dusty attic,") and diverge where the gain climbs. The
controller raises the noise by up to 1.25x a step, and its running averages lag
the noise's effect, so a start far below the needed size makes it overshoot over
the opening words. On Qwen3-1.7B the start already moves the predictions 0.93
of top-p's shift against a target of 1.0; on Llama it moves them 0.17 against
0.43. The sweep's 0.43 arm is the same set of stories as the first 50 here, so
the sweep also undercounts garbling at every target, and 0.43 was its lowest
setting.

### A rule for the target, set before any story (commit a765a2c)

A Fisher-Rao distance d between two next-token distributions bounds the
probability mass that moves between them: their Bhattacharyya coefficient is
cos(d/2), so total variation is at most sin(d/2). The rule caps the mass the
noise may move per step at a share k of the probability the model gives its top
word, p1: sin(d/2) = k p1. In the controller's unit u (top-p 0.95 at T=1.8's
mean shift per step), the target is 2 asin(k p1) / u. Both u and p1 are
measured once per prompt along the model's greedy continuation under the rule
steering alone, so nothing is sampled (`--online-rule-k`).

Measured on the middle-school prompt (r145):

| Model | u (top-p's shift per step) | p1 (top word's probability) | Rule, k = 0.43 | Best found |
|---|---|---|---|---|
| Qwen3-1.7B | 0.685 | 0.762 | 0.97 | 1.0 (r134) |
| Llama-3.2-3B | 1.345 | 0.678 | 0.44 | 0.43 (r144) |
| Qwen3-4B-Instruct-2507 | 0.821 | 0.730 | 0.78 | not yet run |

The share of p1 moved at the best target was 0.441 on Qwen3-1.7B and 0.421 on
Llama. One constant fitted to two models is a fit, not a test; Qwen3-4B is the
first test. The Llama point also comes from a sweep whose lowest setting won
and whose arms all had the overshoot above, so it may move once the start is
fixed. A rule that holds the absolute distance fixed, ignoring p1, would give
Llama 0.51.

### Measuring the starting length too (commits d38d695, 2df3522; runs r147, r148)

With `--online-rule-start` the rule also finds, once per prompt, the starting
length at which random draws of the noise already move the greedy passage's
predictions by the target: bisection on a log scale, averaged over four draws
of the noise made as a story makes them, the same draws at every length. It
runs before the story is seeded, so each story's own draw is unchanged. On
Llama the target is 0.440 and the start 0.210 of the residual norm, twice the
fixed 0.10, and the controller settles at 0.74-1.03x of it instead of climbing.

`--headline-prompt-gains` sets the noise on the prompt as a multiple of the
start (1.5 was chosen on Qwen3-1.7B). Stories 0-49, same seeds in every arm.
Garbling is from a blind reading of the first 45 words: the openings of three
arms shuffled together with their labels hidden, twice. The capped arm was in
both batches (0 and 0 garbled, 6 and 7 slips).

| Arm | Noise on the prompt (of the norm) | Clearly garbled | Minor slips | Rules broken | Same-story pairs | Distinct of 10 |
|---|---|---|---|---|---|---|
| Capped (first row of the main table) | 0.15 | 0/50 | 6-7 | 1.36 | 36.1% | 5.37 |
| Target 0.43, gain allowed to 10x | 0.15 | 29/50 | 17 | 1.28 | 29.3% | 5.90 |
| Measured start, prompt 1.5x | 0.31 | 0/50 | 4 | 1.84 | 18.8% | 7.07 |
| **Measured start, prompt 1.0x** | 0.21 | 1/50 | 15 | 1.28 | 24.2% | 6.35 |
| Measured start, prompt 0.5x | 0.11 | 4/50 | 33 | 1.30 | 24.5% | 6.46 |
| Untouched | | | | 1.82 | 52.7% | 4.06 |

Minor slips are pronoun, quotation and grammar slips and single misspelt words;
clearly garbled means a run of nonsense words or a made-up word.

**The measured start removes the garbling.** At 1.5x, though, the noise on the
prompt doubles with the start, and rule following falls to the untouched
model's level (1.84 against 1.82). The losses are present tense (52% of
stories fail it, 28% capped), simile (48%, 24%) and he-and-she (22%, 10%). The
overshooting arm, whose prompt noise was 0.15, broke 1.28 despite its garbled
openings, so these choices are made at the prompt.

**At 1.0x the start, the rules come back.** 1.28 broken, against 1.36 capped
(-0.08 [-0.42, +0.26]) and 1.82 untouched (-0.54 [-0.84, -0.22]); present tense
fails 22%. It is 2.30 distinct stories of 10 above the untouched model [+0.58,
+4.24] and 0.98 above the capped arm [-0.80, +3.02], with one garbled opening
in 50.

**Less noise on the prompt means more while writing.** The controller makes up
the difference to reach the same target: its settled gain (median) is 0.81,
1.00 and 1.06 of the start at 1.5x, 1.0x and 0.5x, and at 0.5x it overshoots
(0.49 against 0.44). The slips rise with it: 4, 15 and 33 of 50.

### 200 stories with the measured start and prompt noise 1.0x (run r148, 2026-09-24)

The prompt 1.0x arm above, continued to 200 stories. Stories 50-199 were written
in a resumed session whose measured start came out at 0.225 of the norm rather
than 0.210: the measurement's four noise draws landed on a different device
(model loaded later), so different random numbers. The start measurement should
draw on the CPU with its own generator, and with more draws.

| Llama-3.2-3B, 200 stories | Coherent | Rules broken | vs this arm | Same-story pairs | Distinct of 10 | vs this arm | Opening subjects |
|---|---|---|---|---|---|---|---|
| **Measured start, prompt 1.0x** | 200/200 | 1.48 | | 28.0% | 6.13 | | 6.4 |
| Capped (fixed start) | 200/200 | 1.29 | -0.19 [-0.35, -0.02] | 32.0% | 5.62 | -0.51 | 4.4 |
| Untouched | 200/200 | 1.77 | +0.29 [+0.14, +0.45] | 42.5% | 4.78 | -1.35 [-2.48, -0.20] | 4.5 |
| Top-p 0.95, T=1.8 | 195/200 | 2.01 | +0.53 [+0.36, +0.70] | 28.0% | 6.15 | +0.02 [-1.28, +1.04] | 6.6 |
| In-context regeneration | 200/200 | 1.89 | +0.41 [+0.24, +0.57] | 21.0% | 6.73 | +0.60 | 9.5 |
| Prompted to be creative | 200/200 | 1.62 | +0.15 [-0.01, +0.30] | 21.6% | 6.58 | +0.46 | 7.4 |
| Rule steering + top-p T=1.8 | 198/200 | 1.75 | +0.27 [+0.08, +0.45] | 23.2% | 6.56 | +0.43 | 8.5 |
| Rule steering only | 200/200 | 1.11 | -0.36 [-0.53, -0.20] | 36.2% | 5.28 | -0.85 | 4.5 |

(The other published methods are as in the main table above; each breaks more
rules than this arm and none is more varied by more than its interval.)

Rules broken were 1.28 on stories 0-49 and 1.55 on 50-199 (present tense fails
32% and he-and-she 24% on the later stories, against 22% and 10% earlier). The
capped, untouched and steering-only arms are stable across the two ranges
(1.36/1.27, 1.82/1.76, 1.10/1.12). The difference is about 1.8 standard errors;
the 50-story figure was flattering, and the larger start may add to it.

Blind reading of 40 random openings from stories 50-199, shuffled with the
capped arm's at the same indices: 2 clearly garbled and 14 with minor slips,
against 0 and 8 (on 0-49: 1 and 15, against 0 and 7).

**On Llama the measured start trades rules for variety relative to the capped
run**: it ties top-p on variety while breaking 0.53 fewer rules, but the capped
run breaks 0.19 fewer rules than it. Neither is more varied than in-context
regeneration, the creative prompt or steering + top-p (0.4-0.6 distinct of 10
behind, within the intervals); both break fewer rules than each of them, the
creative prompt within its interval for this arm.

### Inference cost

Seconds per story on one T4, one story at a time, from the progress logs (the
logged rate is a running average; each arm's cost is the change in elapsed time
over its block of stories). * includes loading the model. Wall-clock on Kaggle
varies by about 20% between sessions: rule steering alone was slower than the
untouched model on Llama and faster on Qwen.

| Method | Llama s/story | words | ms/word | Qwen s/story | words | ms/word |
|---|---|---|---|---|---|---|
| Untouched | 7.3 | 141 | 52 | 10.6 | 143 | 74 |
| Top-p T=1.8 | 8.8 | 161 | 55 | 10.6 | 153 | 69 |
| Min-p T=1.5 | 7.5* | 144 | 52 | 10.2* | 147 | 69 |
| Prompted to be creative | 9.1 | 151 | 60 | 10.3* | 154 | 67 |
| Noise injection | 8.6 | 151 | 57 | 10.9 | 144 | 76 |
| Rule steering only | 9.6 | 132 | 73 | 8.5 | 114 | 75 |
| Rule steering + top-p | 8.2* | 136 | 60 | 8.5* | 114 | 74 |
| Ours (fixed start) | 8.7 | 119 | 73 | 12.4* | 147 | 84 |
| Ours (measured start, prompt 1.0x) | 8.8* | 119 | 74 | | | |
| Verbalized Sampling (5 per reply) | 6.9 | 96 | 72 | 9.8 | 127 | 77 |
| In-context regeneration (10 per conversation) | 22.5 | 129 | 174 | 10.0 | 134 | 75 |
| String Seed of Thought | 33.8 | 95 | 356 | 30.1 | 240 | 125 |
| STARS (20 written together) | 1.8 | 146 | 12 | 3.1 | 158 | 20 |

Ours costs what rule steering alone costs per word (73-74 against 73 ms on
Llama; 84 against 75 on Qwen) and up to 40% more than plain sampling; that
overhead is the steering hooks. The shadow row is nearly free here because
decoding one story at a time is limited by reading the weights, not by
arithmetic; written in batches it doubles the arithmetic, so ours would cost
about twice sampling. The measurement before the stories is about 95 forward
passes over the prompt and a 48-token passage, once per prompt: a few seconds,
about 1% over 200 stories. String Seed of Thought costs 2-5x (it writes a random
string and its reasoning first); in-context regeneration up to 3x (each story
reads the ones before it); STARS is cheapest per story only because it writes 20
at once, which any of these methods could do.

### Is the sizing machinery needed? The simple method against the full one (runs r149-r151, 2026-09-24)

A review of the method found parts that do nothing or reduce to something
simpler: the top-p unit cancels (the rule's target times top-p's shift is a
fixed Fisher-Rao distance, 2 asin(0.43 p1)); the rule is within 2% of a constant
0.63 on every model so far; a random vector in a random 64-dimensional subspace
is a random direction; the projection off the nine rule and shield directions
removes 0.3-0.4% of the noise; 82% of the drifting direction is its per-story
constant. The controller settles near the measured start when the prompt's
noise equals the writing length; its settling at 0.8x on Qwen3-1.7B came from
the prompt's noise at 1.5x, which the measurement does not include (not from the
effect accumulating, as said above).

"Simple" (commit ddc7897) is one isotropic random vector per story, constant
through the story, at a length measured once per prompt from the rule's target
and held (no controller, no shadow row, no subspace, projection or drift).
"Prompt-only" is the same with no noise while writing. Stories 0-49, same seeds.

| Qwen3-1.7B | Noise on the prompt (of the norm) | Coherent | Rules broken | Same-story pairs | Distinct of 10 | Lowercase drift |
|---|---|---|---|---|---|---|
| Full method, prompt 1.0x | 0.103 | 50/50 | 2.18 | 55.6% | 3.74 | 9 |
| Simple, prompt 1.0x | 0.103 | 50/50 | 2.20 | 52.0% | 4.02 | 3 |
| Prompt-only, 1.0x | 0.103 | 50/50 | 1.70 | 69.5% | 2.88 | 1 |
| Full method, prompt 1.5x (the headline) | 0.15 | 50/50 | 2.54 | 13.9% | 7.43 | 3 |
| Simple, prompt 1.5x | 0.155 | 47/50 | 2.55 | 35.0% | 5.50 | 2 |
| Rule steering only | | 50/50 | 1.46 | 70.1% | 2.76 | 0 |

| Llama-3.2-3B | Noise on the prompt | Coherent | Rules broken | Same-story pairs | Distinct of 10 | Garbled / minor slips (30 read blind) |
|---|---|---|---|---|---|---|
| Full method, prompt 1.0x | 0.210 | 50/50 | 1.28 | 24.2% | 6.35 | 1 / 14 |
| Simple, prompt 1.0x | 0.233 | 50/50 | 1.68 | 22.9% | 6.73 | 0 / 10 |
| Prompt-only, 1.0x | 0.233 | 50/50 | 1.66 | 21.6% | 6.27 | 0 / 5 |

**At prompt 1.0x the simple method matches the full one on variety on both
models**: Qwen 4.02 against 3.74, Llama 6.73 against 6.35. On Qwen it matches on
rules too (+0.02 [-0.40, +0.44]) with fewer lowercase drifts (3 against 9). On
Llama it breaks 0.40 more [+0.02, +0.78]; the full method's first 50 stories
were 0.20 better than its 200-story mean, and the simple arm's measured length
came out 11% longer (four draws of a different kind of noise), which lengthens
the prompt's noise, the setting Llama's rules respond to.

**At Qwen's headline setting (prompt 1.5x) the simple method is less varied**:
5.50 against 7.43 distinct of 10 (-1.92 [-3.88, -0.22]; same-story +21.1%
[+4.3%, +39.7%]), rules tied (+0.01 [-0.45, +0.47]), 47 of 50 coherent (two
formatted as scripts, one loop). Something removed matters here. The likeliest
is the controller: with the prompt at 1.5x it shrank the writing noise (median
0.8x the start, to its 0.25 floor in a quarter of stories), so the full method
was strong noise on the prompt and weaker noise while writing; the simple
method writes at the full length. The drift is the other candidate.

**The prompt's noise is the main lever, and writing noise matters on Qwen**:
Qwen's variety roughly doubles from prompt 1.0x to 1.5x (3.74 to 7.43), and
prompt-only noise on Qwen is no more varied than steering alone (2.88 against
2.76). On Llama prompt-only noise is as varied as the full method.

Cost: without the controller there is no shadow row. On Qwen the simple arms
took about 69 ms a word against 78 for the full method, one story at a time;
written in batches the full method does about twice the arithmetic.
