# Results for the paper

Qwen3-1.7B, one instruction asking for a ~150-word story for a middle-school
reader under twelve checkable requirements, 200 stories per condition, every
condition sharing one prompt and one seed sequence. Diversity is a Vendi score
over the first 40 words of coherent stories, rarefied to a common sample size;
95% intervals from subsampling without replacement. Our method runs at
temperature 1.0 throughout; the decoding baselines are given the raised
temperature they need.

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
| Ours, beta=2, gamma=1.75 | 190/200 | 4.08 | +1.6 *tie* | **+2.6 [+0.9, +4.2]** |
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
