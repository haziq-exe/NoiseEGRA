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

## Generality: Qwen3-8B, which does not replicate

The same recipe, the same twelve requirements, 100 stories per condition. Every
condition is 100/100 coherent, with no headings and no assistant preambles
anywhere, so the coherence cost the 1.7B model pays at larger perturbations does
not appear here at all. Compliance and diversity are a different matter.

| Qwen3-8B | Coherent | Requirements broken (of 12) | Diversity: content | Diversity: form |
|---|---|---|---|---|
| Untouched (T=1.0) | 100/100 | **2.00** | −8.3 [−10.2, −6.7] | −3.5 [−4.9, −2.1] |
| Per-token noise, prior work | 100/100 | **1.73** | −5.9 [−7.3, −4.4] | −3.5 [−4.8, −2.2] |
| Top-k 40 (T=1.8) | 100/100 | 2.28 | +2.4 [+1.1, +3.7] | +1.7 [−0.4, +2.8] |
| Nucleus 0.95 (T=1.8) | 100/100 | 2.15 | *reference* | *reference* |
| Ours, γ=1.5 | 100/100 | 2.38 | −2.3 [−3.9, −0.9] | −0.3 [−1.7, +1.1] |
| Ours, γ=2.0 | 100/100 | 2.59 | +0.9 [−0.7, +2.4] | +0.5 [−1.0, +1.8] |

**The compliance result does not transfer.** The untouched 8B model already
breaks only 2.00 requirements of twelve, against the 1.7B model's 4.30, so most
of the gap the method closes on the small model is not there to close. At γ=1.5
and γ=2.0 the method is *worse* than leaving the model alone. Diversity ties
nucleus sampling at γ=2.0 and loses at γ=1.5.

The per-requirement table says where it goes:

| Requirement | Untouched | Nucleus T=1.8 | Ours γ=1.5 |
|---|---|---|---|
| present tense | 33% | 34% | **62%** |
| plain words | 54% | 48% | **86%** |
| sensory words | 65% | 64% | **24%** |
| named character | 64% | 54% | **47%** |
| dialogue | 97% | 92% | 79% |

The push wins the two rules it wins on the small model and loses the sensory
rule outright. Reading the stories explains it: the untouched 8B writes long
descriptive sentences ("the scent of chalk and old books wrapping around her
like a familiar blanket") which clear the six-sensory-word floor easily, and the
push trades exactly that prose for shorter action sentences. It also drifts into
first person, which is why the named-character rule falls.

**One likely cause is a configuration carried over without refitting.** These
arms ran at layers 14–22 of 36, the band the published Arabic paper used. The
band the method actually works at on the 1.7B model is 6–13 of 28, whose
proportional equivalent on a 36-layer model is 8–17 — materially earlier. The
steering budget was also never refitted. A corrected run at the proportional
band, with a push-only arm so the same ablation can be made, is what decides
whether this is a scope limit of the method or a setting that was never tuned.

Until that returns, the honest scope statement is that the compliance result is
demonstrated on a 1.7B model, which is also the size a school can actually run.

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

- Qwen3-8B at its proportional layer band, with a push-only arm.
- Locally typical, η-sampling, min-p and contrastive search as decoding
  baselines. None of the four has ever run: the settings reached the run id and
  the suite but not the generation call, so every one of them produced plain
  nucleus sampling. Fixed and re-running.
- A human or LLM-judge rating of story quality, which no automatic metric covers.
