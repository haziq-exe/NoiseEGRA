#!/usr/bin/env python
"""Run the orthogonal-constraint-steering ablation.

    # once per model
    python scripts/build_steering_vectors.py --model Fanar

    # then
    python scripts/run_orthosteer_experiment.py --model Fanar --suite core --num-stories 50

Suites
------
``method`` just the proposed method, a single condition -- for a quick test run
           that does not regenerate the Baseline and L-Res references.
``noise``  Baseline plus the four steering variants -- no noise, noise with the
           constraint directions removed, ordinary noise, and noise confined to the
           constraint directions. This is the ablation itself.
``core``   the main comparison: Baseline, published L-Res (isotropic noise),
           steering with no noise, and steering + {orth, iso, para} noise.
           ``para`` is the destructive control that confines all the noise energy
           to the constraint subspace.
``ortho``  steering + orthogonal noise under each orthogonalisation of the
           steering vectors: none (naive sum), Gram-Schmidt, Loewdin.
``offset`` steering plus a per-story constant offset, swept over gamma, drawn both
           with and without the constraint subspace removed. This is the arm where
           the projection is expected to matter.
``alpha``  noise-strength sweep at fixed steering strength.
``beta``   systematically relax the constraint pressure: a sweep over beta.
``loo``    leave-one-out over the constraints, to see which one carries the effect
           and whether they interfere.
``all``    every suite above.

Everything is written under ``--output-dir`` in the same layout the rest of the
repo uses: one CSV of stories per run, per-run RESULTS/*.txt (creativity +
published constraint report + the exact LLM-free constraint report), and a
combined <model>_RESULTS.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from noiseegra import prompts  # noqa: E402
from noiseegra.defaults import MODEL_HF_IDS, MODEL_LAYER_RANGES, RMS_ALPHA  # noqa: E402
from noiseegra.RMS_std import RMSCalibrator  # noqa: E402
from noiseegra.setup_experiment import make_specs, run_story_experiments  # noqa: E402
from noiseegra.steering_vectors import SteeringVectorSet  # noqa: E402
from noiseegra.subspace import ConstraintSpec, SteeringPlan  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_steering_vectors import build_model  # noqa: E402

DEFAULT_CONSTRAINTS = ["closure", "present_tense", "simple_register"]
# The closure direction ramps up: push the model to wrap up harder the longer the
# story has already run. The other two are properties of every sentence, so they
# stay constant.
DEFAULT_SCHEDULES = {}   # flat for every direction unless a caller says otherwise


def make_plan(
    vectors: SteeringVectorSet,
    layers,
    names,
    *,
    beta,
    rms_scale,
    orthogonalize="lowdin",
    noise_mode="orth",
    noise_alpha=RMS_ALPHA,
    protect_rank=0,
    schedules=None,
    horizon=200,
    steer_prefill=False,
    noise_norm_match="energy",
    noise_schedule="constant",
    offset_gamma=0.0,
    offset_mode="none",
    offset_norm="energy",
    offset_basis_kind="step",
    offset_prefill=False,
    offset_basis=None,
    offset_decode=True,
    amplify_lambda=1.0,
    amplify_prefill=False,
    amplify_basis=None,
    amplify_mean=None,
    noise_horizon=None,
    jitter_kappa=0.0,
    jitter_mode="none",
    jitter_draw="iso",
    steer_decode=True,
    steer_budget=None,
    steer_mode="constant",
    feedback_cap=0.1,
    targets=None,
    direction_source="extracted",
    gate_threshold=0.0,
    gate_level="none",
) -> SteeringPlan:
    schedules = schedules or DEFAULT_SCHEDULES
    betas = beta if isinstance(beta, dict) else {n: float(beta) for n in names}
    specs = [
        ConstraintSpec(n, beta=betas.get(n, 0.0), schedule=schedules.get(n, "constant"))
        for n in names
    ]
    extra = vectors.protect_extra(names, protect_rank) if protect_rank > 0 else None
    return SteeringPlan.build(
        vectors.vectors,
        layers,
        specs,
        rms_scale=rms_scale,
        orthogonalize=orthogonalize,
        noise_mode=noise_mode,
        noise_alpha=noise_alpha,
        noise_norm_match=noise_norm_match,
        noise_schedule=noise_schedule,
        horizon=horizon,
        steer_prefill=steer_prefill,
        protect_extra=extra,
        offset_gamma=offset_gamma,
        offset_mode=offset_mode,
        offset_norm=offset_norm,
        offset_basis_kind=offset_basis_kind,
        offset_prefill=offset_prefill,
        offset_basis=offset_basis,
        offset_decode=offset_decode,
        amplify_lambda=amplify_lambda,
        amplify_prefill=amplify_prefill,
        amplify_basis=amplify_basis,
        amplify_mean=amplify_mean,
        noise_horizon=noise_horizon,
        jitter_kappa=jitter_kappa,
        jitter_mode=jitter_mode,
        jitter_draw=jitter_draw,
        steer_decode=steer_decode,
        steer_budget=steer_budget,
        steer_mode=steer_mode,
        feedback_cap=feedback_cap,
        targets=targets,
        direction_source=direction_source,
        gate_threshold=gate_threshold,
        gate_level=gate_level,
    )


def build_suite(name, vectors, layers, names, rms_scale, args):
    """Return (list_of_spec_dicts, description) for one suite."""
    common = dict(
        vectors=vectors, layers=layers, names=names, rms_scale=rms_scale,
        protect_rank=args.protect_rank, horizon=args.horizon,
        steer_prefill=args.steer_prefill,
        direction_source=getattr(args, "direction_source", "extracted"),
    )
    offset_basis = getattr(args, "offset_basis", None)
    items = []

    if name == "baseline":
        # Unmodified generation and nothing else. The reference every other
        # condition is read against, and the first thing to run on a new model.
        # Returned before anything touches `vectors` or `rms_scale`, because a
        # baseline run has neither: it is the model under the prompt alone.
        return (["baseline"], "unmodified generation, no steering and no perturbation")

    if name == "sampling":
        # The decoding-parameter comparison from the published study: raising the
        # temperature and truncating the tail is the obvious way to buy diversity
        # without touching the representation, so it is the reference any
        # representation-level method has to beat. Same magnitudes as the paper.
        t = args.baseline_temperature
        arms = [{"mode": "baseline"}]
        if args.baseline_top_p is not None:
            arms.append({"mode": "baseline", "temperature": t,
                         "top_p": args.baseline_top_p})
        if args.baseline_top_k is not None:
            arms.append({"mode": "baseline", "temperature": t,
                         "top_k": args.baseline_top_k})
        return arms, (f"unmodified generation, plus sampling baselines at "
                      f"temperature {t:g}")

    if vectors is None:
        raise ValueError(f"suite {name!r} needs steering vectors")
    resid_std = args.alpha * rms_scale

    if name == "method":
        # Just the proposed method, one condition. For a quick smoke run that does
        # not regenerate the Baseline and L-Res references you already have.
        items = [{"plan": make_plan(beta=args.beta, noise_mode="orth",
                                    noise_alpha=args.alpha, **common)}]
        return items, "the proposed method only (steering + constraint-free noise)"

    # The four steering variants, differing only in where the noise is allowed to sit.
    noise_arms = [
        {"plan": make_plan(beta=args.beta, noise_mode=mode,
                           noise_alpha=0.0 if mode == "none" else args.alpha, **common)}
        for mode in ("none", "orth", "iso", "para")
    ]

    if name == "offset":
        # Per-story constant offsets. Unlike per-token noise these accumulate
        # linearly, so they actually change the story -- and a leak onto a
        # constraint direction biases that constraint for the whole story, which
        # is what makes the projection load-bearing.
        arms = [
            {"plan": make_plan(beta=args.beta, noise_mode="none", noise_alpha=0.0,
                               noise_schedule="constant", offset_gamma=0.0,
                               offset_mode="none", **common)},
        ]
        for g in args.gamma_sweep:
            for mode in ("orth", "free"):
                arms.append({"plan": make_plan(
                    beta=args.beta, noise_mode="none", noise_alpha=0.0,
                    offset_gamma=g, offset_mode=mode, offset_basis=offset_basis, **common)})
        return arms, (f"steering + per-story constant offsets at gamma {list(args.gamma_sweep)}, "
                      "each drawn with and without the constraint subspace removed")

    if name == "story":
        # Per-story constant offsets, drawn from the directions along which the
        # model's own stories already differ from one another, and projected off
        # the constraint directions. One vector per story, held for the whole
        # generation, so it shifts *which* story gets written rather than jittering
        # every token -- and because nothing is redrawn mid-story, nothing
        # compounds through the KV cache.
        kind = getattr(args, "offset_basis_kind", "step")
        arms = []
        for g in args.gamma_sweep:
            arms.append({"plan": make_plan(
                beta=args.beta, noise_mode="none", noise_alpha=0.0,
                offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, offset_prefill=False, **common)})
            arms.append({"plan": make_plan(
                beta=args.beta, noise_mode="none", noise_alpha=0.0,
                offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, offset_prefill=True, **common)})
        return arms, (f"steering + per-story offsets at gamma {list(args.gamma_sweep)} "
                      f"drawn from the {kind}-level activation directions, each "
                      "applied from the first generated token and from the prompt")

    if name == "control":
        # Steering whose coefficient is the constraint error of the story so far.
        #
        # Steering doubles compliance on the monotone requirements and cuts the
        # counting ones from 23% to 6%. A count has no "more is better"
        # direction, so a constant coefficient sails past the target: pushing
        # "more dialogue" takes the exactly-two rule from 4% to 0%. Feedback on
        # activations does not fix it either -- it saturates at "quote-like
        # enough", which is what the contrast examples encode, not at "two
        # quotes".
        #
        # The generation loop can count. Each decode step the tokens produced so
        # far are decoded and the same checks that score the finished story are
        # run over them, giving a signed error per requirement. A requirement
        # inside its band gets a coefficient of zero and the model writes
        # unsteered.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", 3.0) or 3.0)
        gains = list(getattr(args, "control_betas", [1.0, 2.0]))
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15]))
        kind = getattr(args, "offset_basis_kind", "story")
        ctl = getattr(args, "controller", None)
        flat = {n: 1.0 for n in names}

        items = ["baseline"]
        # the constant arm at the same budget, which is what this has to beat
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        for g in gains:
            items.append({"plan": make_plan(
                beta={n: g for n in names}, steer_mode="error",
                control_state={}, controller=ctl, steer_budget=b,
                steer_prefill=False, **quiet, **base)})
        pick = gains[-1]
        for gamma in gammas:
            items.append({"plan": make_plan(
                beta={n: pick for n in names}, steer_mode="error",
                control_state={}, controller=ctl, steer_budget=b,
                offset_gamma=gamma, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, offset_prefill=True, offset_decode=False,
                steer_prefill=False, **quiet, **base)})
        return items, (
            f"steering whose coefficient is the constraint error of the story so "
            f"far, at gain {gains} and a total push of {b:g}, against the constant "
            f"arm at the same push, with the per-story perturbation on top at "
            f"gamma {gammas}")

    if name == "closure":
        # Can a counting constraint be steered if the *controller* counts?
        #
        # Steering more than doubles compliance on the monotone constraints -- 25%
        # to 62% -- and cuts the counting ones from 23% to 6%. Word count, sentence
        # count and exactly-two-quoted-lines are all about when to stop, and a
        # coefficient that is the same at token five and token ninety cannot say
        # that. The model has no representation of "how many words so far". The
        # generation loop does.
        #
        # So: a direction meaning "bring it to an end", on a schedule that is
        # silent while the story is inside its budget and presses harder the longer
        # it runs over. A 50-to-65-word story is about 75 tokens, which is where
        # the horizon goes.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        base = {k: v for k, v in base.items() if k != "horizon"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", 3.0) or 3.0)
        h = int(getattr(args, "closure_horizon", 80))
        betas = list(getattr(args, "closure_betas", [2.0, 4.0]))
        others = [n for n in names if n != "closure"]
        if "closure" not in names:
            raise SystemExit("--suite closure needs 'closure' in --steer-vectors")

        items = ["baseline"]
        # the rest of the set as it stands, so the closure arms are read against it
        items.append({"plan": make_plan(
            beta={n: (1.0 if n in others else 0.0) for n in names},
            steer_budget=b, horizon=200, steer_prefill=False, **quiet, **base)})
        for cb in betas:
            # closure alone, to see what it does to the counting constraints
            items.append({"plan": make_plan(
                beta={n: (cb if n == "closure" else 0.0) for n in names},
                schedules={"closure": "tail"}, horizon=h,
                steer_prefill=False, **quiet, **base)})
            # and on top of the rest
            items.append({"plan": make_plan(
                beta={n: (cb if n == "closure" else 1.0) for n in names},
                schedules={"closure": "tail"}, horizon=h,
                steer_prefill=False, **quiet, **base)})
        return items, (
            f"a closure direction on a schedule that stays silent for {h} tokens "
            f"and then presses harder the longer the story runs, at beta {betas}, "
            "alone and with the other five directions")

    if name == "headtohead":
        # The head-to-head, now that repetition is scored.
        #
        # Steering buys compliance partly by breaking text, and until this round
        # nothing measured that: the baseline loops two stories in twenty-four,
        # steering five directions under a budget loops six, and a single constant
        # direction loops ten. Feedback steering loops three while keeping most of
        # the gain, which is what it was built to do -- a correction proportional
        # to the shortfall stops when the shortfall does, so it cannot drive a
        # story that is already compliant off a cliff.
        #
        # So: the two steering mechanisms against each other at matched strength,
        # each with and without the per-story perturbation, scored on thirteen
        # requirements including the one that catches looping.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", 3.0) or 3.0)
        fb = list(getattr(args, "feedback_betas", [0.5]))
        cap = float(getattr(args, "feedback_cap", 0.1))
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15]))
        kind = getattr(args, "offset_basis_kind", "story")
        tg = getattr(args, "targets", None)
        flat = {n: 1.0 for n in names}

        def arm(steer, gamma=None):
            extra = dict(offset_gamma=gamma, offset_mode="orth",
                         offset_basis=offset_basis, offset_basis_kind=kind,
                         offset_prefill=True, offset_decode=False) if gamma else {}
            return {"plan": make_plan(steer_prefill=False, **steer, **extra,
                                      **quiet, **base)}

        constant = dict(beta=flat, steer_budget=b)
        items = ["baseline"]
        items.append(arm(constant))
        for g in gammas:
            items.append(arm(constant, g))
        for gain in fb:
            feedback = dict(beta={n: gain for n in names}, steer_mode="feedback",
                            targets=tg, feedback_cap=cap)
            items.append(arm(feedback))
            for g in gammas:
                items.append(arm(feedback, g))
        return items, (
            f"constant steering at a fixed push of {b:g} against feedback steering "
            f"at gain {fb}, each alone and with the per-story perturbation at gamma "
            f"{gammas}, scored on thirteen requirements")

    if name == "assemble":
        # Both halves together, at the settings each was measured at.
        #
        # Steering under a fixed budget now buys 1.25 to 1.33 requirements of
        # twelve on Qwen3-1.7B, which is slack the diversity half can spend. Every
        # earlier attempt to add a perturbation started from a steering
        # configuration that was already losing, so the perturbation had to pay for
        # the steering's mistakes before it could buy anything.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", 3.0) or 3.0)
        kappa = float(getattr(args, "realloc_kappa", 0.6))
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15, 0.25]))
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}

        def offset(g, realloc):
            extra = dict(jitter_mode="gain", jitter_kappa=kappa) if realloc else {}
            return {"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_prefill=True, offset_decode=False, steer_prefill=False,
                **extra, **quiet, **base)}

        items = ["baseline"]
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="gain", jitter_kappa=kappa,
                                        steer_prefill=False, **quiet, **base)})
        for g in gammas:
            items.append(offset(g, False))
        for g in gammas[:2]:
            items.append(offset(g, True))
        return items, (
            f"steering at a fixed total push of {b:g}, with and without the budget "
            f"reallocated per story, and the per-story perturbation on top at gamma "
            f"{gammas}")

    if name == "feedback":
        # Steering that corrects this story rather than biasing every story.
        #
        # Constant steering adds one vector to all of them. That is why it buys
        # compliance by spending diversity: measured at 4.42 -> 3.88 requirements
        # broken and 3.61 -> 3.12 Vendi, forty stories shoved the same way end up
        # more alike. The correction here is a function of where the story already
        # sits on each constraint axis, so a story that already satisfies a
        # constraint receives nothing on that axis and two stories failing
        # different constraints are corrected in different directions -- verified
        # orthogonal, against cosine 1.0 for the constant version.
        #
        # The last arms put the per-story perturbation on top, because the point of
        # a steering component that does not homogenise is that the diversity
        # component no longer has to fight it.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        tg = getattr(args, "targets", None)
        gains = list(getattr(args, "feedback_betas", [0.5, 1.0]))
        cap = float(getattr(args, "feedback_cap", 0.1))
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15]))
        kind = getattr(args, "offset_basis_kind", "story")
        best = float(getattr(args, "combo_beta", 3.0))
        keep = dict(getattr(args, "keep_directions", {}))

        items = ["baseline"]
        # the constant reference, at the coefficient that worked
        if keep:
            items.append({"plan": make_plan(
                beta={n: keep.get(n, 0.0) * best for n in names},
                steer_prefill=False, **quiet, **base)})
        for g in gains:
            items.append({"plan": make_plan(
                beta={n: g for n in names}, steer_mode="feedback", targets=tg,
                feedback_cap=cap, steer_prefill=False, **quiet, **base)})
        pick = gains[-1]
        for gamma in gammas:
            items.append({"plan": make_plan(
                beta={n: pick for n in names}, steer_mode="feedback", targets=tg,
                feedback_cap=cap, offset_gamma=gamma, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_prefill=True, offset_decode=False,
                steer_prefill=False, **quiet, **base)})
        return items, (
            f"steering that corrects each story's own shortfall on every "
            f"constraint, at gain {gains} and capped at {cap:g} of the hidden "
            f"state, against the constant push, and with the per-story "
            f"perturbation on top at gamma {gammas}")

    if name == "budget":
        # Why steering more than one constraint stops working, and whether holding
        # the total push fixed repairs it.
        #
        # Summing k directions at coefficient beta gives a push of length
        # beta*sqrt(k)*rms, so "steer one more constraint" means "push harder" as a
        # side effect. Measured on Qwen3-8B: one direction at 3 is a push of 4.52
        # and helps; two directions at 3 is 6.39 and does not; one direction at 4.5
        # is 6.78 and breaks the text outright. The two-direction arm was never
        # compared against the one-direction arm at the same strength.
        #
        # With a budget the push is renormalised to a fixed length, so adding a
        # constraint redistributes it rather than enlarging it, and the number of
        # constraints stops being a strength knob.
        #
        # The last two arms are the point. Under a fixed budget the *allocation*
        # across constraints can be redrawn per story while the total stays
        # identical: every story gets the same amount of constraint pressure spent
        # differently. That is a perturbation that never leaves the constraint
        # subspace at all, so unlike every noise arm here it has no off-manifold
        # component to cost fluency.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        budgets = list(getattr(args, "budget_sweep", [2.0, 3.0, 4.5]))
        pick = float(getattr(args, "steer_budget", 3.0) or 3.0)
        kappa = float(getattr(args, "realloc_kappa", 0.6))
        h = int(getattr(args, "steer_horizon", 32) or 32)
        keep = dict(getattr(args, "keep_directions", {}))
        flat = {n: 1.0 for n in names}

        items = ["baseline"]
        # the broken reference: every direction at 1, summed as before
        items.append({"plan": make_plan(beta=flat, steer_prefill=False, **quiet, **base)})
        for b in budgets:
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            steer_prefill=False, **quiet, **base)})
        if keep:
            items.append({"plan": make_plan(beta={n: keep.get(n, 0.0) for n in names},
                                            steer_budget=pick, steer_prefill=False,
                                            **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=pick,
                                        jitter_mode="gain", jitter_kappa=kappa,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(
            beta=flat, steer_budget=pick, jitter_mode="gain", jitter_kappa=kappa,
            schedules={n: "cosine_decay" for n in names}, horizon=h,
            steer_prefill=False, **{k: v for k, v in base.items() if k != "horizon"},
            **quiet)})
        return items, (
            f"every direction summed as before, then held to a fixed total push of "
            f"{budgets}, then the same budget with the allocation redrawn per story "
            f"(spread {kappa:g}), with and without the push decaying over the first "
            f"{h} tokens")

    if name == "select":
        # Keep the directions that are measured to help, at the coefficient that
        # helps most, and drop the rest.
        #
        # The per-direction probe found that three of the five directions raise the
        # total violation count at both signs, and that the two that lower it were
        # both mis-set by the earlier "steer what fails" rule. `simple_register`
        # had been switched off entirely because its own requirement already passes
        # at 100%, and it turns out to be the single most useful direction in the
        # set -- not through its own requirement but through the ones it drags with
        # it, word count +21% and the spelled-number rule +29%. `varied_openers`
        # had been set to +1 when -3 is what helps.
        #
        # The lesson is that a direction's value is not what it was extracted for.
        # It is its measured effect on the whole requirement list, and that has to
        # be measured rather than assumed from the name on the contrast pairs.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        keep = dict(getattr(args, "keep_directions", {"simple_register": 1.0,
                                                      "varied_openers": -1.0}))
        mags = list(getattr(args, "tune_betas", [1.5, 3.0, 4.5, 6.0]))
        combo = float(getattr(args, "combo_beta", 3.0))
        gammas = list(getattr(args, "gamma_sweep", [0.15, 0.25]))
        kind = getattr(args, "offset_basis_kind", "story")
        offs = offset_basis

        items = ["baseline"]
        for target, sign in keep.items():
            for m in mags:
                betas = {n: (sign * m if n == target else 0.0) for n in names}
                items.append({"plan": make_plan(beta=betas, steer_prefill=False,
                                                **quiet, **base)})
        combo_betas = {n: keep.get(n, 0.0) * combo for n in names}
        items.append({"plan": make_plan(beta=combo_betas, steer_prefill=False,
                                        **quiet, **base)})
        # The whole point of fixing the steering was to put a perturbation on top
        # of a starting point better than the baseline rather than worse.
        for g in gammas:
            items.append({"plan": make_plan(
                beta=combo_betas, offset_gamma=g, offset_mode="orth",
                offset_basis=offs, offset_basis_kind=kind,
                offset_prefill=True, offset_decode=False,
                steer_prefill=False, **quiet, **base)})
        kept = ", ".join(f"{n} {'+' if v > 0 else '-'}" for n, v in keep.items())
        return items, (
            f"the directions measured to help ({kept}) swept over {mags}, the two "
            f"together at {combo:g}, and the per-story offset at the prompt on top "
            f"at gamma {gammas}")

    if name == "directions":
        # Does each extracted direction move its own requirement, and which way?
        #
        # Nothing so far has asked. Steering has been run as a block of five
        # directions at one coefficient, and it costs both compliance and
        # diversity: 4.65 requirements broken against a 4.55 baseline, with Vendi
        # 3.27 against 4.00. A block that loses on both cannot be fixed by tuning
        # what is added on top of it, and "the block does not work" does not say
        # which of the five is at fault, or whether any of them work.
        #
        # So: one direction at a time, pushed hard in both directions, scored on
        # the requirement that direction was extracted to serve. A direction that
        # moves its own requirement is usable and its sign and size can be tuned.
        # One that does not is a bad direction and no amount of tuning will help.
        # Then the block itself at three sizes, to see whether the failure is the
        # directions or the dose.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        probes = getattr(args, "probe_beta", [3.0])
        probes = [float(p) for p in (probes if isinstance(probes, (list, tuple)) else [probes])]
        block = list(getattr(args, "beta_sweep", [1.0, 2.0, 4.0]))
        nominal = args.beta if isinstance(args.beta, dict) else {n: args.beta for n in names}

        items = ["baseline"]
        for target in names:
            for probe in probes:
                for sign in (+1.0, -1.0):
                    betas = {n: (sign * probe if n == target else 0.0) for n in names}
                    items.append({"plan": make_plan(beta=betas, steer_prefill=False,
                                                    **quiet, **base)})
        for scale in block:
            betas = {n: scale * v for n, v in nominal.items()}
            items.append({"plan": make_plan(beta=betas, steer_prefill=False,
                                            **quiet, **base)})
        return items, (
            f"one direction at a time at beta +/-{probes}, scored on the "
            f"requirement it was extracted to serve, then the whole block at "
            f"{block} times the calibrated coefficients")

    if name == "pareto":
        # Round 2 said three things. Every useful arm costs about one requirement
        # of twelve, so the question is no longer "does a perturbation buy
        # diversity" but "what does a unit of diversity cost, and can the cost be
        # paid back somewhere else". The offset at the prompt bought the most per
        # requirement broken. And the steering every arm sits on top of was signed
        # wrongly, so every one of them inherited a starting point worse than no
        # steering at all.
        #
        # So: the two families that worked, swept over dose at the site that
        # worked, on top of steering whose signs come from which way the model
        # actually errs (--beta-calibration auto). The flat-noise arm is dropped --
        # it is degenerate at this alpha (159 words, 78 sentences) and generates
        # six times slower for it -- and `rotate` and `gain` are dropped as nulls.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        kind = getattr(args, "offset_basis_kind", "story")
        h = int(getattr(args, "noise_horizon", 24) or 24)
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        gammas = list(getattr(args, "gamma_sweep", [0.05, 0.1, 0.15, 0.25]))
        kappas = list(getattr(args, "kappa_sweep", [0.1, 0.15]))
        alphas = [a for a in getattr(args, "alpha_sweep", [0.4, 0.8]) if a > 0]

        items = ["baseline"]
        items.append({"plan": make_plan(beta=args.beta, steer_prefill=False,
                                        **quiet, **base)})
        items.append({"plan": make_plan(beta=args.beta, steer_prefill=True,
                                        steer_decode=False, **quiet, **base)})
        for g in gammas:
            items.append({"plan": make_plan(
                beta=args.beta, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_prefill=True, offset_decode=False,
                steer_prefill=False, **quiet, **base)})
        for k in kappas:
            items.append({"plan": make_plan(
                beta=args.beta, jitter_mode="perp", jitter_kappa=k,
                steer_prefill=True, steer_decode=False, **quiet, **base)})
        for a in alphas:
            items.append({"plan": make_plan(
                beta=args.beta, noise_mode="orth", noise_alpha=a,
                noise_schedule="cosine_decay", noise_horizon=h,
                steer_prefill=False, **base)})
        return items, (
            f"the dose curve at the prompt: per-story offset at gamma {gammas}, "
            f"f(S_c) sideways step at kappa {kappas}, and cosine-decayed per-token "
            f"noise at alpha {alphas} over {h} tokens, with both unperturbed siting "
            "controls")

    if name == "ablate":
        # Does projecting the constraint subspace out of the offset actually buy
        # anything? The round-1 measurement said the between-story axes put 8.2% of
        # their energy inside the 36-dimensional constraint span against 0.88% by
        # chance, which is what makes the projection look load-bearing. That is a
        # correlation. This is the test: the same offset, the same magnitude, the
        # same basis, drawn once with the constraint span removed and once with it
        # left in. If removing an 8% overlap buys compliance back at no cost in
        # diversity, the projection is doing work; if nothing moves, it is
        # decoration and should go.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        kind = getattr(args, "offset_basis_kind", "story")
        g = float(getattr(args, "main_gamma", 0.15))
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        arms = []
        for mode in ("orth", "free"):
            for prompt_only in (False, True):
                arms.append({"plan": make_plan(
                    beta=args.beta, offset_gamma=g, offset_mode=mode,
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    offset_prefill=prompt_only, offset_decode=not prompt_only,
                    steer_prefill=False, **quiet, **base)})
        return arms, (f"the per-story offset at gamma={g:g}, with the constraint "
                      "subspace projected out of it and with it left in, at the "
                      "decode steps and at the prompt")

    if name == "main":
        # The head-to-head. Every perturbed arm has an unperturbed control at the
        # same injection site, so a difference cannot be read as "the prompt is a
        # better place to push" when it is really "pushing harder helps".
        #
        #   site "decode"  the constraint vector is added at every decode step,
        #                  which is what every run so far has done
        #   site "prompt"  it is added to the prompt positions during prefill and
        #                  decoding is left alone
        #
        # Three families are compared at those two sites: the published per-token
        # noise, the per-story offset drawn beside the constraint vector, and
        # f(S_c), which perturbs the constraint vector itself.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        kind = getattr(args, "offset_basis_kind", "story")
        g = float(getattr(args, "main_gamma", 0.15))
        a = float(args.alpha)
        nh = int(getattr(args, "noise_horizon", 64) or 64)
        kappas = {
            "perp": float(getattr(args, "main_kappa_perp", 0.15)),
            "rotate": float(getattr(args, "main_kappa_rotate", 1.0)),
            "gain": float(getattr(args, "main_kappa_gain", 0.5)),
        }
        quiet = dict(noise_mode="none", noise_alpha=0.0)

        items = ["baseline"]
        items.append({"plan": make_plan(beta=args.beta, steer_prefill=False,
                                        **quiet, **base)})
        items.append({"plan": make_plan(beta=args.beta, steer_prefill=True,
                                        steer_decode=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=args.beta, noise_mode="orth", noise_alpha=a,
                                        noise_schedule="constant", steer_prefill=False,
                                        **base)})
        items.append({"plan": make_plan(beta=args.beta, noise_mode="orth", noise_alpha=a,
                                        noise_schedule="cosine_decay", noise_horizon=nh,
                                        steer_prefill=False, **base)})
        items.append({"plan": make_plan(beta=args.beta, offset_gamma=g, offset_mode="orth",
                                        offset_basis=offset_basis, offset_basis_kind=kind,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=args.beta, offset_gamma=g, offset_mode="orth",
                                        offset_basis=offset_basis, offset_basis_kind=kind,
                                        offset_prefill=True, offset_decode=False,
                                        steer_prefill=False, **quiet, **base)})
        for mode in ("perp", "rotate", "gain"):
            items.append({"plan": make_plan(
                beta=args.beta, jitter_mode=mode, jitter_kappa=kappas[mode],
                steer_prefill=False, **quiet, **base)})
            items.append({"plan": make_plan(
                beta=args.beta, jitter_mode=mode, jitter_kappa=kappas[mode],
                steer_prefill=True, steer_decode=False, **quiet, **base)})
        return items, (
            "baseline, the constraint vector alone, per-token noise flat and "
            f"cosine-decayed at a={a:g}, the per-story offset at gamma={g:g}, and "
            f"f(S_c) at kappa {kappas} -- each applied at the decode steps and at "
            "the prompt")

    if name == "prompt":
        # Where the offset is applied, at one magnitude per arm. The prompt-only
        # arm is the interesting one: the model is moved somewhere else before it
        # writes a token and then decodes with nothing touching it, so the shift
        # can be large without costing fluency.
        kind = getattr(args, "offset_basis_kind", "story")
        arms = []
        for g in args.gamma_sweep:
            arms.append({"plan": make_plan(
                beta=args.beta, noise_mode="none", noise_alpha=0.0,
                offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, offset_prefill=True, offset_decode=False,
                **common)})
        return arms, (f"steering + per-story offsets at gamma {list(args.gamma_sweep)} "
                      "applied to the prompt only, with decoding left unperturbed")

    if name == "amplify":
        # No perturbation at all: the component of the current state that lies in
        # the between-story subspace is stretched. Each story is pushed further
        # along the direction it was already taking, so stories are driven apart
        # from each other rather than jointly displaced.
        arms = []
        for lam in args.lambda_sweep:
            for pre in (False, True):
                arms.append({"plan": make_plan(
                    beta=args.beta, noise_mode="none", noise_alpha=0.0,
                    amplify_lambda=lam, amplify_prefill=pre,
                    amplify_basis=getattr(args, "amplify_basis", None),
                    amplify_mean=getattr(args, "amplify_mean", None),
                    **common)})
        return arms, (f"steering + between-story amplification at lambda "
                      f"{list(args.lambda_sweep)}, applied while writing and from "
                      "the prompt")

    if name == "decay":
        # The published cosine decay, but with its own horizon and at magnitudes
        # the flat schedule cannot survive. The reason flat noise collapses is that
        # every perturbed step is written to the key/value cache and read by every
        # later step, so the damage accumulates; a schedule that starts high and is
        # gone by the horizon spends the perturbation where it decides the premise
        # and leaves the rest of the story clean, so alpha can be raised.
        h = int(getattr(args, "noise_horizon", 64) or 64)
        mags = [m for m in args.alpha_sweep if m > 0]
        arms = [
            {"plan": make_plan(
                beta=args.beta, noise_mode="orth", noise_alpha=m,
                noise_schedule="cosine_decay", noise_horizon=h, **common)}
            for m in mags
        ]
        return arms, (f"steering + per-token noise at {mags}, decaying on a cosine "
                      f"from full strength to nothing over the first {h} tokens")

    if name == "window":
        # The same per-token noise as before, switched off after the opening. The
        # premise is chosen in the first few dozen tokens; past that the
        # perturbation can only cost grammar, and every perturbed step is written
        # to the KV cache and read by every later one, so the damage compounds.
        h = int(getattr(args, "noise_horizon", 24) or 24)
        mags = [m for m in args.alpha_sweep if m > 0]
        arms = [
            {"plan": make_plan(
                beta=args.beta, noise_mode="orth", noise_alpha=m,
                noise_schedule="prefix", noise_horizon=h, **common)}
            for m in mags
        ]
        return arms, (f"steering + per-token noise at {mags}, applied only over the "
                      f"first {h} generated tokens")

    if name == "compare":
        # The minimal head-to-head: unmodified generation vs. the proposed method.
        return ([ "baseline", noise_arms[1] ],
                "Baseline vs. the proposed method")

    if name == "noise":
        # Baseline first: without an unsteered reference the four arms can only be
        # compared to each other, not to doing nothing at all.
        return (["baseline"] + noise_arms,
                "Baseline + the four steering variants: no noise / noise with constraint "
                "directions removed / ordinary noise / noise confined to the constraint "
                "directions")

    if name == "core":
        items += [
            "baseline",
            {
                "mode": "residual_stream_noise",
                "residual_layers": layers,
                "residual_noise_std": resid_std,
                "disable_residual_noise_decay": True,
            },
        ]
        items += noise_arms
        return items, "Baseline / L-Res / the four steering variants"

    if name == "ortho":
        items = [
            {"plan": make_plan(beta=args.beta, orthogonalize=method,
                               noise_mode="orth", noise_alpha=args.alpha, **common)}
            for method in ("none", "gram_schmidt", "lowdin")
        ]
        return items, "orthogonalisation of the steering vectors: none / GS / Loewdin"

    if name == "gate":
        # Where the perturbation lands, at one fixed magnitude. The steering runs
        # at every step in all three arms, so the only thing that changes is which
        # decode steps the perturbation is allowed through at:
        #
        #   none    every step, which is what every run so far has done
        #   median  the more uncertain half of steps
        #   high    only the most uncertain tenth
        #
        # A low-entropy step is one where the model is finishing a word or
        # agreeing a verb: perturbing there costs grammar and buys no variety.
        gates = getattr(args, "gate_thresholds", {}) or {}
        items = [{"plan": make_plan(beta=args.beta, noise_mode="none", noise_alpha=0.0,
                                    noise_schedule="constant", **common)}]
        for level in ("none", "median", "high"):
            items.append({"plan": make_plan(
                beta=args.beta, noise_mode="orth", noise_alpha=args.alpha,
                noise_schedule="constant",
                gate_threshold=gates.get(level, 0.0),
                gate_level=level, **common)})
        return items, (f"entropy-gate sweep at a={args.alpha:g}: no gate, the more "
                       "uncertain half of steps, the most uncertain tenth")

    if name == "alpha":
        # Perturbation-magnitude sweep at fixed steering strength, run twice: once
        # with the perturbation redrawn every token, once with a single draw held
        # for the whole story. The magnitude means the same thing in both cases (a
        # multiple of the model's own activation scale), so the comparison isolates
        # how often the perturbation is drawn.
        #
        # No decay schedule on either. The published method decays noise over a
        # cosine horizon; that is left out here so magnitude is the only variable.
        mags = [m for m in args.alpha_sweep if m > 0]
        items = [
            {"plan": make_plan(beta=args.beta, noise_mode="none", noise_alpha=0.0,
                               noise_schedule="constant", **common)},
        ]
        for m in mags:
            items.append({"plan": make_plan(
                beta=args.beta, noise_mode="orth", noise_alpha=m,
                noise_schedule="constant", **common)})
        for m in mags:
            items.append({"plan": make_plan(
                beta=args.beta, noise_mode="none", noise_alpha=0.0,
                offset_gamma=m, offset_mode="orth", offset_basis=offset_basis, **common)})
        return items, (f"magnitude sweep {mags}, drawn per token and per story, "
                       "no decay schedule")

    if name == "beta":
        items = [
            {"plan": make_plan(beta=b, noise_mode="orth", noise_alpha=args.alpha, **common)}
            for b in args.beta_sweep
        ]
        return items, f"beta sweep {args.beta_sweep}"

    if name == "loo":
        for drop in names:
            kept = [n for n in names if n != drop]
            sub = dict(common)
            sub["names"] = kept
            items.append(
                {"plan": make_plan(beta=args.beta, noise_mode="orth",
                                   noise_alpha=args.alpha, **sub)}
            )
        return items, "leave-one-out over constraints"

    raise ValueError(f"unknown suite '{name}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_HF_IDS))
    ap.add_argument("--model-id")
    ap.add_argument("--model-name")
    ap.add_argument("--vectors", help="path to the .pt from build_steering_vectors.py")
    ap.add_argument("--layers", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--suite", nargs="+", default=["core"],
                    choices=["baseline", "sampling", "compare", "method", "noise", "offset", "core", "ortho", "alpha", "gate", "beta",
                             "loo", "all"])
    ap.add_argument("--with-baseline", action="store_true",
                    help="prepend an unsteered baseline condition to whichever suite is run "
                         "(already included in `compare` and `noise`)")
    ap.add_argument("--constraints", nargs="*", default=DEFAULT_CONSTRAINTS)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="steering strength as a multiple of median block RMS (default 1.0)")
    ap.add_argument("--beta-sweep", nargs="*", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--alpha", type=float, default=RMS_ALPHA,
                    help=f"noise strength as a multiple of the model's activation "
                         f"scale (paper default {RMS_ALPHA}); 0 disables noise")
    ap.add_argument("--alpha-sweep", nargs="*", type=float,
                    default=[0.0, 0.0875, 0.175, 0.35, 0.7],
                    help="values used by --suite alpha")
    ap.add_argument("--protect-rank", type=int, default=8,
                    help="principal components protected per constraint on top of the mean "
                         "direction. 0 protects only the C mean directions, which in a "
                         "4096-d stream removes ~0.07%% of a random draw and makes 'orth' "
                         "indistinguishable from 'iso' (default 8)")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--steer-prefill", action="store_true",
                    help="also steer across the prompt positions (CAA convention)")
    ap.add_argument("--num-stories", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output-dir", default="experiment_results/OrthoSteer")
    ap.add_argument("--sanity-check", type=int, default=3,
                    help="print the first N stories of each run (0 disables)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the plans, print the geometry and the run ids, generate nothing")
    args = ap.parse_args()

    if not args.model and not args.model_id:
        ap.error("pass --model or --model-id")

    model_id = args.model_id or MODEL_HF_IDS[args.model]
    model_name = args.model_name or args.model or model_id.split("/")[-1]

    if args.layers:
        lo, hi = args.layers
    elif args.model:
        lo, hi = MODEL_LAYER_RANGES[args.model]
    else:
        ap.error("pass --layers when using --model-id")
    layers = list(range(lo, hi))

    vec_path = Path(args.vectors) if args.vectors else Path("steering_vectors") / f"{model_name}.pt"
    if not vec_path.is_file():
        raise SystemExit(
            f"no steering vectors at {vec_path}. Run:\n"
            f"  python scripts/build_steering_vectors.py --model {args.model or model_name}"
        )
    vectors = SteeringVectorSet.load(vec_path)
    missing = [n for n in args.constraints if n not in vectors.vectors]
    if missing:
        raise SystemExit(f"{vec_path} has no vectors for {missing}; has {vectors.names}")

    print(f"model    : {model_id}")
    print(f"layers   : {layers}")
    print(f"vectors  : {vec_path}")
    vectors.print_report()

    model = build_model(model_id)

    # Same calibration as the paper: sigma = alpha * median_layer(block RMS).
    print("\n[calibration] measuring block RMS ...")
    cal = RMSCalibrator(model)
    rms = cal.collect_block_rms(
        [
            {"role": "system", "content": prompts.SYS_ZERO_SHOT},
            {"role": "user", "content": prompts.PROMPT_ZERO_SHOT},
        ],
        layers=layers,
    )
    rms_scale = float(np.median(list(rms.values())))
    print(f"[calibration] median block RMS = {rms_scale:.6g}  "
          f"-> noise sigma = {args.alpha * rms_scale:.6g}, "
          f"steering magnitude = {args.beta * rms_scale:.6g} per constraint")

    suites = ["core", "ortho", "alpha", "beta", "loo"] if "all" in args.suite else args.suite
    items = ["baseline"] if args.with_baseline else []
    descriptions = []
    for suite in suites:
        built, desc = build_suite(suite, vectors, layers, args.constraints, rms_scale, args)
        items.extend(built)
        descriptions.append(f"{suite}: {desc} ({len(built)} runs)")

    print("\n=== suites ===")
    for d in descriptions:
        print("  " + d)

    # Thread the shared sampling settings onto every run.
    normalised = []
    for it in items:
        d = {"mode": it} if isinstance(it, str) else dict(it)
        d.setdefault("temperature", args.temperature)
        d.setdefault("max_new_tokens_plan", args.max_new_tokens)
        d.setdefault("max_new_tokens_story", args.max_new_tokens)
        normalised.append(d)

    # Overlapping suites can produce the same condition twice; run ids are the
    # output filenames, so a duplicate would append two runs into one CSV.
    from noiseegra.setup_experiment import _spec_to_run_id

    specs, seen = [], set()
    for spec in make_specs(*normalised):
        rid = _spec_to_run_id(model_name, spec)
        if rid in seen:
            print(f"[dedupe] skipping duplicate condition: {rid}")
            continue
        seen.add(rid)
        specs.append(spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_id": model_id,
        "model_name": model_name,
        "layers": layers,
        "rms_scale": rms_scale,
        "alpha": args.alpha,
        "beta": args.beta,
        "constraints": args.constraints,
        "protect_rank": args.protect_rank,
        "suites": suites,
        "num_stories": [args.start, args.start + args.num_stories],
        "runs": [
            {
                "run_id": _spec_to_run_id(model_name, spec),
                "plan": (
                    spec.steering_plan.describe()
                    if spec.steering_plan is not None
                    else None
                ),
            }
            for spec in specs
        ],
    }
    (out_dir / f"{model_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    if args.dry_run:
        print(f"\n=== {len(specs)} run ids (dry run) ===")
        for spec in specs:
            print("  " + _spec_to_run_id(model_name, spec))
        return

    run_story_experiments(
        model=model,
        model_name=model_name,
        num_stories=(args.start, args.start + args.num_stories),
        specs=specs,
        output_dir=str(out_dir),
        sanity_check=args.sanity_check > 0,
        sanity_check_n=args.sanity_check,
    )
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
