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

import math
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

# Settings that belong to the whole run rather than to one suite. A suite that
# sweeps one of them passes it explicitly and wins; every other suite gets the
# run's value without its own call site having to name it.
#
# This exists because naming them at each call site does not work. The
# displacement-size spread was added to two of the twenty-odd `make_plan` calls,
# so every other suite quietly ran at zero -- and a three-arm run was launched,
# generated and scored before anyone noticed it had not been varying the size at
# all. Anything a command-line flag sets for a whole run belongs here.
RUN_DEFAULTS = {"offset_gamma_spread": 0.0, "offset_anchors": None,
                "anchor_scale": None, "offset_norm": "energy",
                "offset_taper": 1.0,
                # The colour of the displacement's wandering, the slowest wobble
                # it is allowed, and how its size runs over the story. These
                # belong to the run, not to a suite, for the same reason the
                # spread does: a suite that does not sweep one must still get
                # the run's value, or it silently runs the old mechanism.
                "noise_beta": None, "noise_fmin_cycles": 0.25,
                "offset_envelope": "flat",
                # A direction's own schedule, where it has one. Only "closure"
                # uses this: it means "bring it to an end", and on a schedule
                # that is quiet early and presses late it is a brake on the
                # over-running that steering causes. Measured on the children's
                # task it took looping from 48% of stories to 15% and improved
                # compliance at the same time.
                "schedules": None, "horizon": None,
                # Per-direction shares of the steering budget, measured from how
                # often each requirement is actually broken. Applied here rather
                # than at each suite for the same reason as everything else in
                # this table: there are twenty-odd places a plan is built, and a
                # setting threaded through by hand reached two of them once.
                "beta_weights": None}


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
    prefill_gain=1.0,
    prompt_tail_clear=0,
    prompt_head_clear=0,
    push_layers=None,
    offset_layers=None,
    offset_decode_steps=0,
    offset_scale=None,
    offset_draw_shape="sphere",
    offset_random_rank=None,
    offset_envelope_steps=0,
    shadow_protect=False,
    offset_secured_boost=0.0,
    steer_split_concentration=0.0,
    offset_front_gain=1.0,
    offset_prefill_gain=1.0,
    offset_taper=None,
    offset_gamma_spread=None,
    guard_direction="",
    noise_norm_match="energy",
    noise_schedule="constant",
    offset_gamma=0.0,
    offset_mode="none",
    offset_norm=None,
    noise_beta=None,
    noise_fmin_cycles=None,
    offset_envelope=None,
    beta_weights=None,
    offset_basis_kind="step",
    offset_draw="iid",
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
    jitter_walk=0.0,
    jitter_names=None,
    jitter_draw="iso",
    steer_decode=True,
    steer_budget=None,
    steer_mode="constant",
    feedback_cap=0.1,
    control_state=None,
    controller=None,
    targets=None,
    direction_source="extracted",
    gate_threshold=0.0,
    gate_level="none",
) -> SteeringPlan:
    schedules = schedules or RUN_DEFAULTS.get("schedules") or DEFAULT_SCHEDULES
    if horizon is None:
        horizon = RUN_DEFAULTS.get("horizon")
    betas = beta if isinstance(beta, dict) else {n: float(beta) for n in names}
    # False opts this arm out of the run's measured allocation, so a suite can
    # hold the old hand-picked split as a control in the same run as the new
    # one. None takes the run's value, which is what every other arm does.
    weights = (RUN_DEFAULTS.get("beta_weights") if beta_weights is None
               else beta_weights)
    if weights:
        # The plan renormalises the sum to the fixed budget afterwards, so
        # scaling here divides that fixed total rather than changing it: a
        # direction whose requirement never fails gives its share to the ones
        # that do.
        betas = {n: betas.get(n, 0.0) * float(weights.get(n, 1.0)) for n in names}
    specs = [
        ConstraintSpec(n, beta=betas.get(n, 0.0), schedule=schedules.get(n, "constant"))
        for n in names
    ]
    extra = vectors.shielded_subspace(names, protect_rank)
    if offset_gamma_spread is None:
        offset_gamma_spread = RUN_DEFAULTS["offset_gamma_spread"]
    if offset_norm is None:
        offset_norm = RUN_DEFAULTS["offset_norm"]
    if noise_beta is None:
        noise_beta = RUN_DEFAULTS["noise_beta"]
    if noise_fmin_cycles is None:
        noise_fmin_cycles = RUN_DEFAULTS["noise_fmin_cycles"]
    if offset_envelope is None:
        offset_envelope = RUN_DEFAULTS["offset_envelope"]
    plan = SteeringPlan.build(
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
        prefill_gain=prefill_gain,
        prompt_tail_clear=prompt_tail_clear,
        prompt_head_clear=prompt_head_clear,
        push_layers=push_layers,
        offset_layers=offset_layers,
        offset_decode_steps=offset_decode_steps,
        offset_scale=offset_scale,
        offset_draw_shape=offset_draw_shape,
        offset_random_rank=(RUN_DEFAULTS.get("offset_random_rank", 0)
                            if offset_random_rank is None else offset_random_rank),
        offset_envelope_steps=offset_envelope_steps,
        shadow_protect=shadow_protect,
        offset_secured_boost=offset_secured_boost,
        steer_split_concentration=steer_split_concentration,
        offset_front_gain=offset_front_gain,
        offset_prefill_gain=offset_prefill_gain,
        offset_anchors=RUN_DEFAULTS["offset_anchors"],
        anchor_scale=RUN_DEFAULTS["anchor_scale"],
        offset_gamma_spread=offset_gamma_spread,
        offset_taper=(RUN_DEFAULTS["offset_taper"] if offset_taper is None
                      else offset_taper),
        guard_direction=guard_direction,
        protect_extra=extra,
        offset_gamma=offset_gamma,
        offset_mode=offset_mode,
        offset_norm=offset_norm,
        noise_beta=noise_beta,
        noise_fmin_cycles=noise_fmin_cycles,
        offset_envelope=offset_envelope,
        offset_basis_kind=offset_basis_kind,
        offset_draw=offset_draw,
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
        jitter_walk=jitter_walk,
        jitter_names=jitter_names,
        jitter_draw=jitter_draw,
        steer_decode=steer_decode,
        steer_budget=steer_budget,
        steer_mode=steer_mode,
        feedback_cap=feedback_cap,
        control_state=control_state,
        controller=controller,
        targets=targets,
        direction_source=direction_source,
        gate_threshold=gate_threshold,
        gate_level=gate_level,
    )
    # Which directions the perturbation was held clear of, carried on the plan so
    # the run id can name them. Read off `vectors`, not off `protect_rank`: the
    # protected subspace is also enlarged by the principal components of the
    # steered constraints, and a tag computed from its rank said "shielded" on
    # every run including the unshielded control.
    plan.shield_names = list(getattr(vectors, "shield", ()) or ())
    plan.shield_rank = int(getattr(vectors, "shield_rank", 0) or 0)
    return plan


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

    if name == "withdecoder":
        # The push and the perturbation combined with a published truncation
        # scheme, at temperature 1.0.
        #
        # Every method arm until now used the checkpoint's own cut-offs, so the
        # method was compared against decoders it had never been combined with.
        # They are different interventions -- one reshapes the representation,
        # the other the distribution over the next token -- and the one thing
        # the perturbation could never buy without breaking stories is variety
        # in what happens, which is exactly what a truncation scheme that keeps
        # more of the tail supplies.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 1.5))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        flat = {n: 1.0 for n in names}
        plan = make_plan(
            beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
            offset_basis=offset_basis, offset_basis_kind=kind,
            offset_scale=getattr(args, "offset_scale", None),
            offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
            steer_prefill=True, prompt_tail_clear=keep,
            offset_prefill=True, offset_decode=False, **quiet, **base)
        items = [
            {"plan": plan, "temperature": 1.0},
            {"plan": plan, "temperature": 1.0, "typical_p": 0.2},
            {"plan": plan, "temperature": 1.0, "min_p": 0.05},
            {"plan": plan, "temperature": 1.0, "eta_cutoff": 2e-3},
        ]
        return items, (
            "the method at temperature 1.0 with the checkpoint's own cut-offs, "
            "and with locally typical, min-p and eta-sampling in their place")

    if name == "literature":
        # The published decoding methods a reviewer will expect to see, beside
        # the two already used. All are run through the generation stack's own
        # implementations rather than reimplemented here, so the comparison is
        # against the reference behaviour.
        #
        #   top-k               Fan et al., ACL 2018
        #   nucleus (top-p)     Holtzman et al., ICLR 2020
        #   locally typical     Meister et al., TACL 2023
        #   eta-sampling        Hewitt et al., EMNLP Findings 2022
        #   min-p               Nguyen et al., ICLR 2025
        #   contrastive search  Su et al., NeurIPS 2022
        #
        # Each is run at the temperature its own paper uses for open-ended
        # generation where that is stated, and at the raised temperature our
        # other baselines use where it is not, so none is handicapped by a
        # setting it was not designed for. Contrastive search is deterministic
        # and takes no temperature at all.
        t = float(getattr(args, "baseline_temperature", 1.8) or 1.8)
        items = [
            {"temperature": t, "top_p": 0.95},
            {"temperature": t, "top_k": 40},
            {"temperature": t, "typical_p": 0.95},
            {"temperature": t, "typical_p": 0.2},
            {"temperature": t, "eta_cutoff": 2e-3},
            {"temperature": t, "min_p": 0.05},
            {"temperature": t, "min_p": 0.1},
            {"penalty_alpha": 0.6, "top_k": 4},
        ]
        return items, (
            "the published decoding methods: nucleus, top-k, locally typical at "
            "two settings, eta-sampling, min-p at two settings, and contrastive "
            "search")

    if name == "sampling":
        # The decoding-parameter comparison. Turning the temperature up and
        # truncating the tail is the obvious way to buy diversity without
        # touching the representation, and it costs nothing, so it is the
        # reference a representation-level method has to beat -- not the greedy
        # baseline, which no one deploying a story generator would use.
        #
        # A single setting is not enough to compare against: temperature trades
        # diversity for compliance along a curve, and a method only beats it if
        # it lands above the whole curve rather than above one point on it. The
        # grid is swept so the curve is drawn.
        # The baseline is the model as it ships: temperature 1.0 and whatever
        # cut-offs the checkpoint's own generation config sets. That is what
        # someone running the model would get, and it is the reference.
        #
        # The comparison arms are the cheap way to buy diversity: turn the
        # temperature up and truncate the tail to keep it from falling apart.
        # Sweeping both is the point -- temperature trades diversity against
        # compliance along a curve, and a representation-level method only beats
        # it by landing above the whole curve rather than above one point.
        #
        # Worth recording: because Qwen3 ships top_p and top_k in its generation
        # config, the baseline is not untruncated sampling. An earlier grid put a
        # top-p 0.95 arm at temperature 1.0 and it came back identical to the
        # baseline in every digit across a hundred stories, which is how the
        # inherited setting was found. Cut-offs are therefore only varied where
        # the temperature is also raised, where they do something.
        grid = getattr(args, "sampling_grid", None) or ["1.0", "1.3:0.95", "1.6:0.95",
                                                        "1.8:0.95", "1.3:0.9", "1.6:0.9"]
        arms, seen = [], set()
        for cell in grid:
            temp, _, p = cell.partition(":")
            spec = {"mode": "baseline", "temperature": float(temp)}
            if p:
                spec["top_p"] = float(p)
            key = (spec["temperature"], spec.get("top_p"))
            if key in seen:
                continue
            seen.add(key)
            arms.append(spec)
        if getattr(args, "baseline_top_k", None):
            arms.append({"mode": "baseline", "temperature": args.baseline_temperature,
                         "top_k": args.baseline_top_k})
        return arms, ("the model as it ships, then the decoding curve: " +
                      ", ".join(sorted(
                          f"T{a['temperature']:g}" +
                          (f" with top-p {a['top_p']:g}" if "top_p" in a else " (defaults)")
                          for a in arms)))

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

    if name == "frontier":
        # The method at several operating points, to be run in the same command as
        # --suite sampling so every condition is scored identically. Diversity is
        # length-sensitive and steering shortens the stories, so a Vendi measured
        # under one truncation cannot be compared with one measured under another;
        # the headline comparison therefore has to be a single run.
        #
        # Two knobs, and they trade against each other. The size of the constraint
        # push sets compliance: at a total push of 3 the broken count is 4.7 of 13
        # against a baseline of 7.8, and at 4.5 it is 3.8. The size of the
        # per-story perturbation sets diversity: gamma 0.1 gives Vendi 8.3 and
        # gamma 0.15 gives 9.6, and each step costs a little compliance back.
        # Sweeping both draws the frontier the decoding curve has to be compared
        # against.
        #
        # The perturbation is drawn independently per story. Choosing the set
        # jointly so the perturbations repel was tried and is a null: four matched
        # pairs, all four slightly worse. Covering the offset subspace evenly does
        # not cover the output space evenly.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        budgets = list(getattr(args, "budget_sweep", [3.0, 4.5]))
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15, 0.25]))

        items = []
        for b in budgets:
            # the push alone, so the perturbation's contribution is readable
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            steer_prefill=False, **quiet, **base)})
            for g in gammas:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=False, offset_prefill=True, offset_decode=False,
                    **quiet, **base)})
        # the perturbation alone, as the other end of the frontier
        for g in gammas:
            items.append({"plan": make_plan(
                beta={n: 0.0 for n in names}, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=False, offset_prefill=True, offset_decode=False,
                **quiet, **base)})
        return items, (
            f"the constraint push at {budgets} crossed with a per-story "
            f"perturbation at gamma {gammas}, plus each alone")

    if name == "spread":
        # The complete method, its ablation, and the arms it has to beat.
        #
        # Compliance comes from the constraint push held to a fixed budget, so the
        # number of requirements does not set the strength. Diversity comes from a
        # per-story perturbation drawn from the subspace the model's own states
        # occupy and projected clear of the constraint directions, applied at the
        # prompt positions -- the best exchange rate any round has found, about
        # +1.1 Vendi for +0.4 requirements.
        #
        # The new part is that the perturbations are chosen as a set rather than
        # drawn one at a time. This is not a re-aiming of a single story's push:
        # turning a push of constant length was measured in round 2 and moved
        # Vendi by +0.12, and re-weighting the requirements moved it by -0.43.
        # Both are nulls, because the diversity a story gets is not a function of
        # where its own perturbation points. It is a property of the *set*, and
        # every round so far has drawn the set independently and hoped it spread.
        # In a rank-8 subspace a hundred independent draws contain pairs 95%
        # alike, so several stories are perturbed almost identically and the
        # diversity paid for is not collected.
        #
        # Every perturbation keeps its length, its subspace and its projection
        # clear of the constraints, so the constraint cost is unchanged by
        # construction. The iid arms at the same gamma are the ablation: any
        # difference between them and the spread arms is the set-level choice
        # alone.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        bud = float(getattr(args, "steer_budget", 3.0) or 3.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        gammas = list(getattr(args, "gamma_sweep", [0.1, 0.15]))

        def offset(g, draw, steered):
            return {"plan": make_plan(
                beta=flat if steered else {n: 0.0 for n in names},
                steer_budget=bud if steered else None,
                offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, offset_draw=draw,
                steer_prefill=False, offset_prefill=True, offset_decode=False,
                **quiet, **base)}

        items = ["baseline"]
        # compliance alone: the push, no perturbation
        items.append({"plan": make_plan(beta=flat, steer_budget=bud,
                                        steer_prefill=False, **quiet, **base)})
        for g in gammas:
            items.append(offset(g, "iid", False))       # perturbation alone, as before
            items.append(offset(g, "spread", False))    # perturbation alone, set chosen
            items.append(offset(g, "iid", True))        # the pair, drawn independently
            items.append(offset(g, "spread", True))     # the complete method
        return items, (
            f"the constraint push held at {bud:g} and a per-story perturbation at "
            f"gamma {gammas}, drawn independently and then chosen as a set")

    if name == "constdose":
        # Diversity at a constant steering dose.
        #
        # Every arm run so far has bought one of the two things and spent the
        # other. Steering takes the monotone requirements from 25% to 64% and
        # takes Vendi from 7.56 down to 6.17. Perturbation does the reverse. The
        # reason is that the two mechanisms are separate vectors: the perturbation
        # is energy added on top of the push, so it either moves a requirement or
        # it costs fluency, and in both cases the steering is no longer the amount
        # of steering that was chosen.
        #
        # These arms perturb the constraint vector itself, at a length that does
        # not change. Under a budget the coefficients are renormalised after the
        # per-story draw, so every story is written under a push of exactly the
        # budgeted length, aimed somewhere different:
        #
        #   turning       f(S) = |S| * (S/|S| + kappa*j)/sqrt(1+kappa^2), j
        #                 perpendicular to S. The length is unchanged exactly; the
        #                 push is turned by atan(kappa).
        #   re-weighting  a lognormal gain per requirement per story, mean 1, then
        #                 renormalised to the budget. Nothing leaves the constraint
        #                 span at all -- each story is written under a different
        #                 emphasis of the same requirements at the same total
        #                 pressure.
        #
        # Both hold the dose fixed by construction rather than by tuning, which is
        # the thing the temperature baselines cannot say: raising the temperature
        # buys variation by changing the whole output distribution and pays for it
        # in compliance.
        #
        # The sideways step is included as the version that does *not* preserve
        # the dose, so the comparison says whether holding it fixed is what
        # matters or whether any perturbation of the vector would do. The frame
        # arms perturb each direction before the directions are made orthogonal,
        # so what varies per story is the frame the push is expressed in rather
        # than the push inside a fixed frame.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        bud = float(getattr(args, "steer_budget", 3.0) or 3.0)
        flat = {n: 1.0 for n in names}
        fixed = dict(beta=flat, steer_budget=bud, steer_prefill=False, **quiet, **base)

        items = ["baseline", {"plan": make_plan(**fixed)}]
        for mode, sweep in (("rotate", getattr(args, "turn_sweep", [0.15, 0.3, 0.5])),
                            ("gain", getattr(args, "realloc_sweep", [0.3, 0.6, 1.0])),
                            ("perp", getattr(args, "kappa_sweep", [0.1, 0.2])),
                            ("frame", getattr(args, "frame_sweep", [0.1, 0.2]))):
            for k in sweep:
                items.append({"plan": make_plan(jitter_mode=mode, jitter_kappa=float(k),
                                                **fixed)})
        # One arm with the sideways direction drawn from the subspace the model's
        # own states occupy rather than isotropically, so the turned vector stays
        # on that manifold.
        if offset_basis is not None:
            items.append({"plan": make_plan(
                jitter_mode="rotate", jitter_kappa=float(getattr(args, "turn_sweep",
                                                                 [0.3])[len(getattr(args, "turn_sweep", [0.3])) // 2]),
                jitter_draw="basis", offset_basis=offset_basis,
                offset_basis_kind=getattr(args, "offset_basis_kind", "story"), **fixed)})
        return items, (
            f"the constraint push held at a total length of {bud:g} and aimed "
            f"differently for each story: turned, re-weighted, stepped sideways, "
            f"and expressed in a per-story frame")

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

    if name == "controls":
        # Which parts of the perturbation earn their place. Three surgeries on
        # the method's diversity half, run in one command so every arm is scored
        # identically:
        #
        #   * the constraint-span projection removed ("free"): the offset is the
        #     same draw from the same story-difference basis at the same length,
        #     but nothing keeps it off the constraint directions. If compliance
        #     falls with diversity unchanged, the projection is load-bearing; if
        #     nothing moves, it is decoration and should be dropped.
        #   * the story-difference basis replaced by an isotropic draw of the
        #     same length ("iso", still projected clear): if diversity falls,
        #     the sampled basis is where the diversity comes from, not the mere
        #     fact of a per-story shift.
        #   * the offset kept on while the model writes rather than held to the
        #     prompt: retried because the last measurement of this predates the
        #     steering budget and the one-sided requirement set.
        #
        # Each surgery appears with the constraint push (does it change the
        # method?) and, for the first two, without it (is the change a property
        # of the perturbation or of the pair?). The push alone and the baseline
        # anchor the comparison inside this run.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        g = float(getattr(args, "main_gamma", 0.15))
        b = float(getattr(args, "steer_budget", None) or 3.0)
        flat = {n: 1.0 for n in names}
        zero = {n: 0.0 for n in names}
        prompt_only = dict(offset_prefill=True, offset_decode=False)

        items = ["baseline"]
        # the push alone: what every perturbed arm here adds to
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        for betas in (flat, zero):
            bud = b if betas is flat else None
            # the method as it stands: story basis, projected clear, prompt only
            items.append({"plan": make_plan(
                beta=betas, steer_budget=bud, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=False, **prompt_only, **quiet, **base)})
            # projection removed, everything else identical
            items.append({"plan": make_plan(
                beta=betas, steer_budget=bud, offset_gamma=g, offset_mode="free",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=False, **prompt_only, **quiet, **base)})
            # basis removed: isotropic draw at the same length, still projected
            items.append({"plan": make_plan(
                beta=betas, steer_budget=bud, offset_gamma=g, offset_mode="orth",
                offset_basis=None, offset_basis_kind="iso",
                steer_prefill=False, **prompt_only, **quiet, **base)})
        # the offset kept on while the model writes, with the push
        items.append({"plan": make_plan(
            beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
            offset_basis=offset_basis, offset_basis_kind=kind,
            offset_prefill=True, offset_decode=True,
            steer_prefill=False, **quiet, **base)})
        return items, (
            f"ablations of the per-story perturbation at gamma={g:g}: the "
            "constraint-span projection removed, the story-difference basis "
            "replaced by an isotropic draw, and the offset kept on while the "
            f"model writes -- each against the push at {b:g} and the baseline")

    if name == "tame":
        # The constraint push at its working dose fragments the text at
        # temperature 1.0: a constant vector added at every decode step
        # accumulates over the story, and the stories collapse into staccato
        # two-word fragments (half of them, by the coherence checks). Every arm
        # here is one way of getting the push's compliance without the
        # accumulation, at temperature 1.0, each alone and with the per-story
        # perturbation on top:
        #   * the push fading out over the opening (cosine decay over
        #     ~steer_horizon tokens): set the style early, then let go;
        #   * the push on for the opening only, then off (prefix schedule);
        #   * the push applied to the prompt positions only, decoding untouched;
        #   * a smaller dose (budget 2);
        #   * the two directions that loop worst when pushed alone (dialogue at
        #     93%, short-sentences at 83% in the r15 probes) dropped, the
        #     remaining six renormalised to the same total push.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 3.0)
        g = float(getattr(args, "main_gamma", 0.15))
        flat = {n: 1.0 for n in names}
        six = {n: (0.0 if n in ("dialogue", "terse") else 1.0) for n in names}
        hz = int(getattr(args, "steer_horizon", 64) or 64)
        sched_base = {**base, "horizon": hz}
        decay = {n: "cosine_decay" for n in names}
        prefix = {n: "prefix" for n in names}
        off = dict(offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                   offset_basis_kind=getattr(args, "offset_basis_kind", "story"),
                   offset_prefill=True, offset_decode=False)

        items = ["baseline"]
        # anchors: the push as it stands, alone and with the perturbation
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **off, **quiet, **base)})
        # fading out over the opening
        items.append({"plan": make_plan(beta=flat, steer_budget=b, schedules=decay,
                                        steer_prefill=False, **quiet, **sched_base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b, schedules=decay,
                                        steer_prefill=False, **off, **quiet, **sched_base)})
        # on for the opening, then off
        items.append({"plan": make_plan(beta=flat, steer_budget=b, schedules=prefix,
                                        steer_prefill=False, **quiet, **sched_base)})
        # at the prompt only
        items.append({"plan": make_plan(beta=flat, steer_budget=b, steer_prefill=True,
                                        steer_decode=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b, steer_prefill=True,
                                        steer_decode=False, **off, **quiet, **base)})
        # a smaller dose
        items.append({"plan": make_plan(beta=flat, steer_budget=2.0,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=2.0,
                                        steer_prefill=False, **off, **quiet, **base)})
        # the two worst-looping directions dropped
        items.append({"plan": make_plan(beta=six, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=six, steer_budget=b,
                                        steer_prefill=False, **off, **quiet, **base)})
        return items, (
            f"ways to keep the push at {b:g} from fragmenting the text at "
            f"temperature 1.0: fading over the first {hz} tokens, on for the "
            f"first {hz} tokens only, prompt-only, budget 2, and the two "
            "worst-looping directions dropped -- each alone and with the "
            f"per-story perturbation at {g:g}")

    if name == "fsc":
        # Two families Haziq asked to bring to this model, at temperature 1.0.
        #
        # f(S_c): the perturbation applied to the constraint vector itself
        # rather than added beside it -- `perp` adds a per-story sideways step
        # while keeping the net push along the constraint vector fixed, `rotate`
        # turns the vector at unchanged length. Measured only on Qwen3-8B with
        # an unnormalised push (round 2); never with a budget or on this model.
        #
        # The original paper's mechanism: per-token Gaussian noise, no steering
        # at all, cosine-decayed over the opening -- the schedule that was the
        # surprise of round 2 (a large win at matched strength). `iso` because
        # the original method had no constraint directions to protect.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 3.0)
        flat = {n: 1.0 for n in names}
        zero = {n: 0.0 for n in names}
        nh = int(getattr(args, "noise_horizon", 64) or 64)

        items = ["baseline"]
        # the push alone, so every f(S_c) arm is readable against it
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        # the original method, cosine-decayed, three strengths
        for a in (0.2, 0.4, 0.6):
            items.append({"plan": make_plan(
                beta=zero, noise_mode="iso", noise_alpha=a,
                noise_schedule="cosine_decay", noise_horizon=nh,
                steer_prefill=False, **base)})
        # f(S_c) sideways step, while writing and at the prompt
        for kappa, prompt_only in ((0.15, False), (0.15, True), (0.3, True)):
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, jitter_mode="perp", jitter_kappa=kappa,
                steer_prefill=prompt_only, steer_decode=not prompt_only,
                **quiet, **base)})
        # f(S_c) turned at unchanged length
        for kappa in (0.5, 1.0):
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, jitter_mode="rotate", jitter_kappa=kappa,
                steer_prefill=False, **quiet, **base)})
        return items, (
            f"f(S_c) on this model under a budget of {b:g} (sideways step at "
            "0.15 and 0.3, turned at 0.5 and 1.0), and the original paper's "
            f"per-token noise cosine-decayed over {nh} tokens at 0.2/0.4/0.6 "
            "with no steering")

    if name == "combine":
        # Round 19 found three mechanisms that attack different parts of the
        # problem at temperature 1.0: the earlier layer band (6-14) stops the
        # push fragmenting text, the push fading out over the opening gets its
        # compliance almost for free, and a per-story sideways step on the
        # constraint vector (or the plain per-story offset) supplies variety.
        # This suite composes them, meant to be run with --layers 6 14. Every
        # arm uses the push at the given budget.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 3.0)
        flat = {n: 1.0 for n in names}
        hz = int(getattr(args, "steer_horizon", 64) or 64)
        sched_base = {**base, "horizon": hz}
        decay = {n: "cosine_decay" for n in names}
        kind = getattr(args, "offset_basis_kind", "story")

        def off(g):
            return dict(offset_gamma=g, offset_mode="orth",
                        offset_basis=offset_basis, offset_basis_kind=kind,
                        offset_prefill=True, offset_decode=False)

        items = ["baseline"]
        # the fading push with the shove, two shove sizes
        for g in (0.15, 0.25):
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            schedules=decay, steer_prefill=False,
                                            **off(g), **quiet, **sched_base)})
        # the constant push with the larger shove
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **off(0.25),
                                        **quiet, **base)})
        # the sideways step at this band: while writing, at the prompt, larger
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="perp", jitter_kappa=0.15,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="perp", jitter_kappa=0.15,
                                        steer_prefill=True, steer_decode=False,
                                        **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="perp", jitter_kappa=0.3,
                                        steer_prefill=True, steer_decode=False,
                                        **quiet, **base)})
        # fading push with the sideways step
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="perp", jitter_kappa=0.15,
                                        schedules=decay, steer_prefill=False,
                                        **quiet, **sched_base)})
        # sideways step and shove together
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        jitter_mode="perp", jitter_kappa=0.15,
                                        steer_prefill=True, steer_decode=False,
                                        **off(0.15), **quiet, **base)})
        return items, (
            f"the round-19 mechanisms composed at one layer band: the push at "
            f"{b:g} fading over {hz} tokens with the per-story shove at 0.15 and "
            "0.25, the constant push with the shove at 0.25, the sideways step "
            "at 0.15 (both sitings) and 0.3 (prompt), the fading push with the "
            "sideways step, and the sideways step plus shove together")

    if name == "dose":
        # A dose-response ladder for the constraint push, to be run twice: once
        # with the extracted directions and once with `--random-directions`,
        # which replaces every direction with a Gaussian draw of the same length.
        #
        # This is the control the project never ran. Every compliance gain so far
        # has been attributed to the extracted directions, but a constant offset
        # of this size added to the residual stream might shorten and simplify
        # the prose whichever way it points -- and on the children's task, where
        # most requirements reward short simple sentences, that alone would look
        # like the method working. If the two ladders lie on top of each other,
        # the extraction is not what is doing the work.
        #
        # A ladder rather than a single strength, because the interesting
        # question is not whether the curves differ at one point but whether the
        # extracted one has a usable window -- a strength that buys compliance
        # before it starts flattening the prose.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        flat = {n: 1.0 for n in names}
        ladder = [float(x) for x in (getattr(args, "budget_sweep", None)
                                     or (1.0, 2.0, 3.0))]
        items = ["baseline"]
        for b in ladder:
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            steer_prefill=False, **quiet, **base)})
        return items, (
            "a dose ladder for the constraint push at total strength "
            + ", ".join(f"{b:g}" for b in ladder)
            + ", against the untouched model. Run a second time with "
              "--random-directions to separate what the extracted directions do "
              "from what any push of that size does")

    if name == "varysize":
        # The per-story displacement given a size that varies between stories.
        #
        # Every run so far gives every story the same size, which puts them all
        # on a shell around the unperturbed state rather than filling the ball
        # inside it. Vendi measures spread and a shell has less of it: on the
        # geometry alone, a hundred points at this basis's rank scored with the
        # same Vendi, a radius drawn uniformly over [0, 2r] scores 2.3 against
        # 1.9 for a fixed radius, with the mean displacement unchanged.
        #
        # It is the one property of the displacement never varied. Where it
        # points, how it is drawn, where it is applied and how far it goes on
        # average have all been swept; how much that distance differs between
        # stories has not.
        #
        # It is also the only remaining candidate that is between-story --
        # which is the only kind of variation that has ever moved this axis --
        # and that does not raise the mean displacement, which is what the
        # formatting ceiling is a ceiling on.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        spreads = [float(x) for x in (getattr(args, "spread_sweep", None)
                                      or (0.5, 1.0))]

        items = []
        for sp in spreads:
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                offset_gamma_spread=sp,
                guard_direction=getattr(args, "guard_direction", ""),
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the per-story displacement at a mean of {g:g} with its size "
            "varying between stories by "
            + ", ".join(f"{s:g}" for s in spreads) + " of that")

    if name == "wander":
        # The aim of the constraint push performs a correlated random walk over
        # decode steps, at a length that never changes.
        #
        # Everything the method does today is decided once per story: which
        # perturbation, which f(S_c), which draw. Raised temperature with top-k
        # sampling varies at every token and leads variety of what happens by
        # 5.2 on an interval that clears zero, and nothing that changes where
        # the perturbation lands, how big it is, which layers it touches, which
        # steps it is let through at, or how its directions are drawn has closed
        # that. A once-per-story mechanism cannot buy token-level variety.
        #
        # The two things that do vary per token are both unusable as they
        # stand. Raising the sampling temperature is out by the terms of the
        # task. Fresh noise at every step is the published method's own and on
        # this model it broke every opening story at 0.4 and matched the
        # baseline's variety exactly at 0.2 -- white noise on the residual
        # stream destroys local structure precisely because it is white.
        #
        # This is the object in between, and it is a function of the constraint
        # vector rather than a term beside it. The push is rotated by a
        # direction that wanders: adjacent tokens are aimed almost identically,
        # so nothing breaks locally, while over a story the aim explores. In
        # rotate mode the length is preserved exactly at every step, so the
        # model is pushed as hard as ever and only the aim moves -- which is why
        # this should not cost compliance the way adding energy does.
        #
        # kappa is how far off the nominal aim the walk can sit; the walk rate
        # is how fast it gets there, with a correlation time of about its
        # reciprocal in steps.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        kappas = [float(x) for x in (getattr(args, "kappa_sweep", None) or (0.3,))]
        walks = [float(x) for x in (getattr(args, "walk_sweep", None) or (0.05, 0.15))]

        items = []
        for kap in kappas:
            for w in walks:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b,
                    jitter_mode="rotate", jitter_kappa=kap, jitter_walk=w,
                    offset_gamma=g, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    offset_scale=getattr(args, "offset_scale", None),
                    offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                    steer_prefill=True, prompt_tail_clear=keep,
                    offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            "the constraint push turned by "
            + ", ".join(f"{k:g}" for k in kappas)
            + " and its aim wandering at "
            + ", ".join(f"{w:g}" for w in walks)
            + " a step, at a length that never changes")

    if name == "pertoken":
        # The published method's own noise, under the constraint push, on this
        # task -- which it has never been run on.
        #
        # The per-story perturbation displaces the prompt once and the story is
        # then written from that one shifted place. Raised temperature with
        # top-k sampling varies at every token, and leads variety of what
        # happens by 5.2 on an interval that clears zero; no siting, size, gate
        # or draw shape has closed that, and the reason may simply be that one
        # shift per story cannot buy what per-token variation buys.
        #
        # Per-token noise is the one mechanism in this architecture that varies
        # at every step. It is also the published Arabic study's mechanism, so
        # this is the paper's own method under the constraint push rather than a
        # new idea. On this model without the push it was unusable -- 0.4 broke
        # every opening story and 0.2 matched the baseline's variety exactly.
        # With the push it may not be: the push is protective, measured three
        # ways, and the entropy gate has already shown a perturbation surviving
        # at more than twice the size it survives alone.
        #
        # Projected clear of the constraint directions, so it cannot undo the
        # compliance the push is buying.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        alphas = [float(x) for x in (getattr(args, "alpha_sweep", None)
                                     or (0.1, 0.2))]

        items = []
        for a in alphas:
            # the noise alone under the push, so its own contribution is readable
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, noise_mode="orth", noise_alpha=a,
                steer_prefill=True, prompt_tail_clear=keep, **base)})
            # and with the per-story perturbation the method already uses
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, noise_mode="orth", noise_alpha=a,
                offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False, **base)})
        return items, (
            "fresh noise at every decode step, projected clear of the constraint "
            "directions, at "
            + ", ".join(f"{a:g}" for a in alphas)
            + f", under the push at {b:g} -- alone and with the per-story "
              f"perturbation at {g:g}")

    if name == "eventvary":
        # A per-story gain on ONE named direction, rather than a displacement of
        # the whole hidden state.
        #
        # The displacement buys variety by moving the state off the region the
        # model writes stories from, which is what it pays for in coherence.
        # Varying how hard a single steered direction is pushed does not leave
        # that region at all: every story is written from a state the push itself
        # produced, just with more or less of one thing in it.
        #
        # It was measured flat when every steered direction was about how a
        # sentence is formed -- varying those varies the style, not the story.
        # With a direction that asks for an event, varying its strength varies
        # how much happens, which is the axis this branch has never won.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        target = getattr(args, "quieten", None) or "something_happens"
        kappas = [float(x) for x in (getattr(args, "kappa_sweep", None) or (0.4, 0.8))]
        if target not in names:
            raise ValueError(
                f"suite 'eventvary' varies the push on {target!r}, which is not in "
                f"the steered set ({sorted(names)}). Every arm would be identical. "
                "Pass --quieten with a direction that is steered."
            )
        flat = {n: 1.0 for n in names}
        items = []
        for k in kappas:
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                offset_gamma_spread=getattr(args, "offset_gamma_spread", 0.0),
                jitter_mode="gain", jitter_kappa=k, jitter_names=[target],
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"how hard the {target} direction is pushed drawn afresh for each "
            "story, at spreads of " + ", ".join(f"{k:g}" for k in kappas)
            + "; every story still written from a state the push produced")

    if name == "quieten":
        # One named direction given *less* of the push than the others.
        #
        # At 200 stories the method loses five: two refusals, and three that
        # stall into listing rather than narrating --
        #
        #     She sees a butterfly fluttering around a flower. She hears the
        #     wind whisper through the trees. She notices a squirrel jumping
        #     over a puddle. She feels the breeze on her face.
        #
        # which is what the senses direction looks like when it is driven past
        # the point of helping: the requirement asks for words about how things
        # look and sound, and the model satisfies it by enumerating them instead
        # of telling a story. The requirement itself is passed comfortably --
        # this is the direction overshooting, not falling short.
        #
        # The total strength is fixed, so quietening one direction gives the
        # others more. That is the trade being measured, and it is the opposite
        # of `weighted`, which is for a direction that needs more.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet_noise = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        target = getattr(args, "quieten", "sensory")
        weights = [float(x) for x in (getattr(args, "quieten_weights", None)
                                      or (0.5, 0.25))]

        if target not in names:
            raise ValueError(
                f"suite 'quieten' lowers the push on {target!r}, which is not in "
                f"the steered set ({sorted(names)}). Every arm would be "
                "identical. Pass --quieten with a direction that is steered."
            )

        items = []
        for w in weights:
            beta = {n: (w if n == target else 1.0) for n in names}
            items.append({"plan": make_plan(
                beta=beta, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                offset_gamma_spread=getattr(args, "offset_gamma_spread", 0.0),
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False,
                **quiet_noise, **base)})
        return items, (
            f"the {target} direction given "
            + ", ".join(f"{w:g}" for w in weights)
            + f" of the push the others get, at a perturbation of {g:g}")

    if name == "weighted":
        # The no-heading direction given more of the push than the others.
        #
        # Steering it at an equal share removes the headings the perturbation
        # induces: at a perturbation of 0.125 the rate goes from 4% of stories to
        # none, and the share a reader would accept from 96% to 99%. At 0.15 it
        # is not enough -- 3% come back -- and 0.15 is where the content variety
        # is, because variety of what happens rises with the size of the
        # perturbation at the prompt and nothing else moves it.
        #
        # So the two are in direct opposition and the balance between them is a
        # number rather than a choice: enough push on the heading requirement to
        # hold it at a perturbation size large enough to pass the baselines. The
        # total strength stays fixed, so weighting this direction up takes push
        # away from the other three, which is the cost being measured.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.0)
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or (0.15, 0.175))]
        weights = [float(x) for x in (getattr(args, "heading_weights", None) or (2.0, 3.0))]

        # Whichever direction holds the story's format is the one weighted.
        # Without it, every weight produces the same plan and the sweep is a
        # null that looks like a result -- which the suite-building test found,
        # because it builds every suite against one fixed set of names.
        target = next((n for n in ("story_format", "no_heading") if n in names), None)
        if target is None:
            raise ValueError(
                "suite 'weighted' varies the share of the push given to the "
                "direction that holds the story's format, and neither "
                "'story_format' nor 'no_heading' is in the steered set "
                f"({sorted(names)}). Every arm would be identical. Add one to "
                "--steer-vectors."
            )

        items = []
        for w in weights:
            beta = {n: (w if n == target else 1.0) for n in names}
            for gam in gammas:
                items.append({"plan": make_plan(
                    beta=beta, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, prompt_tail_clear=keep,
                    offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the {target} direction weighted "
            + ", ".join(f"{w:g}" for w in weights)
            + " times the others, crossed with a per-story perturbation at "
            + ", ".join(f"{x:g}" for x in gammas))

    if name == "final":
        # The head-to-head that settles the comparison, all in one run.
        #
        # Every number so far comes from pooling conditions across separate
        # runs, which is sound but leaves the margins inside the noise of a
        # hundred stories: the best arm sits at 74.6 on variety of what happens
        # against raised temperature's 74.8, and a difference of 0.2 means
        # nothing at that sample size. This puts the untouched model, both
        # decoding baselines and the method in one run at whatever --stories is
        # set to, so they share a prompt, a seed sequence and a pooling size
        # exactly.
        #
        # The method's settings are passed in rather than written here, because
        # which configuration this should be is what the rounds before it decide.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.125))
        keep = int(getattr(args, "prompt_tail", 8) or 8)

        items = [
            "baseline",
            {"mode": "baseline", "temperature": 1.8, "top_p": 0.95},
            {"mode": "baseline", "temperature": 1.8, "top_k": 40},
            {"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False, **quiet, **base)},
        ]
        return items, (
            "the untouched model, temperature 1.8 with nucleus 0.95, temperature "
            f"1.8 with top-k 40, and the constraint push at {b:g} at both sitings "
            f"with a per-story perturbation of {g:g} at the prompt, sparing its "
            f"last {keep} positions -- all in one run")

    if name == "framing":
        # The perturbation kept away from both ends of the prompt.
        #
        # Sparing the end alone -- the chat template's tokens saying the
        # instruction is over -- took titles from 29% to 4% and no further.
        # The other end has never been spared, and it is where the model is told
        # what it is producing: the system line reads "You write short stories
        # for readers in middle school and early high school", and it is the
        # only statement of that in the whole prompt. If perturbing it is what
        # lets the model fall back to writing a document, sparing both ends
        # should take the remaining titles out while leaving the perturbation on
        # the part of the prompt that describes the story, which is where the
        # content variety comes from.
        #
        # 4% of stories is what stands between the best arm and the baseline it
        # has to beat, so this is aimed at that alone; the size is held where it
        # scores best.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.125))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        heads = [int(x) for x in (getattr(args, "head_sweep", None) or (8, 16, 24))]

        items = []
        for head in heads:
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=True, prompt_tail_clear=keep, prompt_head_clear=head,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the per-story perturbation at {g:g} kept off the last {keep} prompt "
            f"positions and off the first "
            + ", ".join(str(h) for h in heads))

    if name == "opening":
        # The per-story perturbation applied to the opening decode steps only,
        # and never to the prompt.
        #
        # The two sitings measured so far do different jobs and neither does
        # both. Perturbing the prompt changes what the story is about -- variety
        # of what happens 71.6 against the untouched model's 55 -- but past a
        # certain displacement the model stops treating the prompt as an
        # instruction and opens with a title, in 4% to 29% of stories depending
        # on size. Perturbing throughout the writing never does that, 0% titles
        # and 100 of 100 coherent, but barely changes what happens: 56.4 at 0.05
        # and 63.1 at 0.1, below what the prompt siting gives at a smaller size.
        #
        # The reading is that a story's content is settled in its opening
        # tokens, and those are written from a prompt this siting has not
        # touched, so perturbing afterwards changes the texture of a story
        # already chosen. If that is right, perturbing the opening steps and
        # then stopping should buy the prompt siting's content variety without
        # touching the prompt, which is where every formatting failure has come
        # from.
        #
        # The window is swept because there is no way to guess where the
        # content stops being decided.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        windows = [int(x) for x in (getattr(args, "opening_steps", None) or (12, 30, 60))]
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or [g])]
        # The two sitings carry content variety by different routes and neither
        # reaches the baseline alone: the prompt gets there and admits titles,
        # the opening steps admit none and stop short. `--opening-at-prompt`
        # uses the same per-story offset at both, so a smaller displacement at
        # the prompt -- where the titles come from -- can be topped up over the
        # steps where the content is still being chosen.
        at_prompt = bool(getattr(args, "opening_at_prompt", False))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        head = int(getattr(args, "prompt_head", 0) or 0)

        # Window and size have to be crossed, not swept one at a time. This
        # siting is the only one that produces no titles at all, so its ceiling
        # is worth finding properly: a short window may carry a displacement
        # that the same siting cannot survive for a whole story. Applied
        # throughout, 0.15 keeps 90 stories of 100 and 0.3 keeps 8.
        items = []
        for w in windows:
            for gam in gammas:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, offset_prefill=at_prompt, offset_decode=True,
                    offset_decode_steps=w, prompt_tail_clear=keep,
                    prompt_head_clear=head, **quiet, **base)})
        return items, (
            "the per-story perturbation at "
            + ", ".join(f"{x:g}" for x in gammas)
            + " applied to the first "
            + ", ".join(str(w) for w in windows)
            + (f" decode steps and to the prompt, sparing its first {head} and "
               f"last {keep} positions" if at_prompt
               else " decode steps and never to the prompt"))

    if name == "promptbudget":
        # The whole of the prompt's displacement spent on the perturbation, with
        # the constraint push kept to the decode steps.
        #
        # The prompt is carrying two things at once. The push at the prompt buys
        # variety of wording -- 9.7 to 16.7 on its own -- and the perturbation
        # buys variety of what happens. Both displace the same representation,
        # and past a certain displacement the model stops treating it as an
        # instruction and writes a document with a title. So the two are
        # competing for one budget, and variety of what happens is the axis
        # still short.
        #
        # There is a direct measurement that they compete. Pushing at both
        # sitings rather than while writing alone takes variety of what happens
        # from 62.7 down to 58.7, while taking variety of wording from 12.2 up to
        # 16.3. A constant offset is the same for every story, so a push at the
        # prompt moves every story the same way and can only make them more
        # alike in content; the perturbation is the only part that differs
        # between stories.
        #
        # This gives the prompt to the perturbation alone, at sizes that were
        # unusable when the push was there as well, with the instruction
        # boundary spared.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None)
                                     or (0.15, 0.2, 0.25))]

        items = []
        for gam in gammas:
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=False, prompt_tail_clear=keep,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the constraint push at {b:g} kept to the decode steps, with the "
            f"prompt given to the per-story perturbation alone at "
            + ", ".join(f"{x:g}" for x in gammas)
            + f", sparing the last {keep} prompt positions")

    if name == "gatedwrite":
        # The per-story perturbation applied while the story is written, and
        # allowed through only at the decode steps where the model was unsure.
        #
        # This is aimed at the one thing standing between the method and the
        # goal. Variety of what happens is 66-70 against raised temperature's
        # 70-74, and every way of enlarging the intervention has broken the
        # formatting instead: a perturbation of 0.15 at the prompt gives 29%
        # titles, twice the prompt push gives 12%, four times gives 44%.
        #
        # The reason to expect gating to help is that those two things happen at
        # different kinds of decode step. Which noun comes next is a step where
        # the model is genuinely unsure; whether to capitalise, close a quote or
        # finish a word is one where it is not. Perturbing everywhere spends the
        # displacement on both. Perturbing only the uncertain steps aims it at
        # the choice that decides what happens and leaves the mechanics alone,
        # which is what should let the magnitude go up rather than down.
        #
        # The thresholds come from the untouched model's own decode entropies,
        # measured once, so "the more uncertain half" means the same thing in
        # every arm.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        gates = getattr(args, "gate_thresholds", {}) or {}
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or (0.15, 0.3))]
        levels = [str(x) for x in (getattr(args, "gate_sweep", None)
                                   or ("none", "median", "high"))]

        items = []
        for gam in gammas:
            for level in levels:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, prompt_tail_clear=keep,
                    offset_prefill=False, offset_decode=True,
                    gate_threshold=gates.get(level, 0.0), gate_level=level,
                    **quiet, **base)})
        return items, (
            "the per-story perturbation applied while the story is written at "
            + ", ".join(f"{x:g}" for x in gammas)
            + ", allowed through at " + ", ".join(levels) + " of the decode steps "
            "by how unsure the model was")

    if name == "whilewriting":
        # The per-story perturbation applied while the story is being written,
        # instead of, and as well as, at the prompt positions.
        #
        # Every run in this project has perturbed the prompt and left decoding
        # alone. That siting is now the thing standing in the way: the variety
        # it buys runs out at a perturbation of about 0.1, and beyond that the
        # model stops treating the prompt as an instruction and writes a
        # document with a title -- 29% of stories at 0.15. The same happens if
        # the constraint push at the prompt is enlarged instead: 12% titles at
        # twice the strength, 44% at four times. Whatever the displacement
        # budget of the prompt representation is, both levers spend it.
        #
        # Perturbing during decoding does not touch the prompt at all. The
        # instruction has already been read, so the failure that limits the
        # other two has no purchase here. What it risks instead is the text
        # itself, since the offset is then added at every step of a 150-word
        # story rather than once.
        #
        # Small sizes only, for that reason, and the prompt siting is kept as
        # the comparison it has to beat.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or (0.05, 0.1))]

        def arm(gam, at_prompt, while_writing):
            return {"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_prefill=at_prompt, offset_decode=while_writing,
                **quiet, **base)}

        items = []
        for gam in gammas:
            items.append(arm(gam, False, True))   # while writing only
            items.append(arm(gam, True, True))    # both sitings
        return items, (
            f"the per-story perturbation at {', '.join(f'{x:g}' for x in gammas)} "
            "applied while the story is written, alone and together with the "
            "prompt siting every run so far has used")

    if name == "bands":
        # The constraint push and the per-story perturbation given different
        # layer bands.
        #
        # They have no reason to want the same one. The push controls properties
        # of the sentence being written -- its tense, whether the character is
        # named -- and works at layers 6-13 of 28 while fragmenting the text at
        # 14-22. The perturbation's job is to send the model to a different
        # story, and which story gets told need not be decided where a sentence
        # is worded. Every run so far has used one band for both because one
        # flag set both.
        #
        # This matters now because the one axis still behind raised temperature
        # is variety of what *happens*, and every way of buying more of it by
        # turning something up has failed the same way: a larger perturbation,
        # and a stronger push at the prompt, both make the model stop treating
        # the prompt as an instruction and start writing a document with a
        # title. Moving the perturbation rather than enlarging it is the
        # remaining structural knob.
        #
        # The push is held where it is known to work and the perturbation is
        # moved later, in bands of the same width.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.1))
        keep = int(getattr(args, "prompt_tail", 8) or 8)
        push_band = list(range(6, 14))
        offset_bands = [list(range(6, 14)), list(range(10, 18)), list(range(14, 22))]

        items = []
        for band in offset_bands:
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=True, prompt_tail_clear=keep,
                push_layers=push_band, offset_layers=band,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the constraint push at {b:g} held at layers {push_band[0]}-{push_band[-1]} "
            f"with the per-story perturbation at {g:g} moved to layers "
            + ", ".join(f"{x[0]}-{x[-1]}" for x in offset_bands))

    if name == "wholefive":
        # The comparison for the whole-story rules, in one run and one seed
        # sequence: the model as it ships, raised temperature with nucleus
        # sampling, the method, the push with no perturbation, and the
        # perturbation with no push.
        #
        # The last two are the ablation. Either half alone is a control for the
        # other: the push is what wins compliance and flattens the output, the
        # perturbation is what varies it, and neither on its own is the method.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.5)
        g = float((getattr(args, "gamma_sweep", None) or [1.5])[0])
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        flat = {n: 1.0 for n in names}
        none = {n: 0.0 for n in names}
        t = float(getattr(args, "baseline_temperature", 1.8) or 1.8)

        def arm(beta, gamma, decode):
            return {"plan": make_plan(
                beta=beta, steer_budget=(b if any(beta.values()) else None),
                offset_gamma=gamma, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=any(beta.values()), prompt_tail_clear=keep,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "manifold"),
                offset_prefill=gamma > 0, offset_decode=decode,
                noise_beta=(cn if gamma > 0 and decode else None),
                **quiet, **base)}

        return [
            "baseline",                                   # the model as it ships
            {"mode": "baseline", "temperature": t, "top_p": 0.95},
            arm(flat, g, True),                           # the method
            arm(flat, 0.0, False),                        # the push alone
            arm(none, g, True),                           # the perturbation alone
        ], (
            "the model as it ships, nucleus sampling at "
            f"temperature {t:g}, the method at {g:g} story-distances with a "
            f"noise colour of {cn:g}, the push alone, and the perturbation alone")

    if name == "randomfisher":
        # The method with nothing learned from the model's outputs. The rule
        # steering is the push alone's, unchanged; the perturbation is drawn in
        # a random subspace redrawn for every story, projected clear of the rule
        # directions, wandering over the story as coloured noise, and sized by
        # how far it moves the model's predictions rather than by its length in
        # activation space. No stories are sampled before generation.
        #
        # The sizes are in units of the change nucleus sampling at the baseline
        # temperature makes to what the model samples from, so 1.0 moves the
        # predictions as far as the baseline this has to beat.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        sizes = [float(x) for x in (getattr(args, "gamma_sweep", None) or [0.5, 1.0, 2.0])]
        flat = {n: 1.0 for n in names}

        def arm(size):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=size, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                **quiet, **base)}

        return [arm(x) for x in sizes], (
            "the rule steering with random noise in a fresh rank-"
            f"{rank} subspace per story, coloured at {cn:g}, sized to move the "
            "predictions " + ", ".join(f"{x:g}" for x in sizes)
            + " times as far as nucleus sampling does")

    if name == "randomvariants":
        # The same method with its two design choices varied one at a time.
        #
        # When the noise changes: once per story, wandering slowly as coloured
        # noise (the method), or a fresh direction at every token, the way the
        # published method adds noise. Per-token noise leaves the prompt alone,
        # as that method does; the per-story offset also shifts how the prompt
        # is read.
        #
        # How big it is: sized per story by how far it moves the predictions
        # (the method), or a fixed multiple of the residual stream's RMS at the
        # layers it is added to, with and without a cosine fade over the story.
        #
        # Everything else -- the rule steering, the random subspace redrawn per
        # story, the projection clear of the rule directions -- is identical.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        size = float(getattr(args, "variant_size", 1.0) or 1.0)
        rms_k = float(getattr(args, "fixed_rms_multiple", 0.4) or 0.4)
        fade = int(getattr(args, "decay_tokens", 260) or 260)
        flat = {n: 1.0 for n in names}

        def arm(per_token, norm, gamma, envelope):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=gamma, offset_mode="orth", offset_norm=norm,
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=not per_token, offset_decode=True,
                noise_beta=(0.0 if per_token else cn),
                offset_envelope=envelope,
                offset_envelope_steps=(fade if envelope != "flat" else 0),
                **quiet, **base)}

        return [
            arm(True, "fisher", size, "flat"),      # per token, sized by the output
            arm(False, "energy", rms_k, "flat"),    # per story, fixed RMS multiple
            arm(False, "energy", rms_k, "decay"),   # the same, fading over the story
            arm(True, "energy", rms_k, "flat"),     # per token, fixed RMS multiple
            arm(True, "energy", rms_k, "decay"),    # the same, fading over the story
        ], (
            f"per-token noise sized to {size:g} nucleus-units; per-story and "
            f"per-token noise at {rms_k:g} times the residual stream's RMS, each "
            f"with and without a cosine fade over {fade} tokens")

    if name == "randomnext":
        # Where the variety that is still missing might come from. At 2.0
        # nucleus-units the per-story noise nearly matched nucleus sampling on
        # wording but a third of the stories opened by answering the reader, the
        # failure this project has traced to perturbing the end of the prompt.
        # So the larger sizes are tried with the noise applied only while the
        # story is written, and once more held back at the start and brought in
        # over the first 260 tokens, where the register is already settled.
        #
        # The noise alone, with no rule steering, is the ablation the five-way
        # comparison asks for.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        fade = int(getattr(args, "decay_tokens", 260) or 260)
        flat = {n: 1.0 for n in names}
        none = {n: 0.0 for n in names}

        def arm(size, *, steer=True, prompt=True, envelope="flat", beta=cn):
            return {"plan": make_plan(
                beta=(flat if steer else none), steer_budget=(b if steer else None),
                offset_gamma=size, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=steer, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=prompt, offset_decode=True, noise_beta=beta,
                offset_envelope=envelope,
                offset_envelope_steps=(fade if envelope != "flat" else 0),
                **quiet, **base)}

        return [
            arm(1.0, steer=False),                       # the noise alone
            arm(1.4),                                    # between the two that worked and failed
            arm(2.0, prompt=False),                      # large, while writing only
            arm(2.0, prompt=False, envelope="rise"),     # and brought in after the opening
            arm(1.0, beta=1.0),                          # wanders more within the story
        ], ("the noise alone at 1.0; the method at 1.4; at 2.0 while writing "
            "only, with and without a rise over the first "
            f"{fade} tokens; and at 1.0 with a noise colour of 1")

    if name == "randombase":
        # The three references, with no perturbation and so nothing sampled
        # first: the model as it ships, nucleus sampling at the baseline
        # temperature, and the rule steering alone.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        t = float(getattr(args, "baseline_temperature", 1.8) or 1.8)
        flat = {n: 1.0 for n in names}
        return [
            "baseline",
            {"mode": "baseline", "temperature": t, "top_p": 0.95},
            {"plan": make_plan(beta=flat, steer_budget=b, offset_gamma=0.0,
                               offset_mode="none", steer_prefill=True,
                               prompt_tail_clear=keep, **quiet, **base)},
        ], (f"the model as it ships, nucleus sampling at temperature {t:g}, "
            "and the rule steering alone")

    if name == "randomfixed":
        # The per-story noise at one fixed length for every story, with and
        # without a cosine fade: the length the output-based sizing reached at
        # most, over every story it sized at 0.5 and 1.0 nucleus-units. Against
        # the sized arms this asks whether sizing each story separately matters,
        # given the same ceiling.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        fade = int(getattr(args, "decay_tokens", 260) or 260)
        length = float(getattr(args, "fixed_length", 14.83) or 14.83)
        vecs = getattr(vectors, "vectors", vectors)
        dim = int(next(iter(next(iter(vecs.values())).values())).numel())
        gamma = length / (float(rms_scale) * math.sqrt(dim))
        flat = {n: 1.0 for n in names}

        def arm(envelope):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=gamma, offset_mode="orth", offset_norm="energy",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                offset_envelope=envelope,
                offset_envelope_steps=(fade if envelope != "flat" else 0),
                **quiet, **base)}

        return [arm("flat"), arm("decay")], (
            f"per-story noise at a fixed length of {length:g} ({gamma:.4f} times "
            f"the RMS per coordinate), with and without a cosine fade over {fade} "
            "tokens")

    if name == "randomshadow":
        # The method with a shadow copy of each story, held level with it along
        # the protected directions at every steered layer. Everything else is
        # suite 'randomfisher': the size still means how far the noise moves
        # the predictions, now measured with the shadow's correction applied.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        sizes = [float(x) for x in (getattr(args, "gamma_sweep", None) or [0.5, 1.0, 1.4])]
        flat = {n: 1.0 for n in names}

        def arm(size):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=size, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank, shadow_protect=True,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                **quiet, **base)}

        return [arm(x) for x in sizes], (
            "the method with a shadow copy protecting the rule directions at every "
            "steered layer, sized to move the predictions "
            + ", ".join(f"{x:g}" for x in sizes) + " times as far as nucleus sampling")

    if name == "randomloop":
        # Steering that reacts to the story, with the random per-story noise.
        # The shadow copy removed the noise's leak into the rule directions and
        # barely changed how many rules broke: they break because the noise
        # changes what the story is about -- one child alone has no girl and
        # nobody to talk to -- and protecting a direction cannot put a second
        # character in. The controller can: it asks for a comparison, speech or
        # a he and a she once the story is far enough in to be missing them,
        # and goes silent once they are there.
        #
        # Once those rules are met they stay met, which is the moment the story
        # needs least from the steering. One arm raises the noise then.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        boost = float(getattr(args, "secured_boost", 1.0) or 1.0)
        ctl = getattr(args, "controller", None)
        if ctl is None:
            raise SystemExit(
                "suite 'randomloop' steers by the error the controller measures, "
                "and no controller was built. It comes from --constraint-set, "
                "which has to be 'whole' for this suite.")
        gains = [float(x) for x in (getattr(args, "feedback_betas", None) or [1.0, 2.0])]
        g_top = max(gains)
        # A constant push on closure would ask every story to end from its first
        # word, so the constant reference leaves it out.
        const = {n: (0.0 if n == "closure" else 1.0) for n in names}

        def arm(size, *, gain=None, envelope="flat"):
            steer = ({} if gain is None else
                     dict(steer_mode="error", control_state={}, controller=ctl))
            beta = const if gain is None else {n: gain for n in names}
            noisy = size > 0
            return {"plan": make_plan(
                beta=beta, steer_budget=b, **steer,
                offset_gamma=size, offset_mode=("orth" if noisy else "none"),
                offset_norm="fisher", offset_basis=None,
                offset_basis_kind=("random" if noisy else "step"),
                offset_random_rank=(rank if noisy else 0),
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=noisy, offset_decode=noisy,
                noise_beta=(cn if noisy else None),
                offset_envelope=envelope,
                offset_secured_boost=(boost if envelope == "secured" else 0.0),
                **quiet, **base)}

        items = [arm(1.0)]
        items += [arm(1.0, gain=g) for g in gains]
        items += [arm(1.4, gain=g_top),
                  arm(1.0, gain=g_top, envelope="secured"),
                  arm(0.0, gain=g_top)]
        return items, (
            "constant steering with the noise at 1.0; steering set by the story at "
            "gains " + ", ".join(f"{x:g}" for x in gains) + " with the noise at 1.0; "
            f"at gain {g_top:g} with the noise at 1.4, with the noise at 1.0 rising "
            f"to {1 + boost:g}x as the rules that stay met are met, and with no noise")

    if name == "randomsplit":
        # The constant push flattens the wording: every story gets the same
        # stylistic shove from its first word, and the arms that pushed less
        # tied nucleus sampling on wording while the constant ones lost. Here the
        # total push is kept and only its split varies, drawn at random per story.
        # Separately, the two rules the model already meets unprompted -- no
        # title, no repeated sentence -- are given no share, so theirs goes to the
        # rules it breaks. All arms carry the per-story noise at 1.0.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        concs = [float(x) for x in (getattr(args, "split_concentrations", None) or [1.0, 4.0])]
        met = {"no_heading", "distinct_sentences"}
        flat = {n: 1.0 for n in names}
        focused = {n: (0.0 if n in met else 1.0) for n in names}

        def arm(beta, conc):
            return {"plan": make_plan(
                beta=beta, steer_budget=b, steer_split_concentration=conc,
                offset_gamma=1.0, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                **quiet, **base)}

        items = [arm(flat, c) for c in concs]
        # Only when there is a rule to take the share from: without either in
        # the steered set the focused arms are the flat ones under another name.
        if met & set(names):
            items += [arm(focused, 0.0), arm(focused, min(concs))]
        return items, (
            "the noise at 1.0 with the steering split drawn per story at "
            "concentrations " + ", ".join(f"{c:g}" for c in concs)
            + "; with no share for the two rules already met; and both together")

    if name == "randomfront":
        # The wording gap sits in the opening: under the push most stories begin
        # alike, and past the first forty words the method nearly matches nucleus
        # sampling on wording. So the noise starts larger and falls back to its
        # size over the opening tokens -- while writing only, and once with the
        # prompt's share raised too, the part that has made the model answer the
        # reader before. Judged at 40 words and at 100, where it cannot act.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        span = int(getattr(args, "front_tokens", 40) or 40)
        gains = [float(x) for x in (getattr(args, "front_gains", None) or [1.5, 2.0])]
        flat = {n: 1.0 for n in names}

        def arm(front, prompt_gain=1.0):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=1.0, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                offset_envelope="front", offset_envelope_steps=span,
                offset_front_gain=front, offset_prefill_gain=prompt_gain,
                **quiet, **base)}

        items = [arm(g) for g in gains] + [arm(max(gains), max(gains))]
        return items, (
            f"the noise at 1.0, starting at {', '.join(f'{g:g}' for g in gains)} times "
            f"that and falling back over {span} tokens; and the largest with the "
            "prompt's share raised to match")

    if name == "randomprompt":
        # Raising the prompt's share of the noise was the one change that beat
        # nucleus sampling on wording, and it cost a quarter of the stories to
        # the model answering the reader instead of telling a story. Two ways to
        # keep the first without the second: fade the prompt's noise toward its
        # end, where that failure has been traced before, or hold the story level
        # with a noiseless shadow along the protected directions, which include
        # the one separating a story from anything else. Noise while writing
        # stays at 1.0.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        rank = int(getattr(args, "offset_random_rank", 64) or 64)
        g = float(getattr(args, "prompt_gain", 2.0) or 2.0)
        fade = float(getattr(args, "prompt_fade_to", 0.25) or 0.25)
        flat = {n: 1.0 for n in names}

        def arm(gain, *, taper=1.0, shadow=False):
            return {"plan": make_plan(
                beta=flat, steer_budget=b,
                offset_gamma=1.0, offset_mode="orth", offset_norm="fisher",
                offset_basis=None, offset_basis_kind="random",
                offset_random_rank=rank, shadow_protect=shadow,
                steer_prefill=True, prompt_tail_clear=keep,
                offset_draw_shape="sphere",
                offset_prefill=True, offset_decode=True, noise_beta=cn,
                offset_prefill_gain=gain, offset_taper=taper,
                **quiet, **base)}

        return [arm(g, taper=fade), arm(g, shadow=True), arm(1.0 + (g - 1.0) / 2)], (
            f"the prompt's noise at {g:g}x fading to {fade:g} of that by its end; "
            f"at {g:g}x with the shadow; and at {1.0 + (g - 1.0) / 2:g}x plain")

    if name == "wholeloop":
        # Steering strength set by the story being written, not by a calibration
        # run. Every rule in this set is satisfied or not, so each direction
        # pushes while its rule is unmet and goes silent once it is met -- the
        # coefficient is a function of the text so far rather than a constant
        # somebody measured on one model and one prompt beforehand.
        #
        # Closure is in the set for the same reason it is a brake elsewhere: the
        # failures this method leaves are stories that do not end, and closure's
        # error is zero until the story runs past the length it was asked for.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.5)
        g = float((getattr(args, "gamma_sweep", None) or [1.5])[0])
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        cn = float((getattr(args, "noise_beta_sweep", None) or [2.0])[0])
        ctl = getattr(args, "controller", None)
        if ctl is None:
            raise SystemExit(
                "suite 'wholeloop' steers by the error the controller measures, "
                "and no controller was built. It comes from --constraint-set, "
                "which has to be 'whole' for this suite.")
        gains = [float(x) for x in (getattr(args, "feedback_betas", None) or [1.0, 2.0])]
        flat = {n: 1.0 for n in names}

        items = [{"plan": make_plan(
            beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
            offset_basis=offset_basis, offset_basis_kind=kind, steer_prefill=True,
            prompt_tail_clear=keep, offset_prefill=True, offset_decode=True,
            offset_draw_shape=getattr(args, "offset_draw_shape", "manifold"),
            offset_scale=getattr(args, "offset_scale", None),
            noise_beta=cn, **quiet, **base)}]
        for gain in gains:
            items.append({"plan": make_plan(
                beta={n: gain for n in names}, steer_mode="error",
                control_state={}, controller=ctl, steer_budget=b,
                offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, steer_prefill=True,
                prompt_tail_clear=keep, offset_prefill=True, offset_decode=True,
                offset_draw_shape=getattr(args, "offset_draw_shape", "manifold"),
                offset_scale=getattr(args, "offset_scale", None),
                noise_beta=cn, **quiet, **base)})
        return items, (
            "a constant push against the same push set by the error the story "
            "itself shows, at gains "
            + ", ".join(f"{x:g}" for x in gains)
            + f", both with the perturbation at {g:g} story-distances")

    if name == "colourfront":
        # The frontier along the axis the colour suite found, not a fresh sweep
        # of an old knob. At beta=2, gamma 1.5, the method beats nucleus
        # sampling on both variety measures and on compliance at once, at 194
        # of 200 coherent. Two things are not won: every story coherent, and
        # variety of what happens against top-k rather than nucleus.
        #
        # Both are governed by how far the displacement goes, which is gamma,
        # and how much of its variation sits between stories rather than within
        # one, which is beta. The exponent's optimum is interior -- 2 beat both
        # 0 and infinity -- so the peak is found either side of it, not at an
        # end. Every arm is allocated and applied while writing, because those
        # were separated already and both helped.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.5)
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or [1.5])]
        betas = [float(x) for x in (getattr(args, "noise_beta_sweep", None) or [2.0])]
        flat = {n: 1.0 for n in names}
        items = []
        for g in gammas:
            for cn in betas:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, prompt_tail_clear=keep,
                    offset_scale=getattr(args, "offset_scale", None),
                    offset_draw_shape=getattr(args, "offset_draw_shape", "manifold"),
                    offset_prefill=True, offset_decode=True,
                    noise_beta=cn, **quiet, **base)})
        if len(items) < 2:
            raise SystemExit(
                "suite 'colourfront' maps a frontier and was given one point: "
                f"gammas {gammas}, exponents {betas}. Pass more than one of "
                "either with --gamma-sweep or --noise-beta-sweep.")
        return items, (
            "the displacement applied while writing, its direction wandering at "
            "exponents " + ", ".join(f"{x:g}" for x in betas)
            + " and its length at " + ", ".join(f"{x:g}" for x in gammas)
            + f" story-distances, every arm allocated, total push {b:g}")

    if name == "colour":
        # Two changes to the method, separated, against the method as it stands.
        #
        # One: which requirements are steered. Five directions were carrying
        # twelve requirements and the five were picked by judgement. Measured on
        # this model's own stories, that pick spent budget on the one
        # requirement that never fails and skipped the second worst. The
        # allocated arms weight every direction by how often its requirement is
        # actually broken.
        #
        # Two: whether the displacement stands still. It has always been one
        # fixed vector, applied over the prompt and then left alone. Here its
        # direction follows a 1/f^beta trajectory along the token axis while its
        # length is held at gamma story-distances, so beta alone decides how
        # much of the variation is between stories and how much is within one.
        # beta=0 is the published per-token mechanism and is included precisely
        # because it should lose: it is the end of the axis where nothing
        # differs between stories, and this project has measured three times
        # that such a mechanism cannot move variety.
        #
        # Wandering only means anything if the displacement is applied while the
        # model writes, which the current method does not do. That is a change
        # of its own, so it gets an arm of its own (fixed, prompt and writing)
        # rather than being confounded with the colour.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        b = float(getattr(args, "steer_budget", None) or 2.5)
        g = float((getattr(args, "gamma_sweep", None) or [1.5])[0])
        keep = int((getattr(args, "tail_sweep", None) or [8])[0])
        # The five that were picked by hand, as a control. Named rather than
        # taken from a flag so the control cannot drift with the flag.
        PICKED = ("present_tense", "sensory", "named_character", "no_heading",
                  "something_happens")
        picked = {n: (1.0 if n in PICKED else 0.0) for n in names}
        if not any(picked.values()):
            raise SystemExit(
                "suite 'colour' holds the hand-picked five as its control, and "
                "none of " + str(sorted(PICKED)) + " is in the steered set "
                + str(sorted(names)) + ". Every arm would differ from the "
                "control by the direction set as well as by what is being "
                "tested. Add them to --steer-vectors.")
        flat = {n: 1.0 for n in names}

        def arm(beta, weights, cn=None, decode=False, env="flat"):
            return {"plan": make_plan(
                beta=beta, beta_weights=weights, steer_budget=b, offset_gamma=g,
                offset_mode="orth", offset_basis=offset_basis,
                offset_basis_kind=kind, steer_prefill=True,
                prompt_tail_clear=keep,
                offset_scale=getattr(args, "offset_scale", None),
                offset_draw_shape=getattr(args, "offset_draw_shape", "manifold"),
                offset_prefill=True, offset_decode=decode,
                noise_beta=cn, offset_envelope=env, **quiet, **base)}

        items = [
            arm(picked, False),                          # the method as it stands
            arm(flat, None),                             # allocation alone
            arm(flat, None, decode=True),                # applied while writing too
            arm(flat, None, cn=2.0, decode=True),        # a slow wander
            arm(flat, None, cn=1.0, decode=True),        # a faster wander
            arm(flat, None, cn=0.0, decode=True),        # the per-token end
            arm(flat, None, cn=2.0, decode=True, env="rise"),   # held back at the start
            arm(picked, False, cn=2.0, decode=True),     # wandering, old direction set
        ]
        return items, (
            "the hand-picked five against a budget divided by measured failure, "
            "and a displacement that stands still against one whose direction "
            f"wanders at exponents 2, 1 and 0, all at {g:g} story-distances "
            f"and a total push of {b:g}")

    if name == "boundary":
        # The best-scoring arm with the final prompt positions left unperturbed.
        #
        # That arm -- four directions pushed at a total strength of 2 at both
        # sitings with a per-story perturbation of 0.15 -- beats both raised
        # temperature settings on requirements broken and on variety of wording,
        # and it opens 29% of its stories with a title, which the instruction
        # explicitly forbids and which no baseline does at all. The prompt ends
        # with the chat template's own tokens saying the instruction is over and
        # the answer begins; perturbing those too is the suspect.
        #
        # Swept rather than set, because the number of template tokens is a
        # property of the tokeniser and guessing it wrong in either direction is
        # invisible: too few leaves the boundary perturbed, too many stops the
        # perturbation reaching the instruction it is supposed to move.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.15))
        tails = [int(x) for x in (getattr(args, "tail_sweep", None) or (2, 4, 8))]
        gammas = [float(x) for x in (getattr(args, "gamma_sweep", None) or [g])]
        # The total strength is swept too, because adding a direction divides a
        # fixed total among more of them: four directions at a total of 2 push
        # each one less hard than three did, and the push is what holds the text
        # together against the perturbation. Restoring the strength per
        # direction is a different lever from re-splitting it.
        # An explicit --steer-budget wins over --budget-sweep's default, which is
        # a non-empty list and so would otherwise always override it. The run
        # that found this asked for one budget and swept three.
        explicit = getattr(args, "steer_budget", None)
        swept = getattr(args, "budget_sweep", None)
        budgets = ([float(explicit)] if explicit
                   else [float(x) for x in (swept or [b])])

        items = []
        for keep in tails:
            for gam in gammas:
              for b in budgets:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=gam, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, prompt_tail_clear=keep,
                    offset_scale=getattr(args, "offset_scale", None),
                    offset_draw_shape=getattr(args, "offset_draw_shape", "sphere"),
                    offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            "the constraint push at "
            + ", ".join(f"{x:g}" for x in budgets)
            + " at both sitings with a per-story perturbation at "
            + ", ".join(f"{x:g}" for x in gammas)
            + ", leaving the last "
            + ", ".join(str(t) for t in tails) + " prompt positions unperturbed")

    if name == "asymmetric":
        # The constraint push given a different strength at the prompt from the
        # one it has while writing.
        #
        # Round 27 measured that the two sitings do different jobs. Pushing
        # while writing is what enforces a sustained property: at total strength
        # 2 it takes present-tense finite verbs from 13% to 94%, and costs
        # sentence length. Pushing at the prompt and leaving decoding alone does
        # none of that -- the tense share stays where the untouched model leaves
        # it -- but takes variety of wording from 9.7 to 16.7, near what raising
        # the sampling temperature to 1.8 reaches, and leaves sentence length
        # alone.
        #
        # A constant offset is the same for every story, so that variety is not
        # per-story variation. The reading is that the shifted prompt state is
        # one the model is less practised at continuing, so it continues it less
        # predictably. If that holds, the prompt strength is a diversity knob
        # and the writing strength is a compliance knob, and they should be set
        # separately. Every run before this tied them together.
        #
        # The writing strength is held at the value that measured best and the
        # prompt multiplier is swept.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        gains = [float(x) for x in (getattr(args, "prefill_gains", None) or (2.0, 4.0))]
        g = float(getattr(args, "main_gamma", 0.1))

        items = []
        for pg in gains:
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            steer_prefill=True, prefill_gain=pg,
                                            **quiet, **base)})
            items.append({"plan": make_plan(
                beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                offset_basis=offset_basis, offset_basis_kind=kind,
                steer_prefill=True, prefill_gain=pg,
                offset_prefill=True, offset_decode=False, **quiet, **base)})
        return items, (
            f"the constraint push held at {b:g} while writing and applied to the "
            f"prompt at {', '.join(f'{x:g}' for x in gains)} times that strength, "
            f"each alone and with a per-story perturbation at {g:g}")

    if name == "prefill":
        # The same crossing as `frontier`, but with the constraint push added to
        # the prompt positions during prefill as well as at every decode step.
        #
        # Why: the push is normally added at decode steps only, so the opening
        # sentence is written before it has taken hold. 68% of stories at total
        # strength 2 open in the past tense and switch to the present
        # immediately after -- the first present-tense sentence is the fourth at
        # strength 1, the second at strength 2, the first at strength 3.
        #
        # This is not the prompt-only siting, which was run and does something
        # different: pushing at the prompt and leaving decoding alone gives no
        # tense compliance at all (12% of finite verbs present, against the
        # untouched model's 13%) while lifting variety of wording to 16.7
        # against 9.7. Prompt-only is a diversity mechanism. This arm asks
        # whether having both sitings at once keeps the compliance that decoding
        # supplies and picks up the opening the prefill fixes.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        kind = getattr(args, "offset_basis_kind", "story")
        flat = {n: 1.0 for n in names}
        budgets = list(getattr(args, "budget_sweep", [2.0]))
        gammas = list(getattr(args, "gamma_sweep", [0.1]))

        items = []
        for b in budgets:
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            steer_prefill=True, **quiet, **base)})
            for g in gammas:
                items.append({"plan": make_plan(
                    beta=flat, steer_budget=b, offset_gamma=g, offset_mode="orth",
                    offset_basis=offset_basis, offset_basis_kind=kind,
                    steer_prefill=True, offset_prefill=True, offset_decode=False,
                    **quiet, **base)})
        return items, (
            f"the constraint push at {budgets}, added to the prompt as well as "
            f"to every decode step, alone and crossed with a per-story "
            f"perturbation at gamma {gammas}")

    if name == "dropone":
        # Which of the steered directions costs the reading level.
        #
        # The push buys tense, speech and a named character and pays for them in
        # sentence length: at a total strength of 2 the share of stories
        # reaching the grade-3 floor falls from 57% to 12% while words per
        # sentence fall from 9.3 to 6.7. A push along random directions of the
        # same length does neither -- it leaves sentence length alone and lifts
        # the floor -- so this is something the extracted directions carry, not
        # something any offset of that size does.
        #
        # Every arm holds the same total strength, so dropping a direction gives
        # the remaining ones more of it rather than pushing less hard overall.
        # That is the comparison that matters: what the set would be without
        # this member, not what a weaker push does.
        #
        # `loo` already exists and is not this: it sweeps the old per-token
        # noise path at a per-direction beta, with no fixed budget.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 2.0)

        items = [{"plan": make_plan(beta={n: 1.0 for n in names}, steer_budget=b,
                                    steer_prefill=False, **quiet, **base)}]
        for drop in names:
            kept = [n for n in names if n != drop]
            sub = dict(base)
            sub["names"] = kept
            items.append({"plan": make_plan(beta={n: 1.0 for n in kept},
                                            steer_budget=b, steer_prefill=False,
                                            **quiet, **sub)})
        return items, (
            f"all {len(names)} directions at a total strength of {b:g}, then the "
            f"same strength with each one left out in turn")

    if name == "siting":
        # Where the constraint push is applied, rather than how hard.
        #
        # Every run so far has added the constraint vector at each decode step,
        # so the pull toward the directions acts on every token the model
        # writes. On the middle-school task that accumulates into telegraphic
        # prose: sentences containing no speech at all fall from 9.4 words to
        # 5.4, the story goes from about 20 sentences to about 44, and the share
        # of stories reaching the reading floor falls from 57% to 4%.
        #
        # Applying it during prefill only sets the state the model starts from
        # and then leaves it alone. Properties decided once -- what tense the
        # narration is in, whether a character gets a name -- should survive
        # that; a continuous compression of sentence structure should not.
        # `METHODS_TRIED.md` records the push applied to the prompt *as well as*
        # the story, which is the opposite arm, and never this one.
        #
        # Both sitings appear with and without the per-story perturbation, so a
        # difference cannot be read as the perturbation's doing.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        flat = {n: 1.0 for n in names}
        b = float(getattr(args, "steer_budget", None) or 2.0)
        g = float(getattr(args, "main_gamma", 0.1))
        off = dict(offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                   offset_basis_kind=getattr(args, "offset_basis_kind", "story"),
                   offset_prefill=True, offset_decode=False)
        writing = dict(steer_prefill=False, steer_decode=True)
        prompt_only = dict(steer_prefill=True, steer_decode=False)

        items = ["baseline"]
        for siting in (writing, prompt_only):
            items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                            **siting, **quiet, **base)})
            items.append({"plan": make_plan(beta=flat, steer_budget=b, **off,
                                            **siting, **quiet, **base)})
        # the perturbation with no push at all, so both sitings have a floor
        items.append({"plan": make_plan(beta={n: 0.0 for n in names}, **off,
                                        **writing, **quiet, **base)})
        return items, (
            f"where the push at {b:g} is applied: while writing against at the "
            f"prompt only, each with and without the per-story perturbation at "
            f"{g:g}, plus the perturbation alone and the untouched model")

    if name == "core4":
        # The four arms that anchor any configuration change (a different layer
        # band, a different model): nothing, the push, the perturbation, both.
        base = {k: v for k, v in common.items() if k != "steer_prefill"}
        quiet = dict(noise_mode="none", noise_alpha=0.0)
        b = float(getattr(args, "steer_budget", None) or 3.0)
        g = float(getattr(args, "main_gamma", 0.15))
        flat = {n: 1.0 for n in names}
        zero = {n: 0.0 for n in names}
        off = dict(offset_gamma=g, offset_mode="orth", offset_basis=offset_basis,
                   offset_basis_kind=getattr(args, "offset_basis_kind", "story"),
                   offset_prefill=True, offset_decode=False)
        items = ["baseline"]
        items.append({"plan": make_plan(beta=flat, steer_budget=b,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=zero, **off,
                                        steer_prefill=False, **quiet, **base)})
        items.append({"plan": make_plan(beta=flat, steer_budget=b, **off,
                                        steer_prefill=False, **quiet, **base)})
        return items, (
            f"the four anchor arms: baseline, the push at {b:g}, the per-story "
            f"perturbation at {g:g} alone, and both together")

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

    if name == "vstopp":
        # The two raised-temperature baselines and nothing else.
        #
        # `sampling` sweeps a grid and `compare` gives the untouched model plus a
        # noise arm; neither is what a head-to-head against the decoding
        # baselines needs, and on an 8B model a condition costs about four
        # minutes a story, so generating arms that will not be used is the
        # difference between a run that finishes and one that does not.
        t = float(getattr(args, "baseline_temperature", 1.8) or 1.8)
        return ([{"temperature": t, "top_p": float(getattr(args, "baseline_top_p", 0.95) or 0.95)},
                 {"temperature": t, "top_k": int(getattr(args, "baseline_top_k", 40) or 40)}],
                f"raised temperature ({t:g}) with nucleus and with top-k, and nothing else")

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
