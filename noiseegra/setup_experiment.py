from __future__ import annotations
from . import prompts
from .EGRA_functions import EGRA
from .egra_constraint_checker import EGRAConstraintChecker
from .creativity_metrics import CreativityScorer
from .constraint_metrics import ExactConstraintChecker
import csv
import gc
from dataclasses import dataclass, field
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

@dataclass(frozen=True)
class ExperimentSpec:
    use_two_stage_zero_shot: bool = False
    use_two_stage_residual_noise: bool = False
    use_double_residual_noise: bool = False
    use_residual_noise: bool = False
    use_attention_output_noise: bool = False
    use_attention_entropy_noise: bool = False
    use_residual_and_entropy_noise: bool = False
    use_embedding_noise: bool = False
    use_orthogonal_steering: bool = False

    # Prebuilt noiseegra.subspace.SteeringPlan (carries layers, directions,
    # protected basis, betas, schedules and the noise arm).
    steering_plan: Optional[Any] = None

    residual_layers: Optional[Sequence[int]] = None
    residual_noise_std: float = 0.0
    residual_noise_std_stage2: float = 0.0
    residual_noise_decay: float = 0.0
    disable_residual_noise_decay: bool = False

    attention_layers: Optional[Sequence[int]] = None
    attention_noise_std: float = 0.0

    attn_entropy_layers: Optional[Sequence[int]] = None
    attn_entropy_noise_std: float = 0.0
    entropy_calc: str = "max_weight"   # one of: "max_weight", "topk_entropy", "gini", "renyi2"
    top_k_size: int = 10               # only used when entropy_calc="topk_entropy"

    embed_noise_std: float = 0.0

    max_noise_tokens: int = 200

    logits_noise_std: float = 0.0
    logits_noise_decay: float = 0.0
    max_new_tokens_plan: int = 500
    max_new_tokens_story: int = 500
    do_sample: bool = True
    include_sys: bool = True
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # The published truncation schemes, for comparison arms.
    typical_p: Optional[float] = None      # Meister et al., TACL 2023
    min_p: Optional[float] = None          # Nguyen et al., ICLR 2025
    eta_cutoff: Optional[float] = None     # Hewitt et al., EMNLP Findings 2022
    penalty_alpha: Optional[float] = None  # Su et al., NeurIPS 2022
    # A published method run as a comparison (noiseegra.prior_methods): its name,
    # and its settings as sorted (key, value) pairs so the spec stays hashable.
    prior_method: Optional[str] = None
    prior_params: tuple = ()


def _seed_for_story(x: int) -> int:
    return 42 * (x**7) * 217


def _append_csv(path: Path, row: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(row))


def _make_plan_prompt():
    return [
        {"role": "system", "content": prompts.SYS_NOISE},
        {"role": "user", "content": prompts.NOISE_1},
    ]


def _make_story_prompt(plan_text: str):
    return [
        {"role": "system", "content": prompts.SYS_NOISE},
        {"role": "user", "content": prompts.NOISE_3.replace("[STORY PLAN]", plan_text)},
    ]


def _story_prompt():
    return [
        {"role": "system", "content": prompts.SYS_ZERO_SHOT},
        {"role": "user", "content": prompts.PROMPT_ZERO_SHOT},
    ]


def _layers_tag(layers: Optional[Sequence[int]]) -> str:
    if not layers:
        return "Lnone"
    layers = list(layers)
    return f"L{min(layers)}-{max(layers)}"


def _float_tag(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def _spec_mode(spec: ExperimentSpec) -> str:
    if spec.prior_method:
        if spec.use_orthogonal_steering or spec.steering_plan is not None:
            raise ValueError("a prior-method spec cannot also carry a steering plan")
        return "prior_method"
    active = sum([
        spec.use_two_stage_zero_shot,
        spec.use_two_stage_residual_noise,
        spec.use_double_residual_noise,
        spec.use_residual_noise,
        spec.use_attention_output_noise,
        spec.use_attention_entropy_noise,
        spec.use_residual_and_entropy_noise,
        spec.use_embedding_noise,
        spec.use_orthogonal_steering,
    ])
    if active > 1:
        raise ValueError("ExperimentSpec cannot enable more than one noise mode at a time.")
    if spec.use_orthogonal_steering:
        return "orthogonal_steering"
    if spec.use_double_residual_noise:
        return "double_residual_noise"
    if spec.use_two_stage_residual_noise:
        return "two_stage_residual_noise"
    if spec.use_two_stage_zero_shot:
        return "two_stage_zero_shot"
    if spec.use_residual_and_entropy_noise:
        return "residual_and_entropy_noise"
    if spec.use_attention_entropy_noise:
        return "attention_entropy_noise"
    if spec.use_attention_output_noise:
        return "attention_output_noise"
    if spec.use_residual_noise:
        return "residual_stream_noise"
    if spec.use_embedding_noise:
        return "embedding_noise"
    return "baseline"


def _sampling_tag(spec: ExperimentSpec) -> str:
    parts: list[str] = []

    if not spec.do_sample:
        parts.append("greedy")
    else:
        if spec.temperature != 1.0:
            parts.append(f"temp{_float_tag(spec.temperature)}")
        if spec.top_p is not None:
            parts.append(f"topp{_float_tag(spec.top_p)}")
        if spec.top_k is not None:
            parts.append(f"topk{spec.top_k}")
        if spec.typical_p is not None:
            parts.append(f"typ{_float_tag(spec.typical_p)}")
        if spec.min_p is not None:
            parts.append(f"minp{_float_tag(spec.min_p)}")
        if spec.eta_cutoff is not None:
            parts.append(f"eta{_float_tag(spec.eta_cutoff)}")
    if spec.penalty_alpha is not None:
        parts.append(f"cs{_float_tag(spec.penalty_alpha)}k{spec.top_k}")

    if not parts:
        return ""
    return "__" + "__".join(parts)


_SCHEDULE_CODE = {"constant": "c", "cosine_decay": "d", "ramp": "r", "linear_decay": "l",
                  "prefix": "p"}


# A file name cannot exceed 255 bytes on any filesystem this runs on, and the
# stories go in `<run_id>.csv`. Well under it, to leave room for the suffixes a
# future mechanism adds.
_MAX_RUN_ID = 200


def _ortho_tag(model_name: str, spec: ExperimentSpec) -> str:
    """Encode a SteeringPlan into a filesystem-safe, ablation-distinguishing run id.

    Spelling out every direction and every weight is readable at five
    directions and impossible at thirteen. Measured weights are fractions like
    0.28125, not ones and zeros, so thirteen of them run to eighty characters
    and the name reaches 282 -- past the limit, and only for the arms whose
    weights are fractions. The first arm generated fine and the second could not
    open its file, so an eight-arm run spent an hour to return one arm.

    So the id is built, measured, and rebuilt compressed if it does not fit. The
    compressed form digests the full (name, weight) pairs, so two arms differing
    in any weight still differ here, and nothing is lost: the run's state.json
    records every direction and every beta by name.
    """
    for compress in (False, True):
        tag = _ortho_tag_once(model_name, spec, compress=compress)
        if len(tag) <= _MAX_RUN_ID:
            return tag
    return tag


def _ortho_tag_once(model_name: str, spec: ExperimentSpec, *, compress: bool) -> str:
    plan = spec.steering_plan
    if plan is None:
        raise ValueError("orthogonal_steering specs require a `steering_plan`.")

    if compress:
        import hashlib
        full = ";".join(f"{sp.name}={sp.beta!r}" for sp in plan.specs)
        names = f"{len(plan.specs)}dir"
        betas = "w" + hashlib.sha1(full.encode()).hexdigest()[:10]
    else:
        names = "-".join(s.name[:3] for s in plan.specs)
        betas = "-".join(_float_tag(s.beta) for s in plan.specs)
    parts = [
        f"{model_name}__ORTHO",
        f"__{_layers_tag(plan.layers)}",
        f"__C{names}",
        f"__b{betas}",
        f"__{plan.orthogonalize}",
        f"__nz{plan.noise_mode}",
        f"__a{_float_tag(plan.noise_alpha)}",
        f"__k{plan.protect_rank}",
    ]
    if getattr(plan, "offset_gamma", 0) and plan.offset_mode != "none":
        parts.append(f"__g{_float_tag(plan.offset_gamma)}{plan.offset_mode}")
        if getattr(plan, "offset_basis_kind", "step") != "step":
            parts.append(f"__ob{plan.offset_basis_kind}")
        if getattr(plan, "offset_prefill", False):
            parts.append("__opre")
        if not getattr(plan, "offset_decode", True):
            parts.append("__ponly")
        if getattr(plan, "offset_norm", "energy") != "energy":
            parts.append(f"__on{plan.offset_norm}")
        if getattr(plan, "offset_draw", "iid") != "iid":
            parts.append(f"__od{plan.offset_draw}")
    if getattr(plan, "steer_budget", None):
        parts.append(f"__bud{_float_tag(plan.steer_budget)}")
    # How the coefficient is decided, not just how large it is. Without this a
    # constant arm and an error-driven arm at the same betas share a run id and
    # overwrite each other's stories, which is a silent wrong answer rather than
    # a crash.
    if getattr(plan, "steer_mode", "constant") != "constant":
        parts.append(f"__sm{plan.steer_mode}")
    if getattr(plan, "jitter_mode", "none") != "none" and getattr(plan, "jitter_kappa", 0):
        parts.append(f"__j{_float_tag(plan.jitter_kappa)}{plan.jitter_mode}")
        if getattr(plan, "jitter_draw", "iso") != "iso":
            parts.append(f"__jd{plan.jitter_draw}")
        # Which directions the gain varies, when it is not all of them. Two arms
        # varying different directions are different arms.
        names = list(getattr(plan, "jitter_names", ()) or ())
        if names:
            parts.append("__jn" + "-".join(n[:4] for n in sorted(names)))
        if getattr(plan, "jitter_walk", 0.0):
            parts.append(f"__jw{_float_tag(plan.jitter_walk)}")
    if not getattr(plan, "steer_decode", True):
        parts.append("__sdec0")
    if getattr(plan, "direction_source", "extracted") != "extracted":
        parts.append(f"__dir{plan.direction_source}")
    if getattr(plan, "amplify_lambda", 1.0) != 1.0:
        parts.append(f"__amp{_float_tag(plan.amplify_lambda)}")
        if getattr(plan, "amplify_prefill", False):
            parts.append("__apre")
    if getattr(plan, "gate_level", "none") not in ("none", None):
        parts.append(f"__gate{plan.gate_level}")
    schedules = [s.schedule for s in plan.specs]
    if any(sc != "constant" for sc in schedules):
        parts.append("__sch" + "-".join(_SCHEDULE_CODE.get(sc, sc[:1]) for sc in schedules))
    if plan.noise_norm_match != "energy":
        parts.append(f"__nm{plan.noise_norm_match}")
    if plan.noise_schedule != "constant":
        parts.append(f"__nsch{_SCHEDULE_CODE.get(plan.noise_schedule, plan.noise_schedule[:1])}")
        if getattr(plan, "noise_horizon", None):
            parts.append(f"__nh{int(plan.noise_horizon)}")
    # Which layers each half acts on, when they differ. Without this in the id,
    # a run pushing at one band and perturbing at another shares an id with one
    # that does both over the same band, and they overwrite each other.
    ol = sorted(getattr(plan, "offset_layers", ()) or ())
    pl = sorted(getattr(plan, "push_layers", ()) or ())
    if ol and ol != pl:
        parts.append(f"__ol{ol[0]}-{ol[-1]}")
    if pl and ol and pl != ol:
        parts.append(f"__pl{pl[0]}-{pl[-1]}")
    # Directions the perturbation is held clear of beyond the steered ones
    # themselves. Without this in the id a shielded arm and an unshielded one
    # share a name and the second overwrites the first, which is the fault that
    # produced byte-identical arms three times on this project. Named, not
    # counted: the first version of this counted the protected subspace's rank,
    # which the steered constraints' principal components also enlarge, so it
    # printed the same thing on a shielded arm and on its unshielded control.
    shielded = list(getattr(plan, "shield_names", ()) or ())
    if shielded:
        parts.append("__sh" + "-".join(n[:4] for n in sorted(shielded)))
        # How wide the shield is, not only what it is named after. A one-axis
        # shield and a subspace shield of the same name are different arms.
        rank = int(getattr(plan, "shield_rank", 0) or 0)
        if rank:
            parts.append(f"r{rank}")
    if getattr(plan, "guard_direction", ""):
        parts.append(f"__guard{plan.guard_direction[:4]}")
    if float(getattr(plan, "offset_taper", 1.0) or 1.0) != 1.0:
        parts.append(f"__tap{_float_tag(plan.offset_taper)}")
    if getattr(plan, "offset_gamma_spread", 0.0):
        parts.append(f"__gs{_float_tag(plan.offset_gamma_spread)}")
    if getattr(plan, "offset_draw_shape", "sphere") != "sphere":
        parts.append(f"__od{plan.offset_draw_shape}")
    if int(getattr(plan, "offset_random_rank", 0) or 0) > 0:
        parts.append(f"__rr{int(plan.offset_random_rank)}")
    if getattr(plan, "shadow_protect", False):
        parts.append("__shadow")
    if float(getattr(plan, "offset_online", 0.0) or 0.0) > 0:
        parts.append(f"__online{_float_tag(plan.offset_online)}")
        if float(getattr(plan, "online_max_gain", 2.5) or 2.5) != 2.5:
            parts.append(f"max{_float_tag(plan.online_max_gain)}")
        if getattr(plan, "online_carry", False):
            parts.append("carry")
        if float(getattr(plan, "online_rule_k", 0.0) or 0.0) > 0:
            parts.append(f"rule{_float_tag(plan.online_rule_k)}"
                         + ("start" if getattr(plan, "online_rule_start", False) else ""))
    if getattr(plan, "offset_measured", False):
        parts.append(f"__meas{_float_tag(getattr(plan, 'online_rule_k', 0.0) or 0.0)}")
    if float(getattr(plan, "output_tilt", 0.0) or 0.0) > 0:
        parts.append(f"__otilt{_float_tag(plan.output_tilt)}")
    if float(getattr(plan, "offset_prefill_gain", 1.0) or 1.0) != 1.0:
        parts.append(f"__opg{_float_tag(plan.offset_prefill_gain)}")
    if float(getattr(plan, "steer_split_concentration", 0.0) or 0.0) > 0:
        parts.append(f"__split{_float_tag(plan.steer_split_concentration)}")
    if getattr(plan, "arch_mechanism", "") and float(getattr(plan, "arch_size", 0.0) or 0.0) > 0:
        parts.append(f"__arch{plan.arch_mechanism}{_float_tag(plan.arch_size)}")
        if getattr(plan, "arch_per_sentence", False):
            parts.append("ps")
    # The colour of the displacement's wandering, and the slowest wobble it is
    # allowed. Both change what is generated, so both have to be in the id: two
    # arms differing only in a setting the id does not carry write to the same
    # file, and the second silently replaces the first.
    if getattr(plan, "noise_beta", None) is not None:
        parts.append(f"__cn{_float_tag(plan.noise_beta)}")
        cyc = float(getattr(plan, "noise_fmin_cycles", 0.25) or 0.25)
        if cyc != 0.25:
            parts.append(f"__cyc{_float_tag(cyc)}")
    if str(getattr(plan, "offset_envelope", "flat") or "flat") != "flat":
        parts.append(f"__env{plan.offset_envelope}")
        if int(getattr(plan, "offset_envelope_steps", 0) or 0) > 0:
            parts.append(f"{int(plan.offset_envelope_steps)}")
        if float(getattr(plan, "offset_secured_boost", 0.0) or 0.0) > 0:
            parts.append(f"{_float_tag(plan.offset_secured_boost)}")
        if float(getattr(plan, "offset_front_gain", 1.0) or 1.0) != 1.0:
            parts.append(f"g{_float_tag(plan.offset_front_gain)}")
    if getattr(plan, "offset_decode_steps", 0):
        parts.append(f"__ods{int(plan.offset_decode_steps)}")
    if getattr(plan, "prompt_head_clear", 0):
        parts.append(f"__head{int(plan.prompt_head_clear)}")
    if getattr(plan, "prompt_tail_clear", 0):
        parts.append(f"__tail{int(plan.prompt_tail_clear)}")
    if plan.steer_prefill:
        parts.append("__prefill")
        # The prompt siting's own strength, as a multiple of the writing
        # strength. Without it in the id, two runs pushing the prompt at
        # different strengths share a run id and overwrite each other.
        if getattr(plan, "prefill_gain", 1.0) != 1.0:
            parts.append(f"__pg{_float_tag(plan.prefill_gain)}")
    return "".join(parts)


def _spec_to_run_id(model_name: str, spec: ExperimentSpec) -> str:
    mode = _spec_mode(spec)
    sampling_tag = _sampling_tag(spec)

    if mode == "baseline":
        return f"{model_name}__BASELINE{sampling_tag}"

    if mode == "prior_method":
        params = "".join(f"__{k}{_float_tag(v) if isinstance(v, (int, float)) else v}"
                         for k, v in spec.prior_params)
        return f"{model_name}__PRIOR__{spec.prior_method}{params}{sampling_tag}"

    if mode == "orthogonal_steering":
        return _ortho_tag(model_name, spec) + sampling_tag

    if mode == "two_stage_zero_shot":
        parts = [
            f"{model_name}__TWOSTAGE_ZERO"
        ]
        if not spec.include_sys:
            parts.append("__nosys")
    elif mode == "two_stage_residual_noise":
        parts = [
            f"{model_name}__TWOSTAGE_RESID"
            f"__{_layers_tag(spec.residual_layers)}"
            f"__std{_float_tag(spec.residual_noise_std)}"
            f"__decay{_float_tag(spec.residual_noise_decay)}"
        ]
        if not spec.include_sys:
            parts.append("__nosys")
    elif mode == "double_residual_noise":
        parts = [
            f"{model_name}__DOUBLE_RESID"
            f"__{_layers_tag(spec.residual_layers)}"
            f"__std1{_float_tag(spec.residual_noise_std)}"
            f"__std2{_float_tag(spec.residual_noise_std_stage2)}"
            f"__decay{_float_tag(spec.residual_noise_decay)}"
        ]
        if not spec.include_sys:
            parts.append("__nosys")
    elif mode == "residual_stream_noise":
        parts = [
            f"{model_name}__{_layers_tag(spec.residual_layers)}"
            f"__std{_float_tag(spec.residual_noise_std)}"
            f"__decay{_float_tag(spec.residual_noise_decay)}"
        ]
    elif mode == "residual_and_entropy_noise":
        parts = [
            f"{model_name}__RESID_ENTROPY"
            f"__RES{_layers_tag(spec.residual_layers)}"
            f"__Rstd{_float_tag(spec.residual_noise_std)}"
            f"__Rdecay{_float_tag(spec.residual_noise_decay)}"
            f"__ENT{_layers_tag(spec.attn_entropy_layers)}"
            f"__Estd{_float_tag(spec.attn_entropy_noise_std)}"
        ]
    elif mode == "attention_output_noise":
        parts = [
            f"{model_name}__ATTN__{_layers_tag(spec.attention_layers)}"
            f"__std{_float_tag(spec.attention_noise_std)}"
        ]
    elif mode == "embedding_noise":
        parts = [
            f"{model_name}__EMBED__std{_float_tag(spec.embed_noise_std)}"
        ]
    else:  # attention_entropy_noise
        parts = [
            f"{model_name}__ENTROPY__{_layers_tag(spec.attn_entropy_layers)}"
            f"__std{_float_tag(spec.attn_entropy_noise_std)}"
        ]

    if spec.max_noise_tokens != 200:
        parts.append(f"__maxtok{spec.max_noise_tokens}")
    if spec.logits_noise_std != 0.0:
        parts.append(f"__logstd{_float_tag(spec.logits_noise_std)}")
    if spec.logits_noise_decay != 0.0:
        parts.append(f"__logdecay{_float_tag(spec.logits_noise_decay)}")
    if mode in {
        "two_stage_residual_noise",
        "double_residual_noise",
        "residual_stream_noise",
        "residual_and_entropy_noise",
    } and spec.disable_residual_noise_decay:
        parts.append("__nodecay")
    if sampling_tag:
        parts.append(sampling_tag)

    return "".join(parts)


def run_story_experiments(
    model: Any,
    model_name: str,
    num_stories,
    specs: Sequence[ExperimentSpec],
    *,
    output_dir: str = "results",
    clear_cuda_each_iter: bool = True,
    seed_fn: Optional[Callable[[int], Optional[int]]] = _seed_for_story,
    sanity_check: bool = False,
    sanity_check_n: int = 2,
    spacy_model: Optional[str] = None,
    exact_constraints: bool = True,
) -> dict[str, list[str]]:
    """
    Auto filenames: {run_id}.csv where run_id encodes model + spec.
    Also writes per-run RESULTS/{run_id}.txt and combined {model_name}_RESULTS.txt.
    Returns: {run_id: [story_text, ...]}.
    """
    checker = EGRAConstraintChecker(spacy_model=spacy_model)
    exact_checker = ExactConstraintChecker() if exact_constraints else None
    if spacy_model is None:
        print(
            "NOTE: EGRA name constraints (C14, C17, C18) are not checked without "
            "spacy_model. Install optional deps: pip install -e \".[names]\" and "
            "pass spacy_model=\"en_core_web_sm\" (or your NER model)."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "RESULTS"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_ids: list[str] = [_spec_to_run_id(model_name, s) for s in specs]
    out_paths: dict[str, Path] = {rid: out_dir / f"{rid}.csv" for rid in run_ids}

    outputs: dict[str, list[str]] = {rid: [] for rid in run_ids}

    # ---- EXPERIMENT SUMMARY ----
    print("\n================ RUN CONFIGURATION ================\n")
    for spec, rid in zip(specs, run_ids):
        mode = _spec_mode(spec)
        print(f"RUN ID: {rid}")
        if mode == "orthogonal_steering":
            print("  type: ORTHOGONAL CONSTRAINT STEERING")
            spec.steering_plan.print_report()
        if mode == "baseline":
            print("  type: BASELINE (no noise)")
        elif mode == "two_stage_zero_shot":
            print("  type: TWO-STAGE ZERO SHOT")
            print(f"  include_sys: {spec.include_sys}")
        elif mode == "two_stage_residual_noise":
            print("  type: TWO-STAGE RESIDUAL NOISE")
            print(f"  include_sys: {spec.include_sys}")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std: {spec.residual_noise_std}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  disable_residual_noise_decay: {spec.disable_residual_noise_decay}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "double_residual_noise":
            print("  type: DOUBLE RESIDUAL NOISE")
            print(f"  include_sys: {spec.include_sys}")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std_stage1: {spec.residual_noise_std}")
            print(f"  residual_noise_std_stage2: {spec.residual_noise_std_stage2}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  disable_residual_noise_decay: {spec.disable_residual_noise_decay}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "residual_stream_noise":
            print("  type: RESIDUAL NOISE")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std: {spec.residual_noise_std}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  disable_residual_noise_decay: {spec.disable_residual_noise_decay}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "residual_and_entropy_noise":
            print("  type: RESIDUAL + ATTENTION ENTROPY NOISE")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std: {spec.residual_noise_std}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  disable_residual_noise_decay: {spec.disable_residual_noise_decay}")
            print(f"  attn_entropy_layers: {list(spec.attn_entropy_layers or [])}")
            print(f"  attn_entropy_noise_std: {spec.attn_entropy_noise_std}")
            print(f"  entropy_calc: {spec.entropy_calc}")
            if spec.entropy_calc == "topk_entropy":
                print(f"  top_k_size: {spec.top_k_size}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "attention_output_noise":
            print("  type: ATTENTION OUTPUT NOISE")
            print(f"  attention_layers: {list(spec.attention_layers or [])}")
            print(f"  attention_noise_std: {spec.attention_noise_std}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "embedding_noise":
            print("  type: EMBEDDING NOISE")
            print(f"  embed_noise_std: {spec.embed_noise_std}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        else:  # attention_entropy_noise
            print("  type: ATTENTION ENTROPY NOISE")
            print(f"  attn_entropy_layers: {list(spec.attn_entropy_layers or [])}")
            print(f"  attn_entropy_noise_std: {spec.attn_entropy_noise_std}")
            print(f"  entropy_calc: {spec.entropy_calc}")
            if spec.entropy_calc == "topk_entropy":
                print(f"  top_k_size: {spec.top_k_size}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        print(f"  do_sample: {spec.do_sample}")
        if spec.do_sample:
            print(f"  temperature: {spec.temperature}")
            if spec.top_p is not None:
                print(f"  top_p: {spec.top_p}")
            if spec.top_k is not None:
                print(f"  top_k: {spec.top_k}")
        print()

    print("===================================================\n")

    start_story = num_stories[0]
    end_story = num_stories[1]
    for x in range(start_story, end_story):
        seed = seed_fn(x) if seed_fn is not None else None

        for spec, rid in zip(specs, run_ids):
            story_prompt = _story_prompt()
            mode = _spec_mode(spec)

            if mode == "two_stage_zero_shot":
                if not spec.include_sys:
                    two_stage_prompt = [
                        {"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}
                    ]
                else:
                    two_stage_prompt = [
                        {"role": "system", "content": prompts.SYS_NOISE},
                        {"role": "user", "content": prompts.NOISE_1},
                    ]

                first_stage = model.generate(
                    two_stage_prompt,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )

                two_stage_prompt.append({"role": "assistant", "content": first_stage})
                two_stage_prompt.append({"role": "user", "content": prompts.NOISE_2})

                story_text = model.generate(
                    two_stage_prompt,
                    max_new_tokens=spec.max_new_tokens_story,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )
            elif mode == "two_stage_residual_noise":
                if not spec.include_sys:
                    two_stage_prompt = [
                        {"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}
                    ]
                else:
                    two_stage_prompt = [
                        {"role": "system", "content": prompts.SYS_NOISE},
                        {"role": "user", "content": prompts.NOISE_1},
                    ]

                first_stage = model.generate_with_residual_stream_noise(
                    two_stage_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens=spec.max_noise_tokens,
                    disable_residual_noise_decay=spec.disable_residual_noise_decay,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )

                two_stage_prompt.append({"role": "assistant", "content": first_stage})
                two_stage_prompt.append({"role": "user", "content": prompts.NOISE_2})

                story_text = model.generate(
                    two_stage_prompt,
                    max_new_tokens=spec.max_new_tokens_story,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )
            elif mode == "double_residual_noise":
                if not spec.include_sys:
                    two_stage_prompt = [
                        {"role": "user", "content": prompts.SYS_NOISE + "\n\n\n" + prompts.NOISE_1}
                    ]
                else:
                    two_stage_prompt = [
                        {"role": "system", "content": prompts.SYS_NOISE},
                        {"role": "user", "content": prompts.NOISE_1},
                    ]

                first_stage = model.generate_with_residual_stream_noise(
                    two_stage_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens=spec.max_noise_tokens,
                    disable_residual_noise_decay=spec.disable_residual_noise_decay,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )

                two_stage_prompt.append({"role": "assistant", "content": first_stage})
                two_stage_prompt.append({"role": "user", "content": prompts.NOISE_2})

                story_text = model.generate_with_residual_stream_noise(
                    two_stage_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std_stage2,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens=spec.max_noise_tokens,
                    disable_residual_noise_decay=spec.disable_residual_noise_decay,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_story,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )
            elif mode == "embedding_noise":
                story_text = model.generate_with_embedding_noise(
                    story_prompt,
                    embed_noise_std=spec.embed_noise_std,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    temperature=spec.temperature,
                    seed=seed,
                    max_noise_tokens=spec.max_noise_tokens,
                )
            elif mode == "residual_stream_noise":
                story_text = model.generate_with_residual_stream_noise(
                    story_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens=spec.max_noise_tokens,
                    disable_residual_noise_decay=spec.disable_residual_noise_decay,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )
            elif mode == "residual_and_entropy_noise":
                story_text = model.generate_with_residual_and_entropy_noise(
                    story_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
                    attn_entropy_layers=list(spec.attn_entropy_layers or []),
                    attention_noise_std=spec.attn_entropy_noise_std,
                    entropy_calc=spec.entropy_calc,
                    top_k_size=spec.top_k_size,
                    residual_noise_decay=spec.residual_noise_decay,
                    max_noise_tokens=spec.max_noise_tokens,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                    disable_residual_noise_decay=spec.disable_residual_noise_decay,
                )
            elif mode == "attention_output_noise":
                story_text = model.generate_with_attention_output_noise(
                    story_prompt,
                    attn_layers=list(spec.attention_layers or []),
                    attn_noise_std=spec.attention_noise_std,
                    max_noise_tokens=spec.max_noise_tokens,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    seed=seed,
                )
            elif mode == "orthogonal_steering":
                story_text = model.generate_with_orthogonal_steering(
                    story_prompt,
                    spec.steering_plan,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    typical_p=spec.typical_p,
                    min_p=spec.min_p,
                    eta_cutoff=spec.eta_cutoff,
                    seed=seed,
                )
            elif mode == "attention_entropy_noise":
                story_text = model.generate_with_entropy_noise(
                    story_prompt,
                    attention_noise_std=spec.attn_entropy_noise_std,
                    attn_entropy_layers=list(spec.attn_entropy_layers or []),
                    entropy_calc=spec.entropy_calc,
                    top_k_size=spec.top_k_size,
                    logits_noise_std=spec.logits_noise_std,
                    logits_noise_decay=spec.logits_noise_decay,
                    max_new_tokens=spec.max_new_tokens_plan,
                    temperature=spec.temperature,
                    seed=seed,
                    max_noise_tokens=spec.max_noise_tokens,
                )
            else:
                story_text = model.generate(
                    story_prompt,
                    max_new_tokens=spec.max_new_tokens_plan,
                    do_sample=spec.do_sample,
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=spec.top_k,
                    typical_p=spec.typical_p,
                    min_p=spec.min_p,
                    eta_cutoff=spec.eta_cutoff,
                    penalty_alpha=spec.penalty_alpha,
                    seed=seed,
                )

            outputs[rid].append(story_text)
            _append_csv(out_paths[rid], [story_text])

            if sanity_check and x < sanity_check_n:
                print(f"======== SANITY CHECK: STORY {x} ({rid}) =======\n{story_text}\n=============================================\n")

        if clear_cuda_each_iter:
            gc.collect()
            torch.cuda.empty_cache()

    # scoring + constraints
    results_path = out_dir / f"{model_name}_RESULTS.txt"

    combined_buf = io.StringIO()
    for rid in run_ids:
        stories = outputs[rid]
        run_buf = io.StringIO()
        with redirect_stdout(run_buf):
            print(f"\n\n---- {rid} ----\n")
            CreativityScorer(stories).creativity_score(print_report=True)

            print(f"\n\n------------------ {rid} CONSTRAINT ---------------------\n\n\n")
            checker.print_report(stories)

            if exact_constraints:
                print()
                exact_checker.print_report(stories, label=rid)

        run_text = run_buf.getvalue()
        (results_dir / f"{rid}.txt").write_text(run_text, encoding="utf-8")
        combined_buf.write(run_text)

    results_path.write_text(combined_buf.getvalue(), encoding="utf-8")

    return outputs


def make_specs(*items: Any) -> list[ExperimentSpec]:
    """
    Build experiment specs from mode strings or mapping dictionaries.

    Supported modes: baseline, two_stage_zero_shot,
    two_stage_residual_noise, double_residual_noise, residual_stream_noise,
    residual_and_entropy_noise, attention_output_noise,
    attention_entropy_noise, embedding_noise, orthogonal_steering.

    ``orthogonal_steering`` requires a prebuilt
    :class:`noiseegra.subspace.SteeringPlan` passed as ``steering_plan`` (or
    ``plan``); every other knob for that mode lives on the plan.
    """
    specs: list[ExperimentSpec] = []

    def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def _normalize_mode(value: str) -> str:
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "baseline": "baseline",
            "zero_shot": "baseline",
            "generate": "baseline",
            "two_stage_zero_shot": "two_stage_zero_shot",
            "twostage_zero_shot": "two_stage_zero_shot",
            "twostage_zero": "two_stage_zero_shot",
            "twostage_zeroshot": "two_stage_zero_shot",
            "generate_two_stage_zero_shot": "two_stage_zero_shot",
            "twostage_residual": "two_stage_residual_noise",
            "two_stage_residual": "two_stage_residual_noise",
            "two_stage_residual_noise": "two_stage_residual_noise",
            "twostage_residual_noise": "two_stage_residual_noise",
            "twostage_residualnoise": "two_stage_residual_noise",
            "generate_two_stage_residual_noise": "two_stage_residual_noise",
            "double_residual": "double_residual_noise",
            "double_residual_noise": "double_residual_noise",
            "double_resid": "double_residual_noise",
            "double_res": "double_residual_noise",
            "double_residuals": "double_residual_noise",
            "residual": "residual_stream_noise",
            "residual_noise": "residual_stream_noise",
            "residual_stream_noise": "residual_stream_noise",
            "generate_with_residual_stream_noise": "residual_stream_noise",
            "residual_and_entropy": "residual_and_entropy_noise",
            "residual_and_entropy_noise": "residual_and_entropy_noise",
            "residual_entropy": "residual_and_entropy_noise",
            "residual_entropy_noise": "residual_and_entropy_noise",
            "generate_with_residual_and_entropy_noise": "residual_and_entropy_noise",
            "attention": "attention_output_noise",
            "attention_noise": "attention_output_noise",
            "attention_output": "attention_output_noise",
            "attention_output_noise": "attention_output_noise",
            "attn": "attention_output_noise",
            "attn_noise": "attention_output_noise",
            "generate_with_attention_output_noise": "attention_output_noise",
            "attn_entropy": "attention_entropy_noise",
            "attention_entropy": "attention_entropy_noise",
            "attention_entropy_noise": "attention_entropy_noise",
            "generate_with_entropy_noise": "attention_entropy_noise",
            "embedding": "embedding_noise",
            "embed": "embedding_noise",
            "embedding_noise": "embedding_noise",
            "embed_noise": "embedding_noise",
            "generate_with_embedding_noise": "embedding_noise",
            "ortho": "orthogonal_steering",
            "orthosteer": "orthogonal_steering",
            "ortho_steering": "orthogonal_steering",
            "orthogonal_steering": "orthogonal_steering",
            "steering": "orthogonal_steering",
            "constraint_steering": "orthogonal_steering",
            "generate_with_orthogonal_steering": "orthogonal_steering",
        }
        if key not in aliases:
            raise ValueError(f"Unsupported experiment mode: {value}")
        return aliases[key]

    def _detect_mode(mapping: Mapping[str, Any]) -> str:
        two_stage_zero_flag = bool(mapping.get("use_two_stage_zero_shot", False))
        two_stage_residual_flag = bool(mapping.get("use_two_stage_residual_noise", False))
        double_residual_flag = bool(mapping.get("use_double_residual_noise", False))
        residual_flag = bool(mapping.get("use_residual_noise", False))
        attention_flag = bool(mapping.get("use_attention_output_noise", False))
        att_entropy_flag = bool(mapping.get("use_attention_entropy_noise", False))
        residual_and_entropy_flag = bool(mapping.get("use_residual_and_entropy_noise", False))
        embed_flag = bool(mapping.get("use_embedding_noise", False))
        ortho_flag = bool(mapping.get("use_orthogonal_steering", False)) or (
            _first_present(mapping, "steering_plan", "plan") is not None
        )

        active = sum([
            two_stage_zero_flag,
            two_stage_residual_flag,
            double_residual_flag,
            residual_flag,
            attention_flag,
            att_entropy_flag,
            residual_and_entropy_flag,
            embed_flag,
            ortho_flag,
        ])
        if active > 1:
            raise ValueError("Spec mapping cannot enable more than one noise mode at a time.")

        if ortho_flag:
            return "orthogonal_steering"

        if double_residual_flag:
            return "double_residual_noise"
        if two_stage_residual_flag:
            return "two_stage_residual_noise"
        if two_stage_zero_flag:
            return "two_stage_zero_shot"
        if residual_and_entropy_flag:
            return "residual_and_entropy_noise"
        if att_entropy_flag:
            return "attention_entropy_noise"
        if attention_flag:
            return "attention_output_noise"
        if residual_flag:
            return "residual_stream_noise"
        if embed_flag:
            return "embedding_noise"

        mode_value = _first_present(
            mapping,
            "generator", "generation_mode", "mode", "type", "kind", "method", "function",
        )
        if mode_value is not None:
            return _normalize_mode(str(mode_value))

        # Auto-detect from keys present
        has_entropy_keys = _first_present(
            mapping,
            "attention_entropy_layers", "attn_entropy_layers", "attn_entropy_noise_std",
            "entropy_calc",
        ) is not None

        has_residual_keys = _first_present(
            mapping,
            "residual_layers", "residual_noise_std", "residual_noise_decay",
        ) is not None

        has_double_residual_keys = _first_present(
            mapping,
            "residual_noise_std_stage2", "stage2_residual_noise_std", "second_stage_residual_noise_std",
        ) is not None

        if has_double_residual_keys:
            return "double_residual_noise"

        if has_entropy_keys and has_residual_keys:
            return "residual_and_entropy_noise"

        if has_entropy_keys:
            return "attention_entropy_noise"

        if _first_present(
            mapping,
            "attention_layers", "attention_output_layers", "attn_layers",
            "attention_noise_std", "attention_output_noise_std", "attn_noise_std",
        ) is not None:
            return "attention_output_noise"

        if _first_present(
            mapping,
            "residual_layers", "residual_noise_std", "residual_noise_decay",
        ) is not None:
            return "residual_stream_noise"

        if _first_present(mapping, "embed_noise_std") is not None:
            return "embedding_noise"

        return "baseline"

    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def _resolve_stage_residual_stds(mapping: Mapping[str, Any]) -> tuple[float, float]:
        stage1 = _first_present(
            mapping,
            "residual_noise_std_stage1",
            "stage1_residual_noise_std",
            "residual_noise_std",
        )
        if stage1 is None:
            stage1 = 0.0

        stage2 = _first_present(
            mapping,
            "residual_noise_std_stage2",
            "stage2_residual_noise_std",
            "second_stage_residual_noise_std",
        )
        if stage2 is None:
            stage2 = stage1

        return float(stage1), float(stage2)

    for it in items:
        if isinstance(it, str):
            mode = _normalize_mode(it)
            if mode == "baseline":
                specs.append(ExperimentSpec())
                continue
            if mode == "two_stage_zero_shot":
                specs.append(ExperimentSpec(use_two_stage_zero_shot=True))
                continue
            raise ValueError(
                f"String spec '{it}' is missing parameters. "
                "Use a mapping for noise experiments."
            )

        if isinstance(it, Mapping):
            mode = _detect_mode(it)
            stage1_std, stage2_std = _resolve_stage_residual_stds(it)
            specs.append(
                ExperimentSpec(
                    use_two_stage_zero_shot=(mode == "two_stage_zero_shot"),
                    use_two_stage_residual_noise=(mode == "two_stage_residual_noise"),
                    use_double_residual_noise=(mode == "double_residual_noise"),
                    use_residual_noise=(mode == "residual_stream_noise"),
                    use_attention_output_noise=(mode == "attention_output_noise"),
                    use_attention_entropy_noise=(mode == "attention_entropy_noise"),
                    use_residual_and_entropy_noise=(mode == "residual_and_entropy_noise"),
                    use_embedding_noise=(mode == "embedding_noise"),
                    use_orthogonal_steering=(mode == "orthogonal_steering"),
                    steering_plan=_first_present(it, "steering_plan", "plan"),
                    residual_layers=_first_present(it, "residual_layers"),
                    residual_noise_std=stage1_std,
                    residual_noise_std_stage2=stage2_std,
                    residual_noise_decay=float(it.get("residual_noise_decay", 0.0)),
                    disable_residual_noise_decay=bool(
                        _first_present(
                            it,
                            "disable_residual_noise_decay",
                            "residual_noise_decay_off",
                            "no_residual_noise_decay",
                        )
                        or False
                    ),
                    attention_layers=_first_present(
                        it, "attention_layers", "attention_output_layers", "attn_layers"
                    ),
                    attention_noise_std=float(
                        _first_present(
                            it, "attention_noise_std", "attention_output_noise_std", "attn_noise_std"
                        ) or 0.0
                    ),
                    attn_entropy_layers=_first_present(
                        it, "attention_entropy_layers", "attn_entropy_layers"
                    ),
                    attn_entropy_noise_std=float(
                        _first_present(it, "attn_entropy_noise_std", "attention_entropy_noise_std") or 0.0
                    ),
                    entropy_calc=str(it.get("entropy_calc", "max_weight")),
                    top_k_size=int(it.get("top_k_size", 10)),
                    embed_noise_std=float(it.get("embed_noise_std", 0.0)),
                    max_noise_tokens=int(it.get("max_noise_tokens", 200)),
                    logits_noise_std=float(it.get("logits_noise_std", 0.0)),
                    logits_noise_decay=float(it.get("logits_noise_decay", 0.0)),
                    max_new_tokens_plan=int(it.get("max_new_tokens_plan", 500)),
                    max_new_tokens_story=int(it.get("max_new_tokens_story", 500)),
                    do_sample=bool(it.get("do_sample", True)),
                    include_sys=bool(it.get("include_sys", True)),
                    temperature=float(it.get("temperature", 1.0)),
                    top_p=_optional_float(it.get("top_p")),
                    top_k=_optional_int(it.get("top_k")),
                    typical_p=_optional_float(it.get("typical_p")),
                    min_p=_optional_float(it.get("min_p")),
                    eta_cutoff=_optional_float(it.get("eta_cutoff")),
                    penalty_alpha=_optional_float(it.get("penalty_alpha")),
                    prior_method=it.get("prior_method"),
                    prior_params=tuple(sorted(dict(it.get("prior_params") or {}).items())),
                )
            )
            continue

        raise TypeError(f"Unsupported spec type: {type(it)}")
    return specs