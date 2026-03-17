from __future__ import annotations
from . import prompts
from .EGRA_functions import EGRA
from .egra_constraint_checker import EGRAConstraintChecker
from .creativity_metrics import CreativityScorer
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
    use_residual_noise: bool = False
    use_attention_output_noise: bool = False
    use_attention_entropy_noise: bool = False

    residual_layers: Optional[Sequence[int]] = None
    residual_noise_std: float = 0.0
    residual_noise_decay: float = 0.0

    attention_layers: Optional[Sequence[int]] = None
    attention_noise_std: float = 0.0

    attn_entropy_layers: Optional[Sequence[int]] = None
    attn_entropy_noise_std: float = 0.0
    entropy_calc: str = "max_weight"   # one of: "max_weight", "topk_entropy", "gini", "renyi2"
    top_k_size: int = 10               # only used when entropy_calc="topk_entropy"

    max_noise_tokens: int = 200

    logits_noise_std: float = 0.0
    logits_noise_decay: float = 0.0
    max_new_tokens_plan: int = 500
    max_new_tokens_story: int = 500
    do_sample: bool = True
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None


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
    active = sum([
        spec.use_residual_noise,
        spec.use_attention_output_noise,
        spec.use_attention_entropy_noise,
    ])
    if active > 1:
        raise ValueError("ExperimentSpec cannot enable more than one noise mode at a time.")
    if spec.use_attention_entropy_noise:
        return "attention_entropy_noise"
    if spec.use_attention_output_noise:
        return "attention_output_noise"
    if spec.use_residual_noise:
        return "residual_stream_noise"
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

    if not parts:
        return ""
    return "__" + "__".join(parts)


def _spec_to_run_id(model_name: str, spec: ExperimentSpec) -> str:
    mode = _spec_mode(spec)
    sampling_tag = _sampling_tag(spec)

    if mode == "baseline":
        return f"{model_name}__BASELINE{sampling_tag}"

    if mode == "residual_stream_noise":
        parts = [
            f"{model_name}__{_layers_tag(spec.residual_layers)}"
            f"__std{_float_tag(spec.residual_noise_std)}"
            f"__decay{_float_tag(spec.residual_noise_decay)}"
        ]
    elif mode == "attention_output_noise":
        parts = [
            f"{model_name}__ATTN__{_layers_tag(spec.attention_layers)}"
            f"__std{_float_tag(spec.attention_noise_std)}"
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
) -> dict[str, list[str]]:
    """
    Auto filenames: {run_id}.csv where run_id encodes model + spec.
    Returns: {run_id: [story_text, ...]}.
    """
    checker = EGRAConstraintChecker()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ids: list[str] = [_spec_to_run_id(model_name, s) for s in specs]
    out_paths: dict[str, Path] = {rid: out_dir / f"{rid}.csv" for rid in run_ids}

    outputs: dict[str, list[str]] = {rid: [] for rid in run_ids}

    # ---- EXPERIMENT SUMMARY ----
    print("\n================ RUN CONFIGURATION ================\n")
    for spec, rid in zip(specs, run_ids):
        mode = _spec_mode(spec)
        print(f"RUN ID: {rid}")
        if mode == "baseline":
            print("  type: BASELINE (no noise)")
        elif mode == "residual_stream_noise":
            print("  type: RESIDUAL NOISE")
            print(f"  residual_layers: {list(spec.residual_layers or [])}")
            print(f"  residual_noise_std: {spec.residual_noise_std}")
            print(f"  residual_noise_decay: {spec.residual_noise_decay}")
            print(f"  logits_noise_std: {spec.logits_noise_std}")
            print(f"  logits_noise_decay: {spec.logits_noise_decay}")
        elif mode == "attention_output_noise":
            print("  type: ATTENTION OUTPUT NOISE")
            print(f"  attention_layers: {list(spec.attention_layers or [])}")
            print(f"  attention_noise_std: {spec.attention_noise_std}")
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

            if mode == "residual_stream_noise":
                story_text = model.generate_with_residual_stream_noise(
                    story_prompt,
                    residual_layers=list(spec.residual_layers or []),
                    residual_noise_std=spec.residual_noise_std,
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

    buf = io.StringIO()
    with redirect_stdout(buf):
        for rid in run_ids:
            stories = outputs[rid]
            print(f"\n\n---- {rid} ----\n")
            CreativityScorer(stories).creativity_score(print_report=True)

            print(f"\n\n------------------ {rid} CONSTRAINT ---------------------\n\n\n")
            checker.print_report(stories)

    results_path.write_text(buf.getvalue(), encoding="utf-8")

    return outputs


def make_specs(*items: Any) -> list[ExperimentSpec]:
    """
    Accepts:
      - "zero_shot"/"baseline"
      - dicts describing:
          * baseline / zero_shot sampling
          * residual stream noise
          * attention output noise
          * attention entropy noise
        Mode aliases accepted via "generator"/"mode"/"type"/"kind"/"method"/
        "function", including:
          * "baseline", "zero_shot", "generate"
          * "residual_stream_noise", "residual", "generate_with_residual_stream_noise"
          * "attention_output_noise", "attention", "attn",
            "generate_with_attention_output_noise"
          * "attention_entropy_noise", "attention_entropy", "attn_entropy"
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
            "residual": "residual_stream_noise",
            "residual_noise": "residual_stream_noise",
            "residual_stream_noise": "residual_stream_noise",
            "generate_with_residual_stream_noise": "residual_stream_noise",
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
        }
        if key not in aliases:
            raise ValueError(f"Unsupported experiment mode: {value}")
        return aliases[key]

    def _detect_mode(mapping: Mapping[str, Any]) -> str:
        residual_flag = bool(mapping.get("use_residual_noise", False))
        attention_flag = bool(mapping.get("use_attention_output_noise", False))
        att_entropy_flag = bool(mapping.get("use_attention_entropy_noise", False))

        active = sum([residual_flag, attention_flag, att_entropy_flag])
        if active > 1:
            raise ValueError("Spec mapping cannot enable more than one noise mode at a time.")

        if att_entropy_flag:
            return "attention_entropy_noise"
        if attention_flag:
            return "attention_output_noise"
        if residual_flag:
            return "residual_stream_noise"

        mode_value = _first_present(
            mapping,
            "generator", "generation_mode", "mode", "type", "kind", "method", "function",
        )
        if mode_value is not None:
            return _normalize_mode(str(mode_value))

        # Auto-detect from keys present
        if _first_present(
            mapping,
            "attention_entropy_layers", "attn_entropy_layers", "attn_entropy_noise_std",
            "entropy_calc",
        ) is not None:
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

        return "baseline"

    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    for it in items:
        if isinstance(it, str):
            mode = _normalize_mode(it)
            if mode != "baseline":
                raise ValueError(
                    f"String spec '{it}' is missing parameters. "
                    "Use a mapping for noise experiments."
                )
            specs.append(ExperimentSpec())
            continue

        if isinstance(it, Mapping):
            mode = _detect_mode(it)
            specs.append(
                ExperimentSpec(
                    use_residual_noise=(mode == "residual_stream_noise"),
                    use_attention_output_noise=(mode == "attention_output_noise"),
                    use_attention_entropy_noise=(mode == "attention_entropy_noise"),
                    residual_layers=_first_present(it, "residual_layers"),
                    residual_noise_std=float(it.get("residual_noise_std", 0.0)),
                    residual_noise_decay=float(it.get("residual_noise_decay", 0.0)),
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
                    max_noise_tokens=int(it.get("max_noise_tokens", 200)),
                    logits_noise_std=float(it.get("logits_noise_std", 0.0)),
                    logits_noise_decay=float(it.get("logits_noise_decay", 0.0)),
                    max_new_tokens_plan=int(it.get("max_new_tokens_plan", 1000)),
                    max_new_tokens_story=int(it.get("max_new_tokens_story", 500)),
                    do_sample=bool(it.get("do_sample", True)),
                    temperature=float(it.get("temperature", 1.0)),
                    top_p=_optional_float(it.get("top_p")),
                    top_k=_optional_int(it.get("top_k")),
                )
            )
            continue

        raise TypeError(f"Unsupported spec type: {type(it)}")
    return specs